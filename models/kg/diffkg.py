import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_sum, scatter_softmax
from config.configurator import configs
import random
import scipy.sparse as sp
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from models.base_model import BaseModel
import math

init = nn.init.xavier_uniform_

class DiffKG(BaseModel):
	def __init__(self, data_handler):
		super(DiffKG, self).__init__(data_handler)
		self.n_relations = configs['data']['relation_num']
		self.n_entities = configs['data']['entity_num']
		self.context_hops = configs['model']['layer_num_kg']
		self.layer_num = configs['model']['layer_num']
		self.mess_dropout_rate = configs['model']['mess_dropout_rate']
		self.device = configs['device']

		self.reg_weight = configs['model']['reg_weight']
		self.temperature = configs['model']['temperature']
		self.cl_weight = configs['model']['cl_weight']

		self.uEmbeds = nn.Parameter(init(torch.empty(self.user_num, self.embedding_size)))
		self.eEmbeds = nn.Parameter(init(torch.empty(self.n_entities, self.embedding_size)))
		self.rEmbeds = nn.Parameter(init(torch.empty(self.n_relations, self.embedding_size)))
		self.rgat = RGAT(self.embedding_size, self.context_hops, self.mess_dropout_rate )

		self.adj = data_handler.torch_adj
		self.kg_dict = data_handler.kg_dict
		self.edge_index, self.edge_type = self._sample_edges_from_dict(self.kg_dict, triplet_num=configs['model']['triplet_num'])

		self.cl_pattern = configs['model']['cl_pattern']

		self.is_training = True
	
	def getEntityEmbeds(self):
		return self.eEmbeds
	
	def getUserEmbeds(self):
		return self.uEmbeds
	
	def setDenoisedKG(self, denoisedKG):
		self.denoisedKG = denoisedKG

	def _get_edges(self, kg_edges):
		graph_tensor = torch.tensor(kg_edges)
		index = graph_tensor[:, :-1]
		type = graph_tensor[:, -1]
		return index.t().long().to(self.device), type.long().to(self.device)
	
	def _pick_embeds(self, user_embeds, item_embeds, batch_data):
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		return anc_embeds, pos_embeds, neg_embeds

	def _sample_edges_from_dict(self, kg_dict, triplet_num=None):
		sampleEdges = []
		for h in kg_dict:
			t_list = kg_dict[h]
			if triplet_num != -1 and len(t_list) > triplet_num:
				sample_edges_i = random.sample(t_list, triplet_num)
			else:
				sample_edges_i = t_list
			for r, t in sample_edges_i:
				sampleEdges.append([h, t, r])
		return self._get_edges(sampleEdges)
	
	def forward(self, adj, mess_dropout=True, kg=None):
		if kg == None:
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, [self.edge_index, self.edge_type], mess_dropout)
		else:
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, kg, mess_dropout)
		
		embeds = torch.concat([self.uEmbeds, hids_KG[:self.item_num, :]], axis=0)
		embedsLst = [embeds]
		for i in range(self.layer_num):
			embeds = self._propagate(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)
		return embeds[:self.user_num], embeds[self.user_num:]
	
	def full_predict(self, batch_data):
		if configs['model']['cl_pattern'] == 0:
			user_embeds, item_embeds = self.forward(self.adj, mess_dropout=False, kg=self.denoisedKG)
		else:
			user_embeds, item_embeds = self.forward(self.adj, mess_dropout=False)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds
	
	def _propagate(self, adj, embeds):
		return torch.spmm(adj, embeds)
	
	def cal_loss(self, batch_data, denoisedKG):
		self.is_training = True

		if configs['model']['cl_pattern'] == 0:
			user_embeds, item_embeds = self.forward(self.adj, kg=denoisedKG)
			user_embeds_kg, item_embeds_kg = self.forward(self.adj)
		else:
			user_embeds, item_embeds = self.forward(self.adj)
			user_embeds_kg, item_embeds_kg = self.forward(self.adj, kg=denoisedKG)
		
		anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)

		bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
		reg_loss = self.reg_weight * reg_params(self)

		anc_embeds_kg, pos_embeds_kg, neg_embeds_kg = self._pick_embeds(user_embeds_kg, item_embeds_kg, batch_data)
		cl_loss = cal_infonce_loss(anc_embeds, anc_embeds_kg, user_embeds_kg, self.temperature) + cal_infonce_loss(pos_embeds, pos_embeds_kg, item_embeds_kg, self.temperature)
		cl_loss /= anc_embeds.shape[0]
		cl_loss *= self.cl_weight

		loss = bpr_loss + reg_loss + cl_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
		
		return loss, losses


class RGAT(nn.Module):
	def __init__(self, channel, n_hops, mess_dropout_rate=0.4):
		super(RGAT, self).__init__()
		self.mess_dropout_rate = mess_dropout_rate
		self.W = nn.Parameter(init(torch.empty(size=(2*channel, channel)), gain=nn.init.calculate_gain('relu')))

		self.leakyrelu = nn.LeakyReLU(0.2)
		self.n_hops = n_hops
		self.dropout = nn.Dropout(p=mess_dropout_rate)

		self.res_lambda = configs['model']['res_lambda']

	def agg(self, entity_emb, relation_emb, kg):
		edge_index, edge_type = kg
		head, tail = edge_index
		a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
		e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)
		e = self.leakyrelu(e_input)
		e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
		agg_emb = entity_emb[tail] * e.view(-1, 1)
		agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
		agg_emb = agg_emb + entity_emb
		return agg_emb
	
	def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
		entity_res_emb = entity_emb
		for _ in range(self.n_hops):
			entity_emb = self.agg(entity_emb, relation_emb, kg)
			if mess_dropout:
				entity_emb = self.dropout(entity_emb)
			entity_emb = F.normalize(entity_emb)

			entity_res_emb = self.res_lambda * entity_res_emb + entity_emb
		return entity_res_emb
	
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(configs['device'])
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h

class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas)
	
	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			x_t = model_mean
		return x_t
			
	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
	
	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)
	
	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def cal_loss_diff(self, model, batch, ui_matrix, userEmbeds, itmEmbeds):
		x_start, batch_index = batch

		batch_size = x_start.size(0)
		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, ts)
		mse = self.mean_flat((x_start - model_output) ** 2)

		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse

		item_user_matrix = torch.spmm(ui_matrix, model_output[:, :configs['data']['item_num']].t()).t()
		itmEmbeds_kg = torch.mm(item_user_matrix, userEmbeds)
		ukgc_loss = self.mean_flat((itmEmbeds_kg - itmEmbeds[batch_index]) ** 2)

		loss = diff_loss.mean() * (1 - configs['model']['e_loss']) + ukgc_loss.mean() * configs['model']['e_loss']
		losses = {'diff loss': diff_loss, 'ukgc loss': ukgc_loss}
		return loss, losses
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

		