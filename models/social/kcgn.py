import torch as t
from torch import nn
import torch.nn.functional as F
import math
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import GCNLayer, GCN, DGIEncoder, DGIDiscriminator

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class KCGN(BaseModel):
	def __init__(self, data_handler):
		super(KCGN, self).__init__(data_handler)
		self.data_handler = data_handler
		self.layer_num = configs['model']['layer_num']
		self.reg_weight = configs['model']['reg_weight']
		self.fuse = configs['model']['fuse']
		self.lam = configs['model']['lam']
		self.subnode = configs['model']['subnode']
		self.time_step = configs['model']['time_step']

		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num * data_handler.rating_class, self.embedding_size)))
		slope = configs['model']['slope']
		self.act = nn.LeakyReLU(negative_slope=slope)
		if self.fuse == 'weight':
			self.w = nn.Parameter(t.Tensor(self.item_num, data_handler.rating_class, 1))
			init(self.w)
		
		self.t_e = TimeEncoding(self.embedding_size, data_handler.max_time)
		self.layers = nn.ModuleList()
		for i in range(self.layer_num - 1):
			self.layers.append(GCNLayer(self.embedding_size, self.embedding_size, weight=True, bias=False, activation=self.act))

		dgi_graph_act = nn.Sigmoid()
		self.out_dim = self.embedding_size * self.layer_num

		self.uu_dgi = DGI(data_handler.uu_graph, self.out_dim, self.out_dim, nn.PReLU(), dgi_graph_act).to(configs['device'])
		self.ii_dgi = DGI(data_handler.ii_graph, self.out_dim, self.out_dim, nn.PReLU(), dgi_graph_act).to(configs['device'])

		self.is_training = True
	
	def forward(self, graph, time_seq, out_dim, r_class=5):
		if not self.is_training:
			return self.final_user_embeds, self.final_item_embeds
		all_user_embeds = [self.user_embeds]
		all_item_embeds = [self.item_embeds]
		if len(self.layers) == 0:
			item_embeds = self.item_embeds.view(-1, r_class, out_dim)
			ret_item_embeds = t.div(t.sum(item_embeds, dim=1), r_class)
			return self.user_embeds, ret_item_embeds
		edge_feat = self.t_e(time_seq)

		for i, layer in enumerate(self.layers):
			if i == 0:
				embeds = layer(graph, self.user_embeds, self.item_embeds, edge_feat)
			else:
				embeds = layer(graph, embeds[: self.user_num], embeddings[self.user_num: ], edge_feat)
			
			norm_embeds = F.normalize(embeds, p=2, dim=1)
			all_user_embeds += [norm_embeds[: self.user_num]]
			all_item_embeds += [norm_embeds[self.user_num: ]]

		user_embeds = t.cat(all_user_embeds, 1)
		item_embeds = t.cat(all_item_embeds, 1)
		if r_class == 1:
			return user_embeds, item_embeds
		item_embeds = item_embeds.view(-1, r_class, out_dim)
		if self.fuse == "mean":
			ret_item_embeds = t.div(t.sum(item_embeds, dim=1), r_class)
		elif self.fuse == "weight":
			weight = t.softmax(self.w, dim=1)
			item_embeds = item_embeds * weight
			ret_item_embeds = t.sum(item_embeds, dim=1)
		self.final_user_embeds = user_embeds
		self.final_item_embeds = ret_item_embeds
		return user_embeds, ret_item_embeds
	
	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds, item_embeds = self.forward(self.data_handler.uv_g, self.data_handler.time_seq_tensor, self.out_dim, self.data_handler.rating_class)
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
		reg_loss = self.reg_weight * reg_pick_embeds([anc_embeds, pos_embeds, neg_embeds])
		
		uu_dgi_pos_loss, uu_dgi_neg_loss = self.uu_dgi(user_embeds, self.data_handler.uu_subgraph_adj_tensor, \
			self.data_handler.uu_subgraph_adj_norm, self.data_handler.uu_node_subgraph, self.data_handler.uu_dgi_node)
		user_mask = t.zeros(self.user_num).to(configs['device'])
		user_mask[ancs] = 1
		user_mask = user_mask * self.data_handler.uu_dgi_node_mask
		uu_dgi_loss = self.lam[0] * ((uu_dgi_pos_loss * user_mask).sum() + (uu_dgi_neg_loss * user_mask).sum())/t.sum(user_mask)

		ii_dgi_pos_loss, ii_dgi_neg_loss = self.ii_dgi(item_embeds, self.data_handler.ii_subgraph_adj_tensor, \
            self.data_handler.ii_subgraph_adj_norm, self.data_handler.ii_node_subgraph, self.data_handler.ii_dgi_node)
		ii_mask = t.zeros(self.item_num).to(configs['device'])
		ii_mask[poss] = 1
		ii_mask[negs] = 1
		ii_mask = ii_mask * self.data_handler.ii_dgi_node_mask
		ii_dgi_loss = self.lam[1] * ((ii_dgi_pos_loss * ii_mask).sum() + (ii_dgi_neg_loss * ii_mask).sum())/t.sum(ii_mask)
		loss = bpr_loss + reg_loss + uu_dgi_loss + ii_dgi_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'uu_dgi_loss': uu_dgi_loss, 'ii_dgi_loss': ii_dgi_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward(self.data_handler.uv_g, self.data_handler.time_seq_tensor, self.out_dim, self.data_handler.rating_class)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds

class TimeEncoding(nn.Module):
	def __init__(self, n_hid, max_len=240, dropout=0.2):
		super(TimeEncoding, self).__init__()
		position = t.arange(0., max_len).unsqueeze(1)
		div_term = 1 / (10000 ** (t.arange(0., n_hid * 2, 2.)) / n_hid / 2)

		self.emb = nn.Embedding(max_len, n_hid * 2)
		self.emb.weight.data[:, 0::2] = t.sin(position * div_term) / math.sqrt(n_hid)
		self.emb.weight.data[:, 1::2] = t.cos(position * div_term) / math.sqrt(n_hid)
		self.emb.weight.requires_grad = False

		self.emb.weight.data[0] = t.zeros_like(self.emb.weight.data[-1])
		self.emb.weight.data[1] = t.zeros_like(self.emb.weight.data[-1])
		self.lin = nn.Linear(n_hid * 2, n_hid)
	
	def forward(self, time):
		return self.lin(self.emb(time))

class DGI(nn.Module):
	def __init__(self, g, in_feats, n_hidden, gcn_act, graph_act):
		super(DGI, self).__init__()
		self.encoder = DGIEncoder(g, in_feats, n_hidden, gcn_act)
		self.discriminator = DGIDiscriminator(n_hidden)
		self.graph_act = graph_act
	
	def forward(self, features, subgraph_adj, subgraph_norm, node_subgraph, node_list):
		positive = self.encoder(features, corrupt=False)
		negative = self.encoder(features, corrupt=True)

		graph_embeds = t.sparse.mm(subgraph_adj, positive) / subgraph_norm
		graph_embeds = self.graph_act(graph_embeds)
		summary = graph_embeds[node_subgraph]
		pos_loss = self.discriminator(positive, summary, corrupt=False)
		neg_loss = self.discriminator(negative, summary, corrupt=True)
		return pos_loss, neg_loss
