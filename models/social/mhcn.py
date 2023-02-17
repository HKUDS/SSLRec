import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class MHCN(BaseModel):
	def __init__(self, data_handler):
		super(MHCN, self).__init__(data_handler)
		self.data_handler = data_handler
		self.layer_num = configs['model']['layer_num']
		self.reg_weight = configs['model']['reg_weight']
		self.ss_rate = configs['model']['ss_rate']
        
		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))        
		
		self.gating1 = nn.Linear(self.embedding_size, self.embedding_size)
		self.gating2 = nn.Linear(self.embedding_size, self.embedding_size)
		self.gating3 = nn.Linear(self.embedding_size, self.embedding_size)
		self.gating4 = nn.Linear(self.embedding_size, self.embedding_size)
		self.sgating1 = nn.Linear(self.embedding_size, self.embedding_size)
		self.sgating2 = nn.Linear(self.embedding_size, self.embedding_size)
		self.sgating3 = nn.Linear(self.embedding_size, self.embedding_size)
		self.attn = nn.Parameter(init(t.empty(1, self.embedding_size)))
		self.attn_mat = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
		self.is_training = True

	def _self_gating(self, em, channel):
		if channel == 1:
			gates = t.sigmoid(self.gating1(em))
		elif channel == 2:
			gates = t.sigmoid(self.gating2(em))
		elif channel == 3:
			gates = t.sigmoid(self.gating3(em))
		elif channel == 4:
			gates = t.sigmoid(self.gating4(em))
		return em * gates

	def _self_supervised_gating(self, em, channel):
		if channel == 1:
			sgates = t.sigmoid(self.sgating1(em))
		elif channel == 2:
			sgates = t.sigmoid(self.sgating2(em))
		elif channel == 3:
			sgates = t.sigmoid(self.sgating3(em))
		return em * sgates

	def _channel_attention(self, *channel_embeds):
		weights = []
		for embed in channel_embeds:
			weight = (self.attn * (embed @ self.attn_mat)).sum(1)
			weights.append(weight)
		weights = t.stack(weights, 0)
		score = F.softmax(t.t(weights), dim=-1)
		mixed_embeds = 0
		for i in range(len(weights)):
			mixed_embeds += t.t(t.multiply(t.t(score)[i], t.t(channel_embeds[i])))
		return mixed_embeds, score

	def forward(self):
		if not self.is_training:
			return self.final_user_embeds, self.final_item_embeds
		user_embeds_c1 = self._self_gating(self.user_embeds, 1)
		user_embeds_c2 = self._self_gating(self.user_embeds, 2)
		user_embeds_c3 = self._self_gating(self.user_embeds, 3)
		simp_user_embeds = self._self_gating(self.user_embeds, 4)
		all_embeds_c1 = [user_embeds_c1]
		all_embeds_c2 = [user_embeds_c2]
		all_embeds_c3 = [user_embeds_c3]
		all_embeds_simp = [simp_user_embeds]
		item_embeds = self.item_embeds
		all_embeds_i = [item_embeds]

		for k in range(self.layer_num):
			mixed_embed = self._channel_attention(user_embeds_c1, user_embeds_c2, user_embeds_c3)[0] + simp_user_embeds / 2

			user_embeds_c1 = t.spmm(self.data_handler.H_s, user_embeds_c1)
			norm_embeds = F.normalize(user_embeds_c1, p=2, dim=1)
			all_embeds_c1 += [norm_embeds]

			user_embeds_c2 = t.spmm(self.data_handler.H_j, user_embeds_c2)
			norm_embeds = F.normalize(user_embeds_c2, p=2, dim=1)
			all_embeds_c2 += [norm_embeds]

			user_embeds_c3 = t.spmm(self.data_handler.H_p, user_embeds_c3)
			norm_embeds = F.normalize(user_embeds_c3, p=2, dim=1)
			all_embeds_c3 += [norm_embeds]

			new_item_embeds = t.spmm(t.t(self.data_handler.R), mixed_embed)
			norm_embeds = F.normalize(new_item_embeds, p=2, dim=1)
			all_embeds_i += [norm_embeds]

			simp_user_embeds = t.spmm(self.data_handler.R, item_embeds)
			norm_embeds = F.normalize(simp_user_embeds, p=2, dim=1)
			all_embeds_simp += [norm_embeds]

			item_embeds = new_item_embeds

		user_embeds_c1 = sum(all_embeds_c1)
		user_embeds_c2 = sum(all_embeds_c2)
		user_embeds_c3 = sum(all_embeds_c3)
		simp_user_embeds = sum(all_embeds_simp)
		item_embeds = sum(all_embeds_i)

		ret_item_embeds = item_embeds
		ret_user_embeds, attn_score = self._channel_attention(user_embeds_c1, user_embeds_c2, user_embeds_c3)
		ret_user_embeds += simp_user_embeds / 2

		self.final_user_embeds = ret_user_embeds
		self.final_item_embeds = ret_item_embeds

		return ret_user_embeds, ret_item_embeds
	
	def _hierarchical_self_supervision(self, em, adj):
		def row_shuffle(embed):
			indices = t.randperm(embed.shape[0])
			return embed[indices]
		def row_col_shuffle(embed):
			indices = t.randperm(t.t(embed).shape[0])
			corrupted_embed = t.t(t.t(embed)[indices])
			indices = t.randperm(corrupted_embed.shape[0])
			return corrupted_embed[indices]
		def score(x1, x2):
			return (x1 * x2).sum(1)
		user_embeds = em
		edge_embeds = t.spmm(adj, user_embeds)

		pos = score(user_embeds, edge_embeds)
		neg1 = score(row_shuffle(user_embeds), edge_embeds)
		neg2 = score(row_col_shuffle(edge_embeds), user_embeds)
		local_ssl = -((pos-neg1).sigmoid().log()+(neg1-neg2).sigmoid().log()).sum()

		graph = edge_embeds.mean(0)
		pos = score(edge_embeds, graph)
		neg1 = score(row_col_shuffle(edge_embeds), graph)
		global_ssl = -(pos-neg1).sigmoid().log().sum()
		return local_ssl + global_ssl

	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds, item_embeds = self.forward()
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
		reg_loss = self.reg_weight * reg_params(self)
		ss_loss = 0
		ss_loss += self._hierarchical_self_supervision(self._self_supervised_gating(user_embeds, 1), self.data_handler.H_s)
		ss_loss += self._hierarchical_self_supervision(self._self_supervised_gating(user_embeds, 2), self.data_handler.H_j)
		ss_loss += self._hierarchical_self_supervision(self._self_supervised_gating(user_embeds, 3), self.data_handler.H_p)
		ss_loss *= self.ss_rate
		loss = bpr_loss + reg_loss + ss_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'ss_loss': ss_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward()
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds