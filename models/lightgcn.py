import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.losses import cal_bpr_loss, reg_pick_embeds

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCN(nn.Module):
	def __init__(self, data_handler):
		super(LightGCN, self).__init__()

		self.adj = data_handler.torch_adj

		self.user_num = configs['data']['user_num']
		self.item_num = configs['data']['item_num']

		self.layer_num = configs['model']['layer_num']
		self.reg_weight = configs['model']['reg_weight']
		self.keep_rate = configs['model']['keep_rate']
		self.embedding_size = configs['model']['embedding_size']

		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

		self.edge_dropper = SpAdjEdgeDrop()
		self.is_training = True
	
	def _propagate(self, adj, embeds):
		return t.spmm(adj, embeds)
	
	def forward(self, adj, keep_rate):
		if not self.is_training:
			return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
		embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
		embeds_list = [embeds]
		adj = self.edge_dropper(adj, keep_rate)
		for i in range(self.layer_num):
			embeds = self._propagate(adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list) / len(embeds_list)
		self.final_embeds = embeds
		return embeds[:self.user_num], embeds[self.user_num:]
	
	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
		reg_loss = reg_pick_embeds([anc_embeds, pos_embeds, neg_embeds])
		loss = bpr_loss + self.reg_weight * reg_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward(self.adj, 1.0)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = full_preds * (1 - train_mask) - 1e8 * train_mask
		return full_preds
	
class SpAdjEdgeDrop(nn.Module):
	def __init__(self):
		super(SpAdjEdgeDrop, self).__init__()

	def forward(self, adj, keep_rate):
		if keep_rate == 1.0:
			return adj
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = (t.rand(edgeNum) + keep_rate).floor().type(t.bool)
		newVals = vals[mask]# / keep_rate
		newIdxs = idxs[:, mask]
		return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)