import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCN(nn.Module):
	def __init__(self, data_handler):
		super(LightGCN, self).__init__()

		self.adj = data_handler.torch_adj

		self.user_embeds = nn.Parameter(init(t.empty(configs['data']['user_num'], configs['data']['item_num'])))
		self.item_embeds = nn.Parameter(init(t.empty(configs['data']['item_num'])))
	
		self.edge_dropper = SpAdjEdgeDrop()
	
	def _propagate(self, adj, embeds):
		return t.spmm(adj, embeds)
	
	def forward(self, adj, keep_rate):
		embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
		embeds_list = [embeds]
		adj = self.edge_dropper(adj, keep_rate)
		for i in range(configs['model']['layer_num']):
			embeds = self._propagate(adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list) / len(embeds_list)
		return embeds[:configs['data']['user_num']], embeds[configs['data']['user_num']:]
	
	def cal_loss(self, batch_data):
		user_embeds, item_embeds = self.forward(self.adj, configs['model']['keep_rate'])
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		score_diff = (anc_embeds * pos_embeds).sum(-1) - (anc_embeds * neg_embeds).sum(-1)
		bpr_loss = - score_diff.sigmoid().log().sum()
		weight_decay_loss = anc_embeds.square().sum() + pos_embeds.square().sum() + neg_embeds.square().sum()
		loss = bpr_loss + configs['optimizer']['weight_decay'] * weight_decay_loss
		losses = {'bpr': bpr_loss, 'weight_decay': weight_decay_loss}
		return loss, losses
	
class SpAdjEdgeDrop(nn.Module):
	def __init__(self):
		super(SpAdjEdgeDrop, self).__init__()

	def forward(self, adj, keepRate):
		if keepRate == 1.0:
			return adj
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
		newVals = vals[mask] / keepRate
		newIdxs = idxs[:, mask]
		return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)