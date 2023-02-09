import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_pick_embeds, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SGL(LightGCN):
	def __init__(self, data_handler):
		super(SGL, self).__init__()
		self.node_dropper = NodeDrop()

	def forward(self, adj, keep_rate):
		user_num = configs['data']['user_num']
		augmentation = configs['model']['augmentation']
		if not self.training:
			return self.final_embeds[:user_num], self.final_embeds[user_num:]
		embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
		if augmentation == 'node_drop':
			embeds = self.node_dropper(embeds, keep_rate)
		embeds_list = [embeds]
		if augmentation == 'edge_drop':
			adj = self.edge_dropper(adj, keep_rate)
		for i in range(configs['model']['layer_num']):
			random_walk = augmentation == 'random_walk'
			tem_adj =  adj if not random_walk else self.edge_dropper(tem_adj, keep_rate)
			embeds = self._propagate(adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list) / len(embeds_list)
		self.final_embeds = embeds
		return embeds[:user_num], embeds[user_num:]

	def cal_loss(self, batch_data):
		self.training = True
		keep_rate = configs['model']['keep_rate']
		user_embeds1, item_embeds1 = self.forward(self.adj, keep_rate)
		user_embeds2, item_embeds2 = self.forward(self.adj, keep_rate)
		user_embeds3, item_embeds3 = self.forward(self.adj, 1.0)

		anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
		anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
		anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

		bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3)
		cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2) + cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2) + cal_infonce_loss(neg_embeds1, neg_embeds2, item_embeds2)
		weight_decay_loss = reg_pick_embeds([anc_embeds3, pos_embeds3, neg_embeds3])
		loss = bpr_loss + configs['model']['reg_weight'] * weight_decay_loss + configs['model']['cl_weight'] * cl_loss
		losses = {'bpr': bpr_loss, 'weight_decay': weight_decay_loss, 'cl': cl_loss}
		return loss, losses

class NodeDrop(nn.Module):
	def __init__(self):
		super(NodeDrop, self).__init__()

	def forward(self, embeds, keep_rate):
		if keep_rate == 1.0:
			return embeds
		data_config = configs['data']
		node_num = data_config['user_num'] + data_config['item_num']
		mask = (t.rand(node_num) + keep_rate).floor().view([-1, 1])
		return embeds * mask