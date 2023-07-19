import torch as t
from torch import nn
from models.aug_utils import NodeDrop
from config.configurator import configs
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SGL(LightGCN):
	def __init__(self, data_handler):
		super(SGL, self).__init__(data_handler)

		self.augmentation = configs['model']['augmentation']
		self.cl_weight = configs['model']['cl_weight']
		self.temperature = configs['model']['temperature']
		self.node_dropper = NodeDrop()

	def forward(self, adj, keep_rate):
		if not self.is_training and self.final_embeds is not None:
			return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
		embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
		if self.augmentation == 'node_drop':
			embeds = self.node_dropper(embeds, keep_rate)
		embeds_list = [embeds]
		if self.augmentation == 'edge_drop':
			adj = self.edge_dropper(adj, keep_rate)
		for i in range(configs['model']['layer_num']):
			random_walk = self.augmentation == 'random_walk'
			tem_adj = adj if not random_walk else self.edge_dropper(tem_adj, keep_rate)
			embeds = self._propagate(adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list)# / len(embeds_list)
		self.final_embeds = embeds
		return embeds[:self.user_num], embeds[self.user_num:]
	
	def _pick_embeds(self, user_embeds, item_embeds, batch_data):
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		return anc_embeds, pos_embeds, neg_embeds

	def cal_loss(self, batch_data):
		self.is_training = True
		keep_rate = configs['model']['keep_rate']
		user_embeds1, item_embeds1 = self.forward(self.adj, keep_rate)
		user_embeds2, item_embeds2 = self.forward(self.adj, keep_rate)
		user_embeds3, item_embeds3 = self.forward(self.adj, 1.0)

		anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
		anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
		anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

		bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]
		cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.temperature) + \
				  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.temperature) + \
				  cal_infonce_loss(neg_embeds1, neg_embeds2, item_embeds2, self.temperature)
		cl_loss /= anc_embeds1.shape[0]
		reg_loss = self.reg_weight * reg_params(self)
		cl_loss *= self.cl_weight
		loss = bpr_loss + reg_loss + cl_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
		return loss, losses
