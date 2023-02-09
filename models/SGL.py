import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.lightgcn import LightGCN

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SGL(LightGCN):
	def __init__(self, data_handler):
		super(SGL, self).__init__()

	def forward(self, adj, keep_rate):
		user_num = configs['data']['user_num']
		augmentation = configs['model']['augmentation']
		if not self.training:
			return self.final_embeds[:user_num], self.final_embeds[user_num:]
		embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
		embeds_list = [embeds]
		if augmentation == 'edge_drop':
			adj = self.edge_dropper(adj, keep_rate)
		for i in range(configs['model']['layer_num']):
			random_walk = augmentation == 'random_walk'
			tem_adj =  adj if not random_walk else self.edge_dropper(adj, keep_rate)
			embeds = self._propagate(adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list) / len(embeds_list)
		self.final_embeds = embeds
		return embeds[:user_num], embeds[user_num:]

	def cal_loss(self, batch_data):
		self.training = True
		if configs['model']['augmentation'] == 'edge_drop':
			