import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class DirectAU(BaseModel):
	def __init__(self, data_handler):
		super(DirectAU, self).__init__(data_handler)

		self.adj = data_handler.torch_adj

		self.layer_num = configs['model']['layer_num']
		self.gamma = configs['model']['gamma']

		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

		self.is_training = True
	
	def _propagate(self, adj, embeds):
		return t.spmm(adj, embeds)
	
	def forward(self, adj):
		if not self.is_training:
			return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
		embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
		embeds_list = [embeds]
		for i in range(self.layer_num):
			embeds = self._propagate(adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list) / len(embeds_list)
		self.final_embeds = embeds
		return embeds[:self.user_num], embeds[self.user_num:]

	def alignment(self, x, y, alpha=2):
		x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
		return (x - y).norm(p=2, dim=1).pow(alpha).mean()

	def uniformity(self, x):
		x = F.normalize(x, dim=-1)
		return t.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds, item_embeds = self.forward(self.adj)
		ancs, poss, _ = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		align_loss = self.alignment(anc_embeds, pos_embeds)
		uniform_loss = self.gamma * (self.uniformity(anc_embeds) + self.uniformity(pos_embeds)) / 2
		loss = align_loss + uniform_loss
		losses = {'align_loss': align_loss, 'uniform_loss': uniform_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward(self.adj)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds