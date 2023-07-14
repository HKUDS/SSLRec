import torch as t
from torch import nn
from config.configurator import configs
from models.aug_utils import EmbedPerturb
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SimGCL(LightGCN):
	def __init__(self, data_handler):
		super(SimGCL, self).__init__(data_handler)
		
		self.cl_weight = configs['model']['cl_weight']
		self.temperature = configs['model']['temperature']
		self.eps = configs['model']['eps']
		self.embed_perturb = EmbedPerturb(eps=self.eps)
	
	def forward(self, adj, perturb=False):
		if not perturb:
			return super(SimGCL, self).forward(adj, 1.0)
		embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
		embeds_list = [embeds]
		for i in range(self.layer_num):
			embeds = self._propagate(adj, embeds_list[-1])
			embeds = self.embed_perturb(embeds)
			embeds_list.append(embeds)
		embeds = sum(embeds_list)
		return embeds[:self.user_num], embeds[self.user_num:]
	
	def _pick_embeds(self, user_embeds, item_embeds, batch_data):
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		return anc_embeds, pos_embeds, neg_embeds
		
	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True)
		user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True)
		user_embeds3, item_embeds3 = self.forward(self.adj, perturb=False)

		anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
		anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
		anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
		bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]
		cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.temperature) + cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.temperature)
		cl_loss /= anc_embeds1.shape[0]
		reg_loss = self.reg_weight * reg_params(self)
		cl_loss *= self.cl_weight
		loss = bpr_loss + reg_loss + cl_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward(self.adj, False)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds