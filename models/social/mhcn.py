import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class MHCN(BaseModel):
	def __init__(self, data_handler):
		super(MHCN, self).__init__(data_handler)
		self.weight = {}
		self.n_channel = 4
		for i in range(self.n_channel):
			self.weights['gating%d' % (i+1)] = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
			self.weights['gating_bias%d' %(i+1)] = nn.Parameter(init(t.empty(1, self.embedding_size)))
			self.weights['sgating%d' % (i + 1)] = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
			self.weights['sgating_bias%d' % (i + 1)] = nn.Parameter(init(t.empty(1, self.embedding_size)))
		self.weights['attention'] = nn.Parameter(init(t.empty(1, self.embedding_size)))
		self.weights['attention_mat'] = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
		self.layer_num = configs['model']['layer_num']
		self.reg_weight = configs['model']['reg_weight']
        
		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))        

		self.is_training = True
        
    # def _self_gating(self, em, channel):
    #     return t.multiply(em, F.sigmoid(t.matmul(em, self.weights['gating%d' % channel]) + self.weights['gating_bias%d' %channel]))
        
    # def _self_supervised_gating(self, em, channel):
    #     return t.multiply(em, F.sigmoid(t.matmul(em, self.weights['sgating%d' % channel]) + self.weights['sgating_bias%d' % channel]))

    # def _channel_attention(self, *channel_embeddings):
    #     weights = []
    #     for embedding in channel_embeddings:
    #         weights.append(t.sum(t.multiply(self.weights['attention'], t.matmul(embedding, self.weights['attention_mat'])), 1))
    #     score = F.softmax(t.t(weights))
    #     mixed_embeddings = 0
    #     for i in range(len(weights)):
    #         mixed_embeddings += t.t(t.multiply(t.t(score)[i], t.t(channel_embeddings[i])))
    #     return mixed_embeddings, score

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
		reg_loss = self.reg_weight * reg_pick_embeds([anc_embeds, pos_embeds, neg_embeds])
		loss = bpr_loss + reg_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward(self.adj, 1.0)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds