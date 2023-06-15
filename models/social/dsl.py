import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class DSL(BaseModel):
	def __init__(self, data_handler):
		super(DSL, self).__init__(data_handler)
		self.data_handler = data_handler
		self.adj = data_handler.torch_adj
		self.u_adj = data_handler.torch_uu_adj

		self.embedding_size = configs['model']['embedding_size']
		self.uugnn_layer = configs['model']['uugnn_layer']
		self.leaky = configs['model']['leaky']
		self.reg_weight = configs['model']['reg_weight']
		self.soc_weight = configs['model']['soc_weight']
		self.sal_weight = configs['model']['sal_weight']
		self.dropout_rate = configs['model']['dropout_rate']
        
		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))        
		
		self.lightgcn = LightGCN(self.user_embeds, self.item_embeds)
		self.lightgcn2 = LightGCN2(self.user_embeds)
		self.linear1 = nn.Linear(2*self.embedding_size, self.embedding_size)
		self.linear2 = nn.Linear(self.embedding_size, 1)
		self.dropout = nn.Dropout(self.dropout_rate)
		self.leakyrelu = nn.LeakyReLU(self.leaky)
		self.sigmoid = nn.Sigmoid()

		self.is_training = True

	def label(self, lat1, lat2):
		lat = t.cat([lat1, lat2], dim=-1)
		lat = self.leakyrelu(self.dropout(self.linear1(lat))) + lat1 + lat2
		ret = self.sigmoid(self.dropout(self.linear2(lat)))
		ret = ret.view(-1)
		return ret

	def forward(self, adj, u_adj):
		if not self.is_training:
			return self.final_user_embeds, self.final_item_embeds, self.final_user_embeds2

		ret_user_embeds, ret_item_embeds = self.lightgcn(adj)
		ret_user_embeds2 = self.lightgcn2(u_adj)

		self.final_user_embeds = ret_user_embeds
		self.final_item_embeds = ret_item_embeds
		self.final_user_embeds2 = ret_user_embeds2

		return ret_user_embeds, ret_item_embeds, ret_user_embeds2
	
	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds, item_embeds, user_embeds2 = self.forward(self.adj, self.u_adj)
		ancs, poss, negs, user0, user_p, user_n, user1, user2 = batch_data

		# bprloss on G_r
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		rec_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
		reg_loss = self.reg_weight * reg_params(self)
		
		# bprloss on G_s
		anc_embeds = user_embeds2[user0]
		pos_embeds = user_embeds2[user_p]
		neg_embeds = user_embeds2[user_n]
		soc_loss = self.soc_weight * cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)

		# self-augmented learning
		scores = self.label(user_embeds[user1], user_embeds[user2])
		preds = (user_embeds2[user1] * user_embeds2[user2]).sum(-1)
		sal_loss = self.sal_weight * (t.maximum(t.tensor(0.0), 1.0-scores*preds)).sum()

		loss = rec_loss + reg_loss + soc_loss + sal_loss
		losses = {'rec_loss': rec_loss, 'reg_loss': reg_loss, 'soc_loss': soc_loss, 'sal_loss': sal_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds, _ = self.forward(self.adj, self.u_adj)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds

class LightGCN(nn.Module):
	def __init__(self, user_embeds, item_embeds, pool='sum'):
		super(LightGCN, self).__init__()
		self.user_embeds = user_embeds
		self.item_embeds = item_embeds
		self.gcn_layers = nn.Sequential(*[GCNLayer() for i in range(configs['model']['gnn_layer'])])
		self.pooling_fn = {
			'mean': lambda x: x.mean(0),
			'sum': lambda x: x.sum(0),
			'concat': lambda x: x.view(x.shape[1], -1),
			'last': lambda x: x[-1]
		}[pool]
			
	def pooling(self, embeds):
		return self.pooling_fn(embeds)
		
	def forward(self, adj):
		embed_list = [t.cat([self.user_embeds, self.item_embeds], dim=0)]
		for gcn in self.gcn_layers:
			embeds = gcn(adj, embed_list[-1])
			embed_list.append(embeds)
		embeds = t.stack(embed_list, dim=0)
		embeds = self.pooling(embeds)
		return embeds[:configs['data']['user_num']], embeds[configs['data']['user_num']:]

class LightGCN2(nn.Module):
	def __init__(self, user_embeds, pool='sum'):
		super(LightGCN2, self).__init__()
		self.user_embeds = user_embeds
		self.gnn_layers = nn.Sequential(*[GCNLayer() for i in range(configs['model']['uugnn_layer'])])
		self.pooling_fn = {
			'mean': lambda x: x.mean(0),
			'sum': lambda x: x.sum(0),
			'concat': lambda x: x.view(x.shape[1], -1),
			'last': lambda x: x[-1]
		}[pool]
	
	def pooling(self, embeds):
		return self.pooling_fn(embeds)
	
	def forward(self, adj):
		ulats = [self.user_embeds]
		for gcn in self.gnn_layers:
			temulat = gcn(adj, ulats[-1])
			ulats.append(temulat)
		ulats = t.stack(ulats, dim=0)
		ulats = self.pooling(ulats)
		return ulats

class GCNLayer(nn.Module):
	def __init__(self,):
		super(GCNLayer, self).__init__()
		self.dropout = nn.Dropout(p=configs['model']['dropout_rate'])
		
	def forward(self, adj, embeds):
		embeds = t.spmm(adj, embeds)
		return embeds