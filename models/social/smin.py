import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
from dgl.nn.pytorch import GraphConv

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SMIN(BaseModel):
	def __init__(self, data_handler):
		super(SMIN, self).__init__(data_handler)
		self.layer_num = configs['model']['layer_num']
		self.reg_weight = configs['model']['reg_weight']
		self.keep_rate = configs['model']['keep_rate']
		self.user_graph_indx = configs['model']['user_graph_indx']
		self.item_graph_indx = configs['model']['item_graph_indx']
		self.gcn_act = configs['model']['gcn_act']
		self.informax = configs['model']['informax']
		self.informax_graph_act = configs['model']['informax_graph_act']
		self.lambda1 = configs['model']['lambda1']
		self.lambda2 = configs['model']['lambda2']
		self.k_hop_num = configs['model']['k_hop_num']

		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

		self.user_metapath_num = len(data_handler.user_graph)
		self.item_metapath_num = len(data_handler.item_graph)

		self.act = nn.PReLU()
		self.is_training = True

		self.user_meta_layers = nn.ModuleList()
		for _ in range(self.user_metapath_num):
			user_layers = nn.ModuleList()
			for i in range(self.layer_num):
				user_layers.append(GraphConv(self.embedding_size,self.embedding_size, bias=False, activation=self.act))
			self.user_meta_layers.append(user_layers)

		self.item_meta_layers = nn.ModuleList()
		for _ in range(self.item_metapath_num):
			item_layers = nn.ModuleList()
			for i in range(self.layer_num):
				item_layers.append(GraphConv(self.embedding_size,self.embedding_size, bias=False, activation=self.act))
			self.item_meta_layers.append(item_layers)
		
		self.semantic_user_attn = SemanticAttention(self.embedding_size)
		self.semantic_item_attn = SemanticAttention(self.embedding_size)
	
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
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_dim=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = t.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],)+beta.shape) 
        return (beta*z).sum(1)