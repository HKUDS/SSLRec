import torch as t
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import numpy as np
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import GCN, DGIEncoder, DGIDiscriminator

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SMIN(BaseModel):
	def __init__(self, data_handler):
		super(SMIN, self).__init__(data_handler)
		self.layer_num = configs['model']['layer_num']
		self.reg_weight = configs['model']['reg_weight']
		self.user_graph_indx = configs['model']['user_graph_indx']
		self.item_graph_indx = configs['model']['item_graph_indx']
		self.gcn_act = configs['model']['gcn_act']
		self.lambda1 = configs['model']['lambda1']
		self.lambda2 = configs['model']['lambda2']
		self.k_hop_num = configs['model']['k_hop_num']

		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

		self.data_handler = data_handler
		self.user_graph = data_handler.user_graph
		self.item_graph = data_handler.item_graph
		self.user_metapath_num = len(self.user_graph)
		self.item_metapath_num = len(self.item_graph)
		self.in_size = self.layer_num * self.embedding_size
		self.out_dim = self.layer_num * self.embedding_size
		self.act = nn.PReLU()
		self.is_training = True

		self.user_meta_layers = nn.ModuleList()
		for _ in range(self.user_metapath_num):
			user_layers = nn.ModuleList()
			for i in range(self.layer_num-1):
				user_layers.append(GraphConv(self.embedding_size, self.embedding_size, bias=False, activation=self.act))
			self.user_meta_layers.append(user_layers)

		self.item_meta_layers = nn.ModuleList()
		for _ in range(self.item_metapath_num):
			item_layers = nn.ModuleList()
			for i in range(self.layer_num-1):
				item_layers.append(GraphConv(self.embedding_size, self.embedding_size, bias=False, activation=self.act))
			self.item_meta_layers.append(item_layers)
		

		self.semantic_user_attn = SemanticAttention(self.in_size)
		self.semantic_item_attn = SemanticAttention(self.in_size)

		informax_graph_act = nn.Sigmoid()
		self.ui_informax = Informax(data_handler.ui_graph, self.out_dim, self.out_dim, nn.PReLU(), informax_graph_act, data_handler.ui_graph_adj).to(configs['device'])
	
	def forward(self, norm=1):
		if not self.is_training:
			return self.final_user_embeds, self.final_item_embeds
		
		semantic_user_embeds = []
		semantic_item_embeds = []

		path_num, block_num = np.shape(self.user_meta_layers)
		for i in range(path_num):
			all_user_embeds = [self.user_embeds]
			layers = self.user_meta_layers[i]
			for j in range(block_num):
				layer = layers[j]
				if j == 0:
					user_embeds = layer(self.user_graph[i], self.user_embeds)
				else:
					user_embeds = layer(self.user_graph[i], user_embeds)
				
				if norm == 1:
					norm_embeds = F.normalize(user_embeds, p=2, dim=1)
					all_user_embeds += [norm_embeds]
				else:
					all_user_embeds += [user_embeds]
			user_embeds = t.cat(all_user_embeds, 1)
			semantic_user_embeds.append(user_embeds)
		
		path_num, block_num = np.shape(self.item_meta_layers)
		for i in range(path_num):
			all_item_embeds = [self.item_embeds]
			layers = self.item_meta_layers[i]
			for j in range(block_num):
				layer = layers[j]
				if j == 0:
					item_embeds = layer(self.item_graph[i], self.item_embeds)
				else:
					item_embeds = layer(self.item_graph[i], item_embeds)
				
				if norm == 1:
					norm_embeds = F.normalize(item_embeds, p=2, dim=1)
					all_item_embeds += [norm_embeds]
				else:
					all_item_embeds += [item_embeds]
			item_embeds = t.cat(all_item_embeds, 1)
			semantic_item_embeds.append(item_embeds)

		semantic_user_embeds = t.stack(semantic_user_embeds, dim=1)
		semantic_item_embeds = t.stack(semantic_item_embeds, dim=1)

		user_embeds = self.semantic_user_attn(semantic_user_embeds)
		item_embeds = self.semantic_item_attn(semantic_item_embeds)
		
		self.final_user_embeds = user_embeds
		self.final_item_embeds = item_embeds

		return user_embeds, item_embeds
	
	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds, item_embeds = self.forward()
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
		reg_loss = self.reg_weight * reg_pick_embeds([anc_embeds, pos_embeds, neg_embeds])
		
		all_embeds = t.cat([user_embeds, item_embeds], dim=0)
		res = self.ui_informax(all_embeds, self.data_handler.ui_subgraph_adj, self.data_handler.ui_subgraph_adj_tensor, self.data_handler.ui_subgraph_adj_norm)
		mask = t.zeros((self.user_num + self.item_num)).to(configs['device'])
		mask[ancs] = 1
		mask[self.user_num + poss] = 1
		mask[self.user_num + negs] = 1
		informax_loss = configs['model']['lambda1'] * (((mask * res[0]).sum() + (mask * res[1]).sum()) / t.sum(mask))\
			+ configs['model']['lambda2'] * (((mask * res[2]).sum() + (mask * res[3]).sum()) / t.sum(mask) + res[4])
		loss = bpr_loss + reg_loss + informax_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'informax_loss': informax_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward()
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

class Informax(nn.Module):
	def __init__(self, g, n_in, n_h, gcn_act, graph_act, graph_adj):
		super(Informax, self).__init__()
		self.encoder = DGIEncoder(g, n_in, n_h, gcn_act)
		self.discriminator = DGIDiscriminator(n_h)
		self.graph_act = graph_act
		graph_adj_coo = graph_adj.tocoo()
		graph_adj_u, graph_adj_v, graph_adj_r = graph_adj_coo.row, graph_adj_coo.col, graph_adj_coo.data
		self.graph_adj_data = np.hstack((graph_adj_u.reshape(-1, 1), graph_adj_v.reshape(-1, 1))).tolist()
		self.graph_adj_data = np.array(self.graph_adj_data)
		self.mse_loss = nn.MSELoss(reduction='sum')
	
	def forward(self, features, subgraph_adj, subgraph_adj_tensor, subgraph_adj_norm):
		positive = self.encoder(features, corrupt=False)
		negative = self.encoder(features, corrupt=True)

		tmp_features = features
		graph_embeds = t.sparse.mm(subgraph_adj_tensor, tmp_features) / subgraph_adj_norm
		graph_embeds = self.graph_act(graph_embeds)

		pos_hi_xj_loss = self.discriminator(positive, graph_embeds, corrupt=False)
		neg_hi_xj_loss = self.discriminator(negative, graph_embeds, corrupt=True)

		pos_hi_xi_loss = self.discriminator(positive, features, corrupt=False)
		neg_hi_xi_loss = self.discriminator(negative, features, corrupt=True)

		tmp = t.sigmoid(t.sum(positive[self.graph_adj_data[:, 0]] * positive[self.graph_adj_data[:,1]], dim=1))
		adj_rebuilt = self.mse_loss(tmp, t.ones_like(tmp)) / positive.shape[0]

		return pos_hi_xj_loss, neg_hi_xj_loss, pos_hi_xi_loss, neg_hi_xi_loss, adj_rebuilt
