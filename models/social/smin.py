import torch as t
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import numpy as np
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import GCN

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

		informax_graph_act = nn.Sigmoid()
		self.ui_informax = Informax(data_handler.ui_graph, self.embedding_size, self.embedding_size, nn.PReLU, informax_graph_act, data_handler.ui_graph_adj).cuda()

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

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, activation):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, activation)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features

class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(nn.init.xavier_uniform_(t.empty(n_hidden, n_hidden)))
        self.loss = nn.BCEWithLogitsLoss(reduction='none') # combines a Sigmoid layer and the BCELoss

    def forward(self,node_embedding,graph_embedding, corrupt=False):
        score = torch.sum(node_embedding*graph_embedding,dim=1) 
        
        if corrupt:
            res = self.loss(score,torch.zeros_like(score))
        else:
            res = self.loss(score,torch.ones_like(score))
        return res

class Informax(nn.Module):
	def __init__(self, g, n_in, n_h, gcn_act, graph_act, graph_adj):
		super(Informax, self).__init__()
		self.encoder = Encoder(g, n_in, n_h, gcn_act)
		self.discriminator = Discriminator(n_h)
		self.graph_act = graph_act
		graph_adj_coo = graph_adj.tocoo()
		graph_adj_u, graph_adj_v, graph_adj_r = graph_adj_coo.row, graph_adj_coo.col, graph_adj_coo.data
		self.graph_adj_data = np.hstack((graph_adj_u.reshape(-1, 1), graph_adj_v.reshape(-1, 1))).tolist()
		self.graph_adj_data = np.array(self.graph_adj_data)
		self.mse_loss = nn.MSELoss(reduction='sum')
	
	def forward(self, features, subgraph_adj, subgraph_adj_tensor,subgraph_adj_norm):
		positive = self.encoder(features, corrupt=False)
		negative = self.encoder(features, corrupt=True)

		tmp_features = features
		graph_embeds = torch.sparse.mm(subgraph_adj_tensor, tmp_features) / subgraph_adj_norm
		graph_embeds = self.graph_act(graph_embeds)

		pos_hi_xj_loss = self.discriminator(positive, graph_embeds, corrupt=False)
		neg_hi_xj_loss = self.discriminator(negative, graph_embeds, corrupt=True)

		pos_hi_xi_loss = self.discriminator(positive, features, corrupt=False)
		neg_hi_xi_loss = self.discriminator(negative, features, corrupt=True)

		tmp = t.sigmoid(t.sum(positive[self.graph_adj_data[:, 0]] * positive[self.graph_adj_data[:,1]], dim=1))
		adj_rebuilt = self.mse_loss(tmp, t.ones_like(tmp)) / positive.shape[0]

		return pos_hi_xj_loss, neg_hi_xj_loss, pos_hi_xi_loss, neg_hi_xi_loss, adj_rebuilt
