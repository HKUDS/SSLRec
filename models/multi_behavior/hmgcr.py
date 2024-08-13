import torch 
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class HMGCR(BaseModel):
    def __init__(self, data_handler):
        super(HMGCR, self).__init__(data_handler)
        self.data_handler = data_handler
        self.userNum = data_handler.userNum
        self.itemNum = data_handler.itemNum
        self.behavior = data_handler.behaviors
        self.behavior_mats = data_handler.behavior_mats
        self.hypergcns = nn.ModuleList()
        for i in range(len(data_handler.beh_meta_path)):
            self.hypergcns.append(GCN(self.userNum, self.itemNum, self.data_handler.beh_meta_path_mats[i]))  

    def _propagate(self, adj, embeds):
        return torch.spmm(adj, embeds)

    def _sim(self, z1, z2):
        z1 = F.normalize(z1)  
        z2 = F.normalize(z2)
        # z1 = z1/((z1**2).sum(-1) + 1e-8)
        # z2 = z2/((z2**2).sum(-1) + 1e-8)
        return torch.mm(z1, z2.t())

    def _batched_contrastive_loss(self, z1, z2, batch_size=1024):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / configs['model']['tau'])      
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]
            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self._sim(z1[tmp_i], z1[tmp_j]))  
                tmp_between_sim = f(self._sim(z1[tmp_i], z2[tmp_j]))  
                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)
            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()/ (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())+1e-8))
            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list      
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def forward(self):
        #step1: normal GNNs
        self.embed_list_user, self.embed_list_item = [], []
        for i in range(len(self.data_handler.beh_meta_path)):
            user_embedding_tmp, item_embedding_tmp = self.hypergcns[i]()
            self.embed_list_user.append(user_embedding_tmp)
            self.embed_list_item.append(item_embedding_tmp)
        #step2: contrastive GNNs
        self.meta_embed_list_user, self.meta_embed_list_item = [], []
        for i in range(1, len(self.data_handler.beh_meta_path)):
            user_embedding_tmp, item_embedding_tmp = self.hypergcns[i-1]()
            self.meta_embed_list_user.append(user_embedding_tmp)
            self.meta_embed_list_item.append(item_embedding_tmp)
        user_embedding = torch.mean( torch.stack(self.embed_list_user), dim=0)
        item_embedding = torch.mean( torch.stack(self.embed_list_item), dim=0)
        return user_embedding, item_embedding

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
        # contractive loss
        cl_loss = 0
        # self.embed_list_user, self.embed_list_item = [], []
        # self.meta_embed_list_user, self.meta_embed_list_item = [], []
        for i in range(1, len(self.embed_list_user)):
            cl_loss += self._batched_contrastive_loss(self.embed_list_user[i], self.meta_embed_list_user[i-1], batch_size=1024)
            cl_loss += self._batched_contrastive_loss(self.embed_list_item[i], self.meta_embed_list_item[i-1], batch_size=1024)
        loss = configs['model']['beta_loss']*bpr_loss + (1-configs['model']['beta_loss'])*cl_loss 
        losses = {'bpr_loss': bpr_loss, 'cl_loss': cl_loss}
        return loss, losses

    def full_predict(self, batch_data):  # todo co-current matrix version
        user_embeds, item_embeds = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

class GCN(nn.Module):
    def __init__(self, userNum, itemNum, mats):
        super(GCN, self).__init__()  
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = configs['model']['hidden_dim']
        self.mats = mats
        self.user_embedding, self.item_embedding = self.init_embedding()         
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(configs['model']['dropout']) 
        self.layer_num = configs['model']['layer_num'] 
        self.layers = nn.ModuleList()
        for i in range(0, self.layer_num):  
            self.layers.append(GCNLayer(configs['model']['hidden_dim'], configs['model']['hidden_dim'], self.userNum, self.itemNum, self.mats))  

    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, configs['model']['hidden_dim'])
        item_embedding = torch.nn.Embedding(self.itemNum, configs['model']['hidden_dim'])
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)
        return user_embedding, item_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))  
        i_concatenation_w = nn.Parameter(torch.Tensor(configs['model']['layer_num']*configs['model']['hidden_dim'], configs['model']['hidden_dim']))
        u_concatenation_w = nn.Parameter(torch.Tensor(configs['model']['layer_num']*configs['model']['hidden_dim'], configs['model']['hidden_dim']))
        i_input_w = nn.Parameter(torch.Tensor(configs['model']['hidden_dim'], configs['model']['hidden_dim']))
        u_input_w = nn.Parameter(torch.Tensor(configs['model']['hidden_dim'], configs['model']['hidden_dim']))
        nn.init.xavier_uniform_(i_concatenation_w)
        nn.init.xavier_uniform_(u_concatenation_w)
        nn.init.xavier_uniform_(i_input_w)
        nn.init.xavier_uniform_(u_input_w)
        # init.xavier_uniform_(alpha)
        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w

    def forward(self, user_embedding_input=None, item_embedding_input=None):
        all_user_embeddings = []
        all_item_embeddings = []
        if user_embedding_input!=None:
            user_embedding = user_embedding_input
            item_embedding = item_embedding_input
        else:
            user_embedding = self.user_embedding.weight
            item_embedding = self.item_embedding.weight
        for i, layer in enumerate(self.layers):
            user_embedding, item_embedding = layer(user_embedding, item_embedding)
            # norm_user_embeddings = F.normalize(user_embedding, p=2, dim=1)
            # norm_item_embeddings = F.normalize(item_embedding, p=2, dim=1)  
            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
        user_embedding = torch.mean( torch.stack(all_user_embeddings), dim=0)
        item_embedding = torch.mean( torch.stack(all_item_embeddings), dim=0)
        return user_embedding, item_embedding


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, mat):
        super(GCNLayer, self).__init__()
        self.mat = mat
        self.userNum = userNum
        self.itemNum = itemNum
        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.i_w)
        nn.init.xavier_uniform_(self.u_w)

    def forward(self, user_embedding, item_embedding):
        user_embedding = torch.spmm(self.mat['A'], item_embedding)
        item_embedding = torch.spmm(self.mat['AT'], user_embedding)
        user_embedding = self.act(torch.matmul(user_embedding, self.u_w))
        item_embedding = self.act(torch.matmul(item_embedding, self.i_w))
        return user_embedding, item_embedding
    




