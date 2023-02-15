import torch 
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import dgl
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SMBRec(BaseModel):
    def __init__(self, data_handler):
        super(SMBRec, self).__init__(data_handler)
        self.data_handler = data_handler
        self.userNum = data_handler.userNum
        self.itemNum = data_handler.itemNum
        self.behavior = data_handler.behaviors
        self.behavior_mats = data_handler.behavior_mats        
        self.embedding_dict = self._init_embedding() 
        self.hypergcns = nn.ModuleList()
        for i in range(len(data_handler.beh_meta_path)):
            self.hypergcns.append(GCN(self.userNum, self.itemNum, self.data_handler.behavior_mats[i]))  
        self.cat_trans = nn.Linear(len(self.data_handler.behaviors)*configs['model']['embedding_size'], configs['model']['embedding_size'])
        self.user_trans = nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size'])
        self.beh_weights = nn.Parameter(torch.ones( len(self.data_handler.behaviors) )) 

    def _init_embedding(self): 
        embedding_dict = {  
            'user_embedding': None,
            'item_embedding': None,
            'user_embeddings': None,
            'item_embeddings': None,
        }
        return embedding_dict

    def _propagate(self, adj, embeds):
        return torch.spmm(adj, embeds)

    def _sim(self, z1, z2, tau=None):
        z1 = F.normalize(z1)  
        z2 = F.normalize(z2)
        # z1 = z1/((z1**2).sum(-1) + 1e-8)
        # z2 = z2/((z2**2).sum(-1) + 1e-8)
        return -(torch.exp(torch.mm(z1, z2.t())/tau)+1e-8).log()


    def _batched_contrastive_loss(self, embed, sample_mat=None, batch_size=128):
        device = embed.device
        num_nodes = embed.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        # f = lambda x: torch.exp(x / configs['model']['tau'])      
        indices = torch.arange(0, num_nodes).to(device)

        losses = 0
        # pos sample
        # neg sample
        x_index, y_index = sample_mat.nonzero()  # torch.where(adj_re!=0)[0], torch.where(adj_re!=0)[1] 
        dgl_g_pos = dgl.graph((x_index, y_index)).to('cuda:0')
        dgl_g_neg = dgl.graph((torch.arange(num_nodes), torch.arange(num_nodes))).to('cuda:0')

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]
            pos_row_list, pos_col_list, neg_row_list, neg_col_list = self._dgl_sample(dgl_g_pos, dgl_g_neg, configs['model']['sample_num_pos'], configs['model']['sample_num_pos'], tmp_i)  #, node_type, edge_type)            
            losses += (self._sim(embed[pos_row_list], embed[pos_col_list], tau=configs['model']['tau']) - self._sim(embed[neg_row_list], embed[neg_col_list], tau=configs['model']['tau']) ).sum()  # todo move 
            # losses += (self._sim(embed[pos_row_list], embed[pos_col_list], tau=configs['model']['tau']).sum() - self._sim(embed[neg_row_list], embed[neg_col_list], tau=configs['model']['tau']).sum() )  # todo move 

        #     tmp_i = indices[i * batch_size:(i + 1) * batch_size]
        #     tmp_refl_sim_list = []
        #     tmp_between_sim_list = []
        #     for j in range(num_batches):
        #         tmp_j = indices[j * batch_size:(j + 1) * batch_size]
        #         tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))  
        #         tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))  
        #         tmp_refl_sim_list.append(tmp_refl_sim)
        #         tmp_between_sim_list.append(tmp_between_sim)
        #     refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
        #     between_sim = torch.cat(tmp_between_sim_list, dim=-1)
        #     losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()/ (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())+1e-8))
        #     del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list
        # loss_vec = torch.cat(losses)

        return losses
    
    def _dgl_sample(self, g_pos, g_neg, samp_num, samp_num_neg, anchor_id):  #, node_type, edge_type): #    
        #pos
        # sub_g_pos = dgl.sampling.sample_neighbors(g_pos, {node_type: anchor_id}, samp_num, edge_dir='out', replace=True) #
        sub_g_pos = dgl.sampling.sample_neighbors(g_pos, anchor_id, samp_num, replace=True) #
        # sub_neg_g = dgl.sampling.global_uniform_negative_sampling(g, samp_num_neg)
        # sub_g_neg = dgl.sampling.sample_neighbors(g_neg, {node_type: anchor_id}, samp_num_neg, edge_dir='out', replace=True) #
        sub_g_neg = dgl.sampling.sample_neighbors(g_neg, anchor_id, samp_num_neg, replace=True) #
        # sub_pos_g = dgl.sampling.select_topk(g_pos.cpu(), nodes={node_type: anchor_id.cpu()}, k=samp_num, weight='weight') #
        # dgl.sampling.sample_neighbors(dgl_g, node_id, fanout=args.pos_samp_num, prob='weight')

        row_pos, col_pos = sub_g_pos.edges()
        row_neg, col_neg = sub_g_neg.edges()
        # row_re = row_pos.reshape(len(row_pos), samp_num)
        # col_re = col_pos.reshape(len(row_pos), samp_num_neg)
        # pos_row_list[i] = pos_row_list[i].reshape(len(pos_row_list[i]), samp_num)
        # pos_col_list[i] = pos_col_list[i].reshape(len(pos_row_list[i]), samp_num)
        # neg_row_list[i] = neg_row_list[i].reshape(len(pos_row_list[i]), samp_num_neg)
        # neg_col_list[i] = neg_col_list[i].reshape(len(pos_row_list[i]), samp_num_neg)

        # sub_g_neg = dgl.sampling.sample_neighbors(g_neg, {node_type:anchor_id}, num_samples=sample_num_neg, etype=relation_type, edge_dir='out', replace=True)
        # row_neg, col_neg = sub_g_neg.edges()
        # row_neg = row_neg.reshape(len(batIds), sample_num_neg)
        # col_neg = col_neg.reshape(len(batIds), sample_num_neg)
        return row_pos, col_pos, row_neg, col_neg


    def forward(self):
        self.embed_list_user, self.embed_list_item = [], []
        for i in range(len(self.data_handler.behaviors)):
            user_embedding_tmp, item_embedding_tmp = self.hypergcns[i]()
            self.embed_list_user.append(user_embedding_tmp)
            self.embed_list_item.append(item_embedding_tmp)
        weighted_user_embedding = F.softmax(self.beh_weights.unsqueeze(dim=-1).repeat(1,self.userNum).unsqueeze(dim=-1)*torch.stack(self.data_handler.beh_degree_list), dim=0)*torch.stack(self.embed_list_user)
        user_embedding = self.user_trans(weighted_user_embedding.sum(dim=0))
        item_embedding = self.cat_trans(torch.cat(self.embed_list_item, dim=1))
        return user_embedding, item_embedding

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
        reg_loss = reg_pick_embeds([anc_embeds, pos_embeds, neg_embeds])
        # contractive loss
        cl_loss = 0
        # self.embed_list_user, self.embed_list_item 
        co_user = self.data_handler.trainLabel*self.data_handler.trainLabel.T
        co_item = self.data_handler.trainLabel.T*self.data_handler.trainLabel
        for i in range(len(self.data_handler.behaviors)):     
            cl_loss += self._batched_contrastive_loss(self.embed_list_user[i], co_user) 
            # cl_loss += self._batched_contrastive_loss(item_embeds, co_item)

        # self.embed_list_user, self.embed_list_item = [], []
        # self.meta_embed_list_user, self.meta_embed_list_item = [], []
        # for i in range(1, len(self.embed_list_user)):
        #     cl_loss += self._batched_contrastive_loss(self.embed_list_user[i], self.meta_embed_list_user[i-1], batch_size=1024)
        #     cl_loss += self._batched_contrastive_loss(self.embed_list_item[i], self.meta_embed_list_item[i-1], batch_size=1024)
        loss = bpr_loss + configs['model']['cl_weight']*cl_loss  + configs['model']['reg_weight'] * reg_loss
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
    
    def para_dict_to_tenser(self, para_dict): 
    
        tensors = []
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors.float()

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_parameters(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_parameters()(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  
                    self.set_param(self, name, param)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class GCN(nn.Module):
    def __init__(self, userNum, itemNum, mats):
        super(GCN, self).__init__()  
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = configs['model']['embedding_size']  

        self.mats = mats

        self.user_embedding, self.item_embedding = self.init_embedding()         
        
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(configs['model']['dropout']) 

        self.layer_num = configs['model']['layer_num'] 
        self.layers = nn.ModuleList()
        for i in range(0, self.layer_num):  
            self.layers.append(GCNLayer(configs['model']['embedding_size'], configs['model']['embedding_size'], self.userNum, self.itemNum, self.mats))  

    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, configs['model']['embedding_size'])
        item_embedding = torch.nn.Embedding(self.itemNum, configs['model']['embedding_size'])
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))  
        i_concatenation_w = nn.Parameter(torch.Tensor(configs['model']['layer_num']*configs['model']['embedding_size'], configs['model']['embedding_size']))
        u_concatenation_w = nn.Parameter(torch.Tensor(configs['model']['layer_num']*configs['model']['embedding_size'], configs['model']['embedding_size']))
        i_input_w = nn.Parameter(torch.Tensor(configs['model']['embedding_size'], configs['model']['embedding_size']))
        u_input_w = nn.Parameter(torch.Tensor(configs['model']['embedding_size'], configs['model']['embedding_size']))
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
    




