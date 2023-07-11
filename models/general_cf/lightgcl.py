import torch as t
import numpy as np
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCL(BaseModel):
    def __init__(self, data_handler):
        super(LightGCL, self).__init__(data_handler)

        train_mat = data_handler._load_one_mat(data_handler.trn_file)
        rowD = np.array(train_mat.sum(1)).squeeze()
        colD = np.array(train_mat.sum(0)).squeeze()
        for i in range(len(train_mat.data)):
            train_mat.data[i] = train_mat.data[i] / pow(rowD[train_mat.row[i]] * colD[train_mat.col[i]], 0.5)
        adj_norm = self._scipy_sparse_mat_to_torch_sparse_tensor(train_mat)

        self.adj = adj_norm.coalesce().cuda()
        self.ut, self.vt, self.u_mul_s, self.v_mul_s = self._svd_reconstruction(self.adj)

        self.layer_num = configs['model']['layer_num']
        self.cl_weight = configs['model']['cl_weight']
        self.dropout = configs['model']['dropout']
        self.temp = configs['model']['temp']

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        self.E_u_list = [None] * (self.layer_num+1)
        self.E_i_list = [None] * (self.layer_num+1)
        self.E_u_list[0] = self.user_embeds
        self.E_i_list[0] = self.item_embeds
        self.Z_u_list = [None] * (self.layer_num+1)
        self.Z_i_list = [None] * (self.layer_num+1)
        self.G_u_list = [None] * (self.layer_num+1)
        self.G_i_list = [None] * (self.layer_num+1)

        self.E_u = None
        self.E_i = None

        self.act = nn.LeakyReLU(0.5)

        self.Ws = nn.ModuleList([W_contrastive(self.embedding_size) for i in range(self.layer_num)])

        self.is_training = True

    def _svd_reconstruction(self,train_mat):
        # adj = self._scipy_sparse_mat_to_torch_sparse_tensor(train_mat).coalesce().cuda()
        adj = train_mat
        print('Performing svd...')
        svd_u, s, svd_v = t.svd_lowrank(adj, q=configs['model']['svd_q'])
        u_mul_s = svd_u @ t.diag(s)
        v_mul_s = svd_v @ t.diag(s)
        # del adj
        del s
        print('svd done.')
        return svd_u.T, svd_v.T, u_mul_s, v_mul_s

    def _scipy_sparse_mat_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = t.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = t.from_numpy(sparse_mx.data)
        shape = t.Size(sparse_mx.shape)
        return t.sparse.FloatTensor(indices, values, shape)

    def _spmm(self,sp, emb):
        sp = sp.coalesce()
        cols = sp.indices()[1]
        rows = sp.indices()[0]
        col_segs = emb[cols] * t.unsqueeze(sp.values(),dim=1)
        result = t.zeros((sp.shape[0],emb.shape[1])).cuda()
        result.index_add_(0, rows, col_segs)
        return result

    def _sparse_dropout(self,mat, dropout):
        indices = mat.indices()
        values = nn.functional.dropout(mat.values(), p=dropout)
        size = mat.size()
        return t.sparse.FloatTensor(indices, values, size)

    def forward(self, test=False):
        if test and self.E_u is not None:
            return self.E_u, self.E_i
        for layer in range(1, self.layer_num+1):
            # GNN propagation
            self.Z_u_list[layer] = self.act(self._spmm(self._sparse_dropout(self.adj,self.dropout), self.E_i_list[layer-1]))
            self.Z_i_list[layer] = self.act(self._spmm(self._sparse_dropout(self.adj,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer-1]
            self.G_u_list[layer] = self.act(self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[layer-1]
            self.G_i_list[layer] = self.act(self.v_mul_s @ ut_eu)

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]  # + self.E_u_list[layer-1]
            self.E_i_list[layer] = self.Z_i_list[layer]  # + self.E_i_list[layer-1]

        # aggregate across layers
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        return self.E_u, self.E_i

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)

        iids = t.cat((poss,negs))

        cl_loss = 0
        for l in range(1,self.layer_num+1):
            u_mask = (t.rand(len(ancs))>0.5).float().cuda()

            gnn_u = nn.functional.normalize(self.Z_u_list[l][ancs], p=2, dim=1)
            hyper_u = nn.functional.normalize(self.G_u_list[l][ancs], p=2, dim=1)
            hyper_u = self.Ws[l-1](hyper_u)
            pos_score = t.exp((gnn_u*hyper_u).sum(1)/self.temp)
            neg_score = t.exp(gnn_u @ hyper_u.T/self.temp).sum(1)
            loss_s_u = ((-1 * t.log(pos_score/(neg_score+1e-8) + 1e-8))*u_mask).sum()
            cl_loss = cl_loss + loss_s_u

            i_mask = (t.rand(len(iids))>0.5).float().cuda()

            gnn_i = nn.functional.normalize(self.Z_i_list[l][iids], p=2, dim=1)
            hyper_i = nn.functional.normalize(self.G_i_list[l][iids], p=2, dim=1)
            hyper_i = self.Ws[l-1](hyper_i)
            pos_score = t.exp((gnn_i*hyper_i).sum(1)/self.temp)
            neg_score = t.exp(gnn_i @ hyper_i.T/self.temp).sum(1)
            loss_s_i = ((-1 * t.log(pos_score/(neg_score+1e-8) + 1e-8))*i_mask).sum()
            cl_loss = cl_loss + loss_s_i

        cl_loss = self.cl_weight * cl_loss
        loss = bpr_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'cl_loss': cl_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(test=True)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


class W_contrastive(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(t.empty(d,d)))

    def forward(self,x):
        return x @ self.W