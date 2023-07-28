import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_sum, scatter_softmax
from logging import getLogger
from config.configurator import configs
import random
import scipy.sparse as sp
from models.loss_utils import cal_bpr_loss
from models.base_model import BaseModel


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def _edge_sampling(edge_index, edge_type, rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]


def _sparse_dropout(x, rate=0.5):
    noise_shape = x._nnz()

    random_tensor = rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)
    i = x._indices()
    v = x._values()

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    return out * (1. / (1 - rate))


class RGAT(nn.Module):
    def __init__(self, channel, n_hops,
                 mess_dropout_rate=0.4):
        super(RGAT, self).__init__()
        self.mess_dropout_rate = mess_dropout_rate
        self.W = nn.Parameter(torch.empty(size=(channel, channel)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*channel, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(2*channel, channel)

        self.leakyrelu = nn.LeakyReLU(0.2)

        self.n_hops = n_hops

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def agg(self, entity_emb, relation_emb, kg):
        edge_index, edge_type = kg
        head, tail = edge_index
        a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
        e_input = torch.multiply(
            self.fc(a_input), relation_emb[edge_type]).sum(-1)  # N,e
        e = self.leakyrelu(e_input)  # (N, e_num)
        e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = entity_emb[tail] * e.view(-1, 1)
        agg_emb = scatter_sum(agg_emb, head, dim=0,
                              dim_size=entity_emb.shape[0])
        # agg_emb = agg_emb + entity_emb
        return agg_emb

    def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
        entity_res_emb = entity_emb
        for _ in range(self.n_hops):
            entity_emb = self.agg(entity_emb, relation_emb, kg)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
        return entity_emb


class KGCL(BaseModel):
    def __init__(self, data_handler):
        super(KGCL, self).__init__(data_handler)
        self.n_users = configs['data']['user_num']
        self.n_items = configs['data']['item_num']
        self.n_relations = configs['data']['relation_num']
        self.n_entities = configs['data']['entity_num']  # include items
        self.n_nodes = configs['data']['node_num']  # n_users + n_entities

        self.tau = 0.2
        self.cl_weight = 0.1
        self.mu = 0.95

        self.decay = configs['model']['decay_weight']
        self.emb_size = configs['model']['embedding_size']
        self.context_hops = configs['model']['layer_num_kg']
        self.layer_num = configs['model']['layer_num']
        self.node_dropout = configs['model']['node_dropout']
        self.node_dropout_rate = configs['model']['node_dropout_rate']
        self.mess_dropout = configs['model']['mess_dropout']
        self.mess_dropout_rate = configs['model']['mess_dropout_rate']
        self.device = configs['device']

        self.ui_mat = data_handler.ui_mat
        self.norm_adj = self._get_norm_adj_mat(
            data_handler.ui_mat).to(self.device)
        self.kg_dict = data_handler.kg_dict
        # self.edge_index, self.edge_type = self._get_edges(
        #     data_handler.kg_edges)
        self.edge_index, self.edge_type = self._samp_edge_from_dict(
            self.kg_dict, triplet_num=15)

        self.all_embed = nn.init.normal_(
            torch.empty(self.n_nodes, self.emb_size), std=0.1)
        self.relation_embed = nn.init.normal_(
            torch.empty(self.n_relations, self.emb_size), std=0.1)
        self.all_embed = nn.Parameter(self.all_embed)
        self.relation_embed = nn.Parameter(self.relation_embed)
        self.rgat = RGAT(self.emb_size, self.context_hops,
                         self.mess_dropout_rate)

    def _get_edges(self, kg_edges):
        graph_tensor = torch.tensor(kg_edges)  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_norm_adj_mat(self, ui_mat):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = ui_mat
        inter_M_t = ui_mat.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col +
                self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL.to(self.device)

    def _samp_edge_from_dict(self, kg_dict, triplet_num=15):
        samp_edges = []
        for h in kg_dict:
            t_list = kg_dict[h]
            if len(t_list) > triplet_num:
                samp_edges_i = random.sample(
                    t_list, triplet_num)
            else:
                samp_edges_i = t_list
            for r, t in samp_edges_i:
                samp_edges.append([h, t, r])
        return self._get_edges(samp_edges)

    def _get_ui_aug_views(self, kg_stability, mu):
        kg_stability = torch.exp(kg_stability)
        kg_weights = (kg_stability - kg_stability.min()) / \
            (kg_stability.max() - kg_stability.min())
        kg_weights = kg_weights.where(
            kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)

        # overall probability of keep
        weights = mu/torch.mean(kg_weights)*(kg_weights)
        weights = weights.where(
            weights < 0.95, torch.ones_like(weights) * 0.95)

        items_in_edges = self.ui_mat.col
        edge_weights = weights[items_in_edges]
        edge_mask = torch.bernoulli(edge_weights).bool().cpu()
        keep_rate = edge_mask.sum().item() / edge_mask.size()[0]
        print(f"u-i edge keep ratio: {keep_rate:.2f}")
        # drop
        col = self.ui_mat.col
        row = self.ui_mat.row
        v = self.ui_mat.data

        col = col[edge_mask]
        row = row[edge_mask]
        v = v[edge_mask]

        masked_ui_mat = sp.coo_matrix(
            (v, (row, col)), shape=(self.n_users, self.n_items))
        return self._get_norm_adj_mat(masked_ui_mat)

    @torch.no_grad()
    def get_aug_views(self):
        edge_index, edge_type = self.edge_index, self.edge_type
        # edge_index, edge_type = self._samp_edge_from_dict(self.kg_dict)
        # edge_index, edge_type = _edge_sampling(self.edge_index, self.edge_type, 0.5)
        entity_emb = self.all_embed[self.n_users:, :]
        kg_view_1 = _edge_sampling(
            edge_index, edge_type, 0.5)
        kg_view_2 = _edge_sampling(
            edge_index, edge_type, 0.5)
        kg_v1_ro = self.rgat(entity_emb, self.relation_embed, kg_view_1, False)[
            :self.n_items, :]
        kg_v2_ro = self.rgat(entity_emb, self.relation_embed, kg_view_2, False)[
            :self.n_items, :]
        stability = F.cosine_similarity(kg_v1_ro, kg_v2_ro, dim=-1)
        ui_view_1 = self._get_ui_aug_views(stability, mu=self.mu)
        ui_view_2 = self._get_ui_aug_views(stability, mu=self.mu)
        return kg_view_1, kg_view_2, ui_view_1, ui_view_2

    def cal_loss(self, batch_data):
        user, pos_item, neg_item = batch_data[:3]
        # kg_view_1, kg_view_2, ui_view_1, ui_view_2 = self.get_aug_views()
        kg_view_1, kg_view_2, ui_view_1, ui_view_2 = batch_data[3:7]

        if self.node_dropout:
            g_droped = _sparse_dropout(self.norm_adj, self.node_dropout_rate)
            edge_index, edge_type = _edge_sampling(
                self.edge_index, self.edge_type, 1-self.node_dropout_rate)
        else:
            g_droped = self.norm_adj
            edge_index, edge_type = self.edge_index, self.edge_type

        user_emb, item_emb = self.forward(edge_index, edge_type, g_droped)
        u_e = user_emb[user]
        pos_e, neg_e = item_emb[pos_item], item_emb[neg_item]
        rec_loss, reg_loss = self._bpr_loss(u_e, pos_e, neg_e)

        # CL
        users_v1_ro, items_v1_ro = self.forward(
            kg_view_1[0], kg_view_1[1], ui_view_1)
        users_v2_ro, items_v2_ro = self.forward(
            kg_view_2[0], kg_view_2[1], ui_view_2)
        user_cl_loss = self._infonce_overall(
            users_v1_ro[user], users_v2_ro[user], users_v2_ro)
        item_cl_loss = self._infonce_overall(
            items_v1_ro[pos_item], items_v2_ro[pos_item], items_v2_ro)

        cl_loss = self.cl_weight * (user_cl_loss + item_cl_loss)
        loss = rec_loss + self.decay * reg_loss + cl_loss

        loss_dict = {
            "rec_loss": rec_loss.item(),
            "cl_loss": cl_loss.item(),
        }
        return loss, loss_dict

    def forward(self, edge_index, edge_type, g):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]

        entity_emb = self.rgat(entity_emb, self.relation_embed, [
                               edge_index, edge_type], self.mess_dropout & self.training)

        all_emb = torch.cat([user_emb, entity_emb[:self.n_items, :]], dim=0)
        emb_list = [all_emb]
        for layer in range(self.layer_num):
            all_emb = torch.sparse.mm(g, all_emb)
            emb_list.append(all_emb)
        all_emb = torch.stack(emb_list, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items])
        return user_emb, item_emb

    def generate(self):
        return self.forward(
            self.edge_index, self.edge_type, self.norm_adj)
    
    def rating(self, u_emb, i_emb):
        return torch.matmul(u_emb, i_emb.t())
    
    def full_predict(self, batch_data):
        users = batch_data[0]
        user_emb, item_emb = self.forward(
            self.edge_index, self.edge_type, self.norm_adj)
        return torch.matmul(user_emb[users], item_emb.t())

    def _bpr_loss(self, users_emb, pos_emb, neg_emb):
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2))/float(users_emb.shape[0])
        loss = cal_bpr_loss(users_emb, pos_emb, neg_emb)
        return loss, reg_loss

    def cal_kg_loss(self, batch_data):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        h, r, pos_t, neg_t = batch_data
        entity_emb = self.all_embed[self.n_users:, :]
        # (kg_batch_size, relation_dim)
        r_embed = self.relation_embed[r]
        h_embed = entity_emb[h]              # (kg_batch_size, entity_dim)
        pos_t_embed = entity_emb[pos_t]      # (kg_batch_size, entity_dim)
        neg_t_embed = entity_emb[neg_t]      # (kg_batch_size, entity_dim)
        # Equation (1)
        pos_score = torch.sum(
            torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)     # (kg_batch_size)
        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + \
            _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def _infonce_overall(self, z1, z2, z_all):

        def sim(z1: torch.Tensor, z2: torch.Tensor):
            if z1.size()[0] == z2.size()[0]:
                return F.cosine_similarity(z1, z2)
            else:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                return torch.mm(z1, z2.t())

        def f(x): return torch.exp(x / self.tau)
        # batch_size
        between_sim = f(sim(z1, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim = f(sim(z1, z_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
        return loss

    def print_shapes(self):
        self.logger.info("########## Model HPs ##########")
        self.logger.info("tau: {}".format(self.tau))
        self.logger.info("########## Model Parameters ##########")
        self.logger.info("context_hops: %d", self.context_hops)
        self.logger.info("node_dropout: %d", self.node_dropout)
        self.logger.info("node_dropout_rate: %.1f", self.node_dropout_rate)
        self.logger.info("mess_dropout: %d", self.mess_dropout)
        self.logger.info("mess_dropout_rate: %.1f", self.mess_dropout_rate)
        self.logger.info('all_embed: {}'.format(self.all_embed.shape))
        self.logger.info('interact_mat: {}'.format(self.norm_adj.shape))
        self.logger.info('edge_index: {}'.format(self.edge_index.shape))
        self.logger.info('edge_type: {}'.format(self.edge_type.shape))
