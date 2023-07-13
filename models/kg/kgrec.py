
import random
import numpy as np
import torch
import torch.nn as nn
from logging import getLogger
import math
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.utils import softmax as scatter_softmax
from config.configurator import configs
import scipy.sparse as sp
from models.base_model import BaseModel


class AttnHGCN(nn.Module):
    """
    Heterogeneous Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_relations,
                node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(AttnHGCN, self).__init__()

        self.logger = getLogger()

        self.no_attn_convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.relation_emb = nn.Parameter(relation_emb)  # [n_relations - 1, in_channel]

        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))

        self.n_heads = 2
        self.d_k = channel // self.n_heads

        nn.init.xavier_uniform_(self.W_Q)

        self.n_hops = n_hops

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
    
    def non_attn_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg
        
    def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)

        relation_emb = relation_emb[edge_type - 1] # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads*self.d_k)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        # w_attn = self.ui_weighting(user_emb, entity_emb, inter_edge)
        # item_agg += w_attn.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg


    # @TimeCounter.count_time(warmup_interval=4)
    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                inter_edge, inter_edge_w, mess_dropout=True, item_attn=None):

        if item_attn is not None:
            item_attn = item_attn[inter_edge[1, :]]
            item_attn = scatter_softmax(item_attn, inter_edge[0, :])
            norm = scatter_sum(torch.ones_like(inter_edge[0, :]), inter_edge[0, :], dim=0, dim_size=user_emb.shape[0])
            norm = torch.index_select(norm, 0, inter_edge[0, :])
            item_attn = item_attn * norm
            inter_edge_w = inter_edge_w * item_attn

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        for i in range(self.n_hops):
            entity_emb, user_emb = self.shared_layer_agg(user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, self.relation_emb)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb
    
    def forward_ui(self, user_emb, item_emb, inter_edge, inter_edge_w, mess_dropout=True):
        item_res_emb = item_emb  # [n_entity, channel]
        for i in range(self.n_hops):
            user_emb, item_emb = self.ui_agg(user_emb, item_emb, inter_edge, inter_edge_w)
            """message dropout"""
            if mess_dropout:
                item_emb = self.dropout(item_emb)
                user_emb = self.dropout(user_emb)
            item_emb = F.normalize(item_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            item_res_emb = torch.add(item_res_emb, item_emb)
        return item_res_emb
    
    def forward_kg(self, entity_emb, edge_index, edge_type, mess_dropout=True):
        entity_res_emb = entity_emb
        for i in range(self.n_hops):
            entity_emb = self.kg_agg(entity_emb, edge_index, edge_type)
            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
        return entity_res_emb

    def ui_agg(self, user_emb, item_emb, inter_edge, inter_edge_w):
        num_items = item_emb.shape[0]
        item_emb = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_emb, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        user_emb = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg = scatter_sum(src=user_emb, index=inter_edge[1, :], dim_size=num_items, dim=0)
        return user_agg, item_agg

    def kg_agg(self, entity_emb, edge_index, edge_type):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = self.relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        return entity_agg

    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, edge_type=None, print=False, return_logits=False):
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        if edge_type is not None:
            key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_logits = edge_attn.mean(-1).detach()
        # softmax by head_node
        edge_attn_score = scatter_softmax(edge_attn_logits, head)
        # normalization by head_node degree
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        norm = torch.index_select(norm, 0, head)
        edge_attn_score = edge_attn_score * norm
        # print attn score
        if print:
            self.logger.info("edge_attn_score std: {}".format(edge_attn_score.std()))
        if return_logits:
            return edge_attn_score, edge_attn_logits
        return edge_attn_score


class Contrast(torch.nn.Module):
    def __init__(self, num_hidden: int, tau: float = 0.7):
        super(Contrast, self).__init__()
        self.tau: float = tau

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def self_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return (z1 * z2).sum(1)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.self_sim(z1, z2))
        rand_item = torch.randperm(z1.shape[0])
        neg_sim = f(self.self_sim(z1, z2[rand_item])) + f(self.self_sim(z2, z1[rand_item]))

        return -torch.log(between_sim / (between_sim + between_sim + neg_sim))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        h1 = self.mlp1(z1)
        h2 = self.mlp2(z2)
        loss = self.loss(h1, h2).mean()
        return loss


def _adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score, keep_rate):
    _, least_attn_edge_id = torch.topk(-edge_attn_score,
                                       int((1-keep_rate) * edge_attn_score.shape[0]), sorted=False)
    cl_kg_mask = torch.ones_like(edge_attn_score).bool()
    cl_kg_mask[least_attn_edge_id] = False
    cl_kg_edge = edge_index[:, cl_kg_mask]
    cl_kg_type = edge_type[cl_kg_mask]
    return cl_kg_edge, cl_kg_type

def _adaptive_ui_drop_cl(item_attn_mean, inter_edge, inter_edge_w, keep_rate=0.7, samp_func = "torch"):
    inter_attn_prob = item_attn_mean[inter_edge[1]]
    # add gumbel noise
    noise = -torch.log(-torch.log(torch.rand_like(inter_attn_prob)))
    """ prob based drop """
    inter_attn_prob = inter_attn_prob + noise
    inter_attn_prob = F.softmax(inter_attn_prob, dim=0)

    if samp_func == "np":
        # we observed abnormal behavior of torch.multinomial on mind
        sampled_edge_idx = np.random.choice(np.arange(inter_edge_w.shape[0]), size=int(keep_rate * inter_edge_w.shape[0]), replace=False, p=inter_attn_prob.cpu().numpy())
    else:
        sampled_edge_idx = torch.multinomial(inter_attn_prob, int(keep_rate * inter_edge_w.shape[0]), replacement=False)

    return inter_edge[:, sampled_edge_idx], inter_edge_w[sampled_edge_idx]/keep_rate


def _relation_aware_edge_sampling(edge_index, edge_type, n_relations, samp_rate=0.5):
    # exclude interaction
    for i in range(n_relations - 1):
        edge_index_i, edge_type_i = _edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
        if i == 0:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
        else:
            edge_index_sampled = torch.cat(
                [edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat(
                [edge_type_sampled, edge_type_i], dim=0)
    return edge_index_sampled, edge_type_sampled


def _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    topk_egde_id = topk_egde_id.cpu().numpy()
    topk_mask = np.zeros(n_edges, dtype=bool)
    topk_mask[topk_egde_id] = True
    # add another group of random mask
    random_indices = np.random.choice(
        n_edges, size=topk_egde_id.shape[0], replace=False)
    random_mask = np.zeros(n_edges, dtype=bool)
    random_mask[random_indices] = True
    # combine two masks
    mask = topk_mask | random_mask

    remain_edge_index = edge_index[:, ~mask]
    remain_edge_type = edge_type[~mask]
    masked_edge_index = edge_index[:, mask]
    masked_edge_type = edge_type[mask]

    return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask

def _edge_sampling(edge_index, edge_type, samp_rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * samp_rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]


def _sparse_dropout(i, v, keep_rate=0.5):
    noise_shape = i.shape[1]

    random_tensor = keep_rate
    # the drop rate is 1 - keep_rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    i = i[:, dropout_mask]
    v = v[dropout_mask] / keep_rate

    return i, v


class KGRec(BaseModel):
    def __init__(self, data_handler):
        super(KGRec, self).__init__(data_handler)
        self.logger = getLogger()

        self.n_users = configs['data']['user_num']
        self.n_items = configs['data']['item_num']
        self.n_relations = configs['data']['relation_num']
        self.n_entities = configs['data']['entity_num']  # include items
        self.n_nodes = configs['data']['node_num']  # n_users + n_entities

        self.decay = configs['model']['decay_weight']
        self.emb_size = configs['model']['embedding_size']
        self.context_hops = configs['model']['layer_num']
        self.node_dropout = configs['model']['node_dropout']
        self.node_dropout_rate = configs['model']['node_dropout_rate']
        self.mess_dropout = configs['model']['mess_dropout']
        self.mess_dropout_rate = configs['model']['mess_dropout_rate']
        self.device = configs['device']
        
        self.mae_coef = configs['model']['mae_coef']
        self.mae_msize = configs['model']['mae_msize']
        self.cl_coef = configs['model']['cl_coef']
        self.tau = configs['model']['tau']
        self.cl_drop = configs['model']['cl_drop_ratio']

        self.samp_func = configs['model']['samp_func']

        self.adj_mat = self._make_si_norm_adj(data_handler.ui_mat)
        
        self.inter_edge, self.inter_edge_w = self._convert_sp_mat_to_tensor(
            self.adj_mat)

        self.edge_index, self.edge_type = self._get_edges(data_handler.kg_edges)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = AttnHGCN(channel=self.emb_size,
                       n_hops=self.context_hops,
                       n_users=self.n_users,
                       n_relations=self.n_relations,
                       node_dropout_rate=self.node_dropout_rate,
                       mess_dropout_rate=self.mess_dropout_rate)
        self.contrast_fn = Contrast(self.emb_size, tau=self.tau)

        # self.print_shapes()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
    
    def _make_si_norm_adj(self, adj_mat):
        # D^{-1}A
        rowsum = np.array(adj_mat.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj_mat)
        return norm_adj.tocoo()

    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return i.to(self.device), v.to(self.device)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, kg_edges):
        graph_tensor = torch.tensor(kg_edges)  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def cal_loss(self, batch_data):
        user, pos_item, neg_item = batch_data[:3]
        epoch_start = 0 in user.cpu().tolist()

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        """node dropout"""
        # 1. graph sprasification;
        edge_index, edge_type = _relation_aware_edge_sampling(
            self.edge_index, self.edge_type, self.n_relations, 1-self.node_dropout_rate)
        # 2. compute rationale scores;
        edge_attn_score, edge_attn_logits = self.gcn.norm_attn_computer(
            item_emb, edge_index, edge_type, print=epoch_start, return_logits=True)
        # for adaptive UI MAE
        item_attn_mean_1 = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=self.n_entities)
        item_attn_mean_1[item_attn_mean_1 == 0.] = 1.
        item_attn_mean_2 = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=self.n_entities)
        item_attn_mean_2[item_attn_mean_2 == 0.] = 1.
        item_attn_mean = (0.5 * item_attn_mean_1 + 0.5 * item_attn_mean_2)[:self.n_items]
        # for adaptive MAE training
        std = torch.std(edge_attn_score).detach()
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score = edge_attn_score + noise
        topk_v, topk_attn_edge_id = torch.topk(
            edge_attn_score, self.mae_msize, sorted=False)
        top_attn_edge_type = edge_type[topk_attn_edge_id]

        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, mask_bool = _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_attn_edge_id)

        inter_edge, inter_edge_w = _sparse_dropout(
            self.inter_edge, self.inter_edge_w, 1-self.node_dropout_rate)

        # rec task
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                item_emb,
                                                enc_edge_index,
                                                enc_edge_type,
                                                inter_edge,
                                                inter_edge_w,
                                                mess_dropout=self.mess_dropout,
                                                )
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        loss, rec_loss, reg_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        # MAE task with dot-product decoder
        # mask_size, 2, channel
        node_pair_emb = entity_gcn_emb[masked_edge_index.t()]
        # mask_size, channel
        masked_edge_emb = self.gcn.relation_emb[masked_edge_type-1]
        mae_loss = self.mae_coef * \
            self.create_mae_loss(node_pair_emb, masked_edge_emb)

        # CL task
        """adaptive sampling"""
        cl_kg_edge, cl_kg_type = _adaptive_kg_drop_cl(
            edge_index, edge_type, edge_attn_score, keep_rate=1-self.cl_drop)
        cl_ui_edge, cl_ui_w = _adaptive_ui_drop_cl(
            item_attn_mean, inter_edge, inter_edge_w, 1-self.cl_drop, samp_func=self.samp_func)

        item_agg_ui = self.gcn.forward_ui(
            user_emb, item_emb[:self.n_items], cl_ui_edge, cl_ui_w)
        item_agg_kg = self.gcn.forward_kg(
            item_emb, cl_kg_edge, cl_kg_type)[:self.n_items]
        cl_loss = self.cl_coef * self.contrast_fn(item_agg_ui, item_agg_kg)

        loss_dict = {
            "rec_loss": loss.item(),
            "mae_loss": mae_loss.item(),
            "cl_loss": cl_loss.item(),
        }
        return loss + mae_loss + cl_loss, loss_dict

    def calc_topk_attn_edge(self, entity_emb, edge_index, edge_type, k):
        edge_attn_score = self.gcn.norm_attn_computer(
            entity_emb, edge_index, edge_type, return_logits=True)
        positive_mask = edge_attn_score > 0
        edge_attn_score = edge_attn_score[positive_mask]
        edge_index = edge_index[:, positive_mask]
        edge_type = edge_type[positive_mask]
        topk_values, topk_indices = torch.topk(
            edge_attn_score, k, sorted=False)
        return edge_index[:, topk_indices], edge_type[topk_indices]

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        entity_emb, user_emb = self.gcn(user_emb,
                        item_emb,
                        self.edge_index,
                        self.edge_type,
                        self.inter_edge,
                        self.inter_edge_w,
                        mess_dropout=False)[:2]
        return user_emb, entity_emb[:self.n_items]

    def rating(self, u_emb, i_emb):
        return torch.matmul(u_emb, i_emb.t())
    
    def full_predict(self, batch_data):
        users = batch_data[0]
        user_emb, item_emb = self.generate()
        batch_u = user_emb[users]
        return batch_u.mm(item_emb.t())

    # @TimeCounter.count_time(warmup_interval=4)
    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        if torch.isnan(mf_loss):
            raise ValueError("nan mf_loss")

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = - \
            torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores

    def print_shapes(self):
        self.logger.info("########## Model HPs ##########")
        self.logger.info("tau: {}".format(self.contrast_fn.tau))
        self.logger.info("cL_drop: {}".format(self.cl_drop))
        self.logger.info("cl_coef: {}".format(self.cl_coef))
        self.logger.info("mae_coef: {}".format(self.mae_coef))
        self.logger.info("mae_msize: {}".format(self.mae_msize))
        self.logger.info("########## Model Parameters ##########")
        self.logger.info("context_hops: %d", self.context_hops)
        self.logger.info("node_dropout: %d", self.node_dropout)
        self.logger.info("node_dropout_rate: %.1f", self.node_dropout_rate)
        self.logger.info("mess_dropout: %d", self.mess_dropout)
        self.logger.info("mess_dropout_rate: %.1f", self.mess_dropout_rate)
        self.logger.info('all_embed: {}'.format(self.all_embed.shape))
        self.logger.info('interact_mat: {}'.format(self.inter_edge.shape))
        self.logger.info('edge_index: {}'.format(self.edge_index.shape))
        self.logger.info('edge_type: {}'.format(self.edge_type.shape))

    def generate_kg_drop(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        edge_index, edge_type = _edge_sampling(
        self.edge_index, self.edge_type, self.kg_drop_test_keep_rate)
        return self.gcn(user_emb,
                        item_emb,
                        edge_index,
                        edge_type,
                        self.inter_edge,
                        self.inter_edge_w,
                        mess_dropout=False)[:2]
    
    def generate_global_attn_score(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        edge_attn_score = self.gcn.norm_attn_computer(
            item_emb, self.edge_index, self.edge_type)

        return edge_attn_score, self.edge_index, self.edge_type
