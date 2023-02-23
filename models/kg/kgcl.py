import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_sum, scatter_softmax
from logging import getLogger
from config.configurator import configs


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
        agg_emb = agg_emb + entity_emb
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


class KGCL(nn.Module):
    def __init__(self, data_handler):
        super(KGCL, self).__init__()
        self.n_users = configs['data']['user_num']
        self.n_items = configs['data']['item_num']
        self.n_relations = configs['data']['relation_num']
        self.n_entities = configs['data']['entity_num']  # include items
        self.n_nodes = configs['data']['node_num']  # n_users + n_entities

        self.tau = 0.2
        self.cl_weight = 0.1
        self.mu = 0.9

        self.decay = configs['model']['decay_weight']
        self.emb_size = configs['model']['embedding_size']
        self.context_hops = configs['model']['layer_num']
        self.node_dropout = configs['model']['node_dropout']
        self.node_dropout_rate = configs['model']['node_dropout_rate']
        self.mess_dropout = configs['model']['mess_dropout']
        self.mess_dropout_rate = configs['model']['mess_dropout_rate']
        self.device = configs['device']

        self.adj_mat = self._convert_sp_mat_to_sp_tensor(
            data_handler.ui_norm_adj).to(self.device)
        self.edge_index, self.edge_type = self._get_edges(
            data_handler.kg_edges)

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

        items_in_edges = self.adj_mat._indices()[1, :]
        edge_weights = weights[items_in_edges]
        edge_mask = torch.bernoulli(edge_weights).to(torch.bool)
        keep_rate = edge_mask.sum().item() / edge_mask.size()[0]
        print(f"u-i edge keep ratio: {keep_rate:.2f}")
        # drop
        i = self.adj_mat._indices()
        v = self.adj_mat._values()

        i = i[:, edge_mask]
        v = v[edge_mask]

        out = torch.sparse.FloatTensor(
            i, v, self.adj_mat.shape).to(self.adj_mat.device)
        return out * (1. / keep_rate)

    @torch.no_grad()
    def get_aug_views(self):
        entity_emb = self.all_embed[self.n_users:, :]
        kg_view_1 = _edge_sampling(
            self.edge_index, self.edge_type, self.node_dropout_rate)
        kg_view_2 = _edge_sampling(
            self.edge_index, self.edge_type, self.node_dropout_rate)
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
        kg_view_1, kg_view_2, ui_view_1, ui_view_2 = self.get_aug_views()
        # kg_view_1, kg_view_2, ui_view_1, ui_view_2 = batch_data['aug_views']

        if self.node_dropout:
            g_droped = _sparse_dropout(self.adj_mat, self.node_dropout_rate)
            edge_index, edge_type = _edge_sampling(
                self.edge_index, self.edge_type, self.node_dropout_rate)
        else:
            g_droped = self.adj_mat
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

        user_embs = [user_emb]
        entity_embs = [entity_emb]
        for layer in range(self.context_hops):
            item_emb_l = torch.sparse.mm(g.t(), user_embs[-1])
            user_emb_l = torch.sparse.mm(g, entity_embs[-1])
            user_embs.append(user_emb_l)
            entity_embs.append(item_emb_l)
        user_embs = torch.stack(user_embs, dim=1)
        entity_embs = torch.stack(entity_embs, dim=1)
        # print(embs.size())
        users = torch.mean(user_embs, dim=1)
        items = torch.mean(entity_embs, dim=1)
        return users, items

    def full_predict(self, batch_data):
        users = batch_data[0]
        user_emb, item_emb = self.forward(
            self.edge_index, self.edge_type, self.adj_mat)
        return torch.matmul(user_emb[users], item_emb.t())

    def _bpr_loss(self, users_emb, pos_emb, neg_emb):
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2))/float(users_emb.shape[0])
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # mean or sum
        loss = torch.sum(
            torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        if (torch.isnan(loss).any().tolist()):
            print("nan loss")
            return None
        return loss, reg_loss

    def cal_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
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
        self.logger.info('interact_mat: {}'.format(self.adj_mat.shape))
        self.logger.info('edge_index: {}'.format(self.edge_index.shape))
        self.logger.info('edge_type: {}'.format(self.edge_type.shape))
