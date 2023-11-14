import numpy as np
from numpy import random
from random import random
import pickle
import scipy.sparse as sp
import gc
import datetime
import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.data as dataloader
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from config.configurator import configs
from models.base_model import BaseModel

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False
torch.autograd.set_detect_anomaly(True)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)



class KMCLR(BaseModel):
    def __init__(self, data_handler):
        super(KMCLR, self).__init__(data_handler)

    # def __init__(self, userNum, itemNum, behavior, behavior_mats):  
        self.data_handler = data_handler       
        self.userNum = data_handler.userNum
        self.itemNum = data_handler.itemNum
        self.behavior = data_handler.behaviors
        self.behavior_mats = data_handler.behavior_mats
        
        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats)


    def forward(self):
        user_embed, item_embed, user_embeds, item_embeds = self.gcn()
        return user_embed, item_embed, user_embeds, item_embeds 


    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _ = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


    def kg_init_transR(self, kgdataset, recommend_model, opt, index):
        Recmodel = recommend_model
        Recmodel.train()
        kgloader = DataLoader(kgdataset, batch_size=4096, drop_last=True)
        trans_loss = 0.
        for data in tqdm(kgloader, total=len(kgloader), disable=True):
            heads = data[0].to(configs['device'])
            relations = data[1].to(configs['device'])
            pos_tails = data[2].to(configs['device'])
            neg_tails = data[3].to(configs['device'])
            kg_batch_loss = Recmodel.calc_kg_loss_transR(heads, relations, pos_tails, neg_tails, index)
            trans_loss += kg_batch_loss / len(kgloader)
            opt.zero_grad()
            kg_batch_loss.backward()
            opt.step()


    def kg_init_TATEC(self, kgdataset, recommend_model, opt, index):
        Recmodel = recommend_model
        Recmodel.train()
        kgloader = DataLoader(kgdataset, batch_size=4096, drop_last=True)
        trans_loss = 0.
        for data in tqdm(kgloader, total=len(kgloader), disable=True):
            heads = data[0].to(configs['device'])
            relations = data[1].to(configs['device'])
            pos_tails = data[2].to(configs['device'])
            neg_tails = data[3].to(configs['device'])
            kg_batch_loss = Recmodel.calc_kg_loss_TATEC(heads, relations, pos_tails, neg_tails, index)
            trans_loss += kg_batch_loss / len(kgloader)
            opt.zero_grad()
            kg_batch_loss.backward()
            opt.step()


    def BPR_train_contrast(self, dataset, recommend_model, loss_class, contrast_model, contrast_views, optimizer, neg_k=1, w=None, ssl_reg=0.1):
        Recmodel: Kg_Model.Model = recommend_model
        Recmodel.train()
        bpr: utils.BPRLoss = loss_class
        batch_size = configs['model']['bpr_batch_size']
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

        aver_loss = 0.
        aver_loss_main = 0.
        aver_loss_ssl = 0.

        uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]

        for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=True):
            batch_users = train_data[0].long().to(configs['device'])
            batch_pos = train_data[1].long().to(configs['device'])
            batch_neg = train_data[2].long().to(configs['device'])

            l_main = bpr.compute(batch_users, batch_pos, batch_neg)
            l_ssl = list()
            items = batch_pos

            usersv1_ro, itemsv1_ro = Recmodel.view_computer_all(uiv1, index=0)
            usersv2_ro, itemsv2_ro = Recmodel.view_computer_all(uiv2, index=1)

            items_uiv1 = itemsv1_ro[items]
            items_uiv2 = itemsv2_ro[items]
            l_item = contrast_model.grace_loss(items_uiv1, items_uiv2)

            users = batch_users
            users_uiv1 = usersv1_ro[users]
            users_uiv2 = usersv2_ro[users]
            l_user = contrast_model.grace_loss(users_uiv1, users_uiv2)
            l_ssl.extend([l_user * ssl_reg, l_item * ssl_reg])

            if l_ssl:
                l_ssl = torch.stack(l_ssl).sum()
                #l_all = l_ssl
                l_all = l_main + l_ssl
                aver_loss_ssl += l_ssl.cpu().item()
            else:
                l_all = l_main
                #l_all = l_ssl
            optimizer.zero_grad()
            l_all.backward()
            optimizer.step()

            aver_loss_main += l_main.cpu().item()
            aver_loss += l_all.cpu().item()

        # timer.zero()





class GCN(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats):
        super(GCN, self).__init__()  
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = configs['model']['embedding_size'] 

        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.user_embedding, self.item_embedding = self.init_embedding()         
        
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(configs['model']['drop_rate'])

        self.gnn_layer = configs['model']['layer_num']
        self.layers = nn.ModuleList()
        for i in range(0, self.gnn_layer):  
            self.layers.append(GCNLayer(configs['model']['embedding_size'], configs['model']['embedding_size'], self.userNum, self.itemNum, self.behavior, self.behavior_mats))  

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
        init.xavier_uniform_(i_concatenation_w)
        init.xavier_uniform_(u_concatenation_w)
        init.xavier_uniform_(i_input_w)
        init.xavier_uniform_(u_input_w)

        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w

    def forward(self, user_embedding_input=None, item_embedding_input=None):
        all_user_embeddings = []
        all_item_embeddings = []
        all_user_embeddingss = []
        all_item_embeddingss = []

        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight

        for i, layer in enumerate(self.layers):
            user_embedding, item_embedding, user_embeddings, item_embeddings = layer(user_embedding, item_embedding)

            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)
            
        user_embedding = torch.cat(all_user_embeddings, dim=1)
        item_embedding = torch.cat(all_item_embeddings, dim=1)
        user_embeddings = torch.cat(all_user_embeddingss, dim=2)
        item_embeddings = torch.cat(all_item_embeddingss, dim=2)

        user_embedding = torch.matmul(user_embedding , self.u_concatenation_w)
        item_embedding = torch.matmul(item_embedding , self.i_concatenation_w)
        user_embeddings = torch.matmul(user_embeddings , self.u_concatenation_w)
        item_embeddings = torch.matmul(item_embeddings , self.i_concatenation_w)
            

        return user_embedding, item_embedding, user_embeddings, item_embeddings



class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats):
        super(GCNLayer, self).__init__()

        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.userNum = userNum
        self.itemNum = itemNum

        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)

    def forward(self, user_embedding, item_embedding):

        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)

        for i in range(len(self.behavior)):
            user_embedding_list[i] = torch.spmm(self.behavior_mats[i]['A'], item_embedding)
            item_embedding_list[i] = torch.spmm(self.behavior_mats[i]['AT'], user_embedding)

        user_embeddings = torch.stack(user_embedding_list, dim=0) 
        item_embeddings = torch.stack(item_embedding_list, dim=0)

        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w))
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w))

        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w))
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w))

        return user_embedding, item_embedding, user_embeddings, item_embeddings             



def drop_edge_random(item2entities, p_drop, padding):
    res = dict()
    for item, es in item2entities.items():
        new_es = list()
        for e in es.tolist():
            if (random() > p_drop):
                new_es.append(e)
            else:
                new_es.append(padding)
        res[item] = torch.IntTensor(new_es).to(configs['device'])
    return res


class Contrast(nn.Module):
    def __init__(self, gcn_model, tau):
        super(Contrast, self).__init__()
        self.gcn_model: Model = gcn_model
        self.tau = tau

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def pair_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.pair_sim(z1, z2))
        return torch.sum(-torch.log(between_sim.diag() / (between_sim.sum(1) - between_sim.diag())))

    def grace_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = False, batch_size: int = 0):
        h1 = z1
        h2 = z2
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            # l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        return l1

    def get_kg_views(self):
        kg = self.gcn_model.kg_dict
        view1 = drop_edge_random(kg, configs['model']['kg_p_drop'],
                                 self.gcn_model.num_entities)
        view2 = drop_edge_random(kg, configs['model']['kg_p_drop'],
                                 self.gcn_model.num_entities)
        return view1, view2

    def item_kg_stability(self, view1, view2, index):
        kgv1_ro = self.gcn_model.cal_item_embedding_from_kg(view1, index=index)
        kgv2_ro = self.gcn_model.cal_item_embedding_from_kg(view2, index=index)
        sim = self.sim(kgv1_ro, kgv2_ro)
        return sim


    def get_adj_mat(self, tmp_adj):
        adj_mat = tmp_adj + tmp_adj.T

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum+1e-8, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(
            coo.shape)).coalesce().to(configs['device'])
        g.requires_grad = False
        return g

    def ui_batch_drop_weighted(self, item_mask, start, end):
        item_mask = item_mask.cpu().numpy()
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        item_np = self.gcn_model.dataset.trainItem
        user_np = self.gcn_model.dataset.trainUser
        indices = np.where((user_np >= start) & (user_np < end))[0]
        batch_item = item_np[indices]
        batch_user = user_np[indices]

        keep_idx = list()
        for u, i in zip(batch_user, batch_item):
            if item_mask[u - start, i]:
                keep_idx.append([u, i])

        keep_idx = np.array(keep_idx)
        user_np = keep_idx[:, 0]
        item_np = keep_idx[:, 1] + self.gcn_model.num_users
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np)),
            shape=(n_nodes, n_nodes))
        return tmp_adj

    def get_ui_views_weighted_with_uemb(self, item_stabilities, user_score, start, end, init_view):
        user_item_stabilities = F.softmax(user_score, dim=-1) * item_stabilities
        k = (1 - 0.6) / (user_item_stabilities.max() - user_item_stabilities.min())
        weights = 0.6 + k * (user_item_stabilities - user_item_stabilities.min())
        item_mask = torch.bernoulli(weights).to(torch.bool)
        tmp_adj = self.ui_batch_drop_weighted(item_mask, start, end)
        if init_view != None:
            tmp_adj = init_view + tmp_adj
        return tmp_adj

    def get_ui_kg_view(self, aug_side="both"):
        if aug_side == "ui":
            kgv1, kgv2 = None, None
            kgv3, kgv4 = None, None
        else:
            kgv1, kgv2 = self.get_kg_views()
            kgv3, kgv4 = self.get_kg_views()

        stability1 = self.item_kg_stability(kgv1, kgv2, index=0).to(configs['device'])
        stability2 = self.item_kg_stability(kgv3, kgv4, index=1).to(configs['device'])
        u = self.gcn_model.embedding_user.weight
        i1 = self.gcn_model.emb_item_list[0].weight
        i2 = self.gcn_model.emb_item_list[1].weight

        user_length = self.gcn_model.num_users
        size = 2048
        step = user_length // size + 1
        init_view1, init_view2 = None, None
        for s in range(step):
            start = s * size
            end = (s + 1) * size
            u_i_s1 = u[start:end] @ i1.T
            u_i_s2 = u[start:end] @ i2.T
            uiv1_batch_view = self.get_ui_views_weighted_with_uemb(stability1, u_i_s1, start, end, init_view1)
            uiv2_batch_view = self.get_ui_views_weighted_with_uemb(stability2, u_i_s2, start, end, init_view2)
            init_view1 = uiv1_batch_view
            init_view2 = uiv2_batch_view

        uiv1 = self.get_adj_mat(init_view1)
        uiv2 = self.get_adj_mat(init_view2)

        contrast_views = {
            "uiv1": uiv1,
            "uiv2": uiv2
        }
        return contrast_views


class KGModel(nn.Module):
    def __init__(self, dataset, kg_dataset):
        super(KGModel, self).__init__()
        self.dataset = dataset
        self.kg_dataset = kg_dataset
        self.__init_weight()
        self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()


    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        print("user:{}, item:{}, entity:{}".format(self.num_users, self.num_items, self.num_entities))
        self.latent_dim = configs['model']['latent_dim_rec']
        self.n_layers = configs['model']['lightGCN_n_layers']
        self.keep_prob = configs['model']['keep_prob']
        self.A_split = configs['model']['A_split']

        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.emb_item_list = nn.ModuleList([
            torch.nn.Embedding(self.num_items, self.latent_dim),
            torch.nn.Embedding(self.num_items, self.latent_dim)
        ])
        self.emb_entity_list = nn.ModuleList([
            nn.Embedding(self.num_entities + 1, self.latent_dim),
            nn.Embedding(self.num_entities + 1, self.latent_dim)
        ])
        self.emb_relation_list = nn.ModuleList([
            nn.Embedding(self.num_relations + 1, self.latent_dim),
            nn.Embedding(self.num_relations + 1, self.latent_dim)
        ])

        for i in range(2):
            nn.init.normal_(self.emb_item_list[i].weight, std=0.1)
            nn.init.normal_(self.emb_entity_list[i].weight, std=0.1)
            nn.init.normal_(self.emb_relation_list[i].weight, std=0.1)

        self.transR_W = nn.Parameter(torch.Tensor(self.num_relations + 1, self.latent_dim, self.latent_dim))
        self.TATEC_W = nn.Parameter(torch.Tensor(self.num_relations + 1, self.latent_dim, self.latent_dim))

        nn.init.xavier_uniform_(self.transR_W, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.TATEC_W, gain=nn.init.calculate_gain('relu'))

        self.W_R = nn.Parameter(
            torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.embedding_user.weight, std=0.1)

        self.co_user_score = nn.Linear(self.latent_dim, 1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.num_items)


    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


    def cal_item_embedding_from_kg(self, kg: dict = None, index=0):
        if kg is None:
            kg = self.kg_dict

        return self.cal_item_embedding_rgat(kg, index)


    def cal_item_embedding_rgat(self, kg: dict, index):
        item_embs = self.emb_item_list[index](
            torch.IntTensor(list(kg.keys())).to(
                configs['device']))
        item_entities = torch.stack(list(
            kg.values()))
        item_relations = torch.stack(list(self.item2relations.values()))
        entity_embs = self.emb_entity_list[index](
            item_entities)
        relation_embs = self.emb_relation_list[index](
            item_relations)
        padding_mask = torch.where(item_entities != self.num_entities,
                                   torch.ones_like(item_entities),
                                   torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs,
                                         padding_mask)


    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)

        pos_emb_ego0 = self.emb_item_list[0](pos_items)
        pos_emb_ego1 = self.emb_item_list[1](pos_items)
        neg_emb_ego0 = self.emb_item_list[0](neg_items)
        neg_emb_ego1 = self.emb_item_list[1](neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego0, pos_emb_ego1, neg_emb_ego0, neg_emb_ego1

    def getAll(self):
        all_users, all_items = self.computer()
        return all_users, all_items


    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, pos_emb_ego0,
         pos_emb_ego1, neg_emb_ego0, neg_emb_ego1) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + pos_emb_ego0.norm(2).pow(2) + pos_emb_ego1.norm(2).pow(2)
                              + neg_emb_ego0.norm(2).pow(2) + neg_emb_ego1.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        return loss, reg_loss

    def computer(self):
        users_emb = self.embedding_user.weight

        items_emb0 = self.cal_item_embedding_from_kg(index=0)
        items_emb1 = self.cal_item_embedding_from_kg(index=1)

        items_emb = (items_emb0 + items_emb1) / 2

        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if configs['model']['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def view_computer_all(self, g_droped, index):
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(index=index)
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def calc_kg_loss_transR(self, h, r, pos_t, neg_t, index):
        r_embed = self.emb_relation_list[index](r).unsqueeze(-1)
        h_embed = self.emb_item_list[index](h).unsqueeze(-1)
        pos_t_embed = self.emb_entity_list[index](pos_t).unsqueeze(-1)
        neg_t_embed = self.emb_entity_list[index](neg_t).unsqueeze(-1)

        r_matrix = self.transR_W[r]
        h_embed = torch.matmul(r_matrix, h_embed)
        pos_t_embed = torch.matmul(r_matrix, pos_t_embed)
        neg_t_embed = torch.matmul(r_matrix, neg_t_embed)

        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2),
                              dim=1)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2),
                              dim=1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = self._L2_loss_mean(h_embed) + self._L2_loss_mean(
            r_embed) + self._L2_loss_mean(pos_t_embed) + self._L2_loss_mean(neg_t_embed) + torch.norm(self.transR_W)

        loss = kg_loss + 1e-3 * l2_loss

        return loss

    def calc_kg_loss_TATEC(self, h, r, pos_t, neg_t, index):
        r_embed = self.emb_relation_list[index](r).unsqueeze(-1)
        h_embed = self.emb_item_list[index](h).unsqueeze(-1)
        pos_t_embed = self.emb_entity_list[index](pos_t).unsqueeze(-1)
        neg_t_embed = self.emb_entity_list[index](neg_t).unsqueeze(-1)

        r_matrix = self.TATEC_W[r]
        pos_mrt = torch.matmul(r_matrix, pos_t_embed)
        neg_mrt = torch.matmul(r_matrix, neg_t_embed)

        pos_hmrt = torch.sum(h_embed * pos_mrt, dim=1)
        neg_hmrt = torch.sum(h_embed * neg_mrt, dim=1)

        hr = torch.sum(h_embed * r_embed, dim=1)
        pos_tr = torch.sum(pos_t_embed * r_embed, dim=1)
        neg_tr = torch.sum(neg_t_embed * r_embed, dim=1)

        pos_ht = torch.sum(h_embed * pos_t_embed, dim=1)
        neg_ht = torch.sum(h_embed * neg_t_embed, dim=1)

        pos_score = pos_hmrt + hr + pos_tr + pos_ht
        neg_score = neg_hmrt + hr + neg_tr + neg_ht

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = self._L2_loss_mean(h_embed) + self._L2_loss_mean(
            r_embed) + self._L2_loss_mean(pos_t_embed) + self._L2_loss_mean(neg_t_embed) + torch.norm(self.TATEC_W)

        loss = kg_loss + 1e-3 * l2_loss

        return loss



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_heads = 1
        self.layers = nn.ModuleList([GraphAttentionLayer(nfeat,nhid,dropout=dropout,alpha=alpha,concat=True) for _ in range(self.num_heads)])
        self.out = nn.Linear(nhid * self.num_heads, nhid)

    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.out(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward_relation(self, item_embs, entity_embs, w_r, adj):
        x = F.dropout(entity_embs, self.dropout, training=self.training)
        x = torch.cat([att.forward_relation(item_embs, x, w_r, adj) for att in self.layers ], dim=1)

        x = self.out(x + item_embs)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.fc = nn.Linear(in_features * 3, 1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_relation(self, item_embs, entity_embs, relations, adj):
        Wh = item_embs.unsqueeze(1).expand(entity_embs.shape[0],entity_embs.shape[1], -1)
        We = entity_embs
        e_input = self.fc(torch.cat([Wh, relations, We], dim=-1)).squeeze()
        e = self.leakyrelu(e_input)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout,training=self.training)
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted
        return h_prime

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs,self.W)
        We = torch.matmul(entity_embs, self.W)
        a_input = self._prepare_cat(Wh, We)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout,training=self.training)
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1),entity_embs).squeeze()
        h_prime = entity_emb_weighted + item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size())
        return torch.cat((Wh, We), dim=-1)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class BPRLoss:
    def __init__(self, recmodel, opt):
        self.model = recmodel
        self.opt = opt
        self.weight_decay = configs["model"]["decay"]

    def compute(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        return loss

