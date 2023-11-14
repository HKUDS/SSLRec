import numpy as np
import collections
from scipy.sparse import coo_matrix, dok_matrix, csr_matrix
import pandas as pd
from tqdm import tqdm
import datetime
import random
import json
import dgl
# from dgl.data import DGLDataset
import pickle
from time import time
import scipy.sparse as sp

import torch
import torch.utils.data as data
from torch.utils.data import Dataset

from config.configurator import configs


class AllRankTestData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0

        user_pos_lists = [list() for i in range(coomat.shape[0])]
        # user_pos_lists = set()
        test_users = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            user_pos_lists[row].append(col)
            test_users.add(row)
        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.csrmat[pck_user].toarray()
        pck_mask = np.reshape(pck_mask, [-1])
        return pck_user, pck_mask


class PairwiseTrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def sample_negs(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(configs['data']['item_num'])
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class CMLData(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):

        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.beh = beh
        self.behaviors_data = behaviors_data

        self.length = self.data.shape[0]
        self.neg_data = [None] * self.length
        self.pos_data = [None] * self.length

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        for i in range(self.length):
            self.neg_data[i] = [None] * len(self.beh)
            self.pos_data[i] = [None] * len(self.beh)

        for index in range(len(self.beh)):

            train_u, train_v = self.behaviors_data[index].nonzero()
            beh_dok = self.behaviors_data[index].todok()

            set_pos = np.array(list(set(train_v)))

            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
            self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

            for i in range(self.length):  #

                uid = self.data[i][0]
                iid_neg = self.neg_data[i][index] = self.neg_data_index[i]
                iid_pos = self.pos_data[i][index] = self.pos_data_index[i]

                if (uid, iid_neg) in beh_dok:
                    while (uid, iid_neg) in beh_dok:
                        iid_neg = np.random.randint(low=0, high=self.num_item)
                        self.neg_data[i][index] = iid_neg
                    self.neg_data[i][index] = iid_neg

                if index == (len(self.beh) - 1):
                    self.pos_data[i][index] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[index].tocsr()[uid].data) == 0:
                        self.pos_data[i][index] = -1
                    else:
                        t_array = self.behaviors_data[index].tocsr()[uid].toarray()
                        pos_index = np.where(t_array != 0)[1]
                        iid_pos = np.random.choice(pos_index, size=1, replace=True, p=None)[0]
                        self.pos_data[i][index] = iid_pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.pos_data[idx]

        if self.is_training:
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i


class HMGCRData(data.Dataset):
    def __init__(self, data, num_item, train_mat=None, num_ng=0, is_training=None):
        super(HMGCRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
        self.data = np.array(data)
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def sample_negs(self):
        # assert self.is_training, 'no need to sampling when testing'
        tmp_trainMat = self.train_mat.todok()
        length = self.data.shape[0]
        self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)

        for i in range(length):
            uid = self.data[i][0]
            iid = self.neg_data[i]
            if (uid, iid) in tmp_trainMat:
                while (uid, iid) in tmp_trainMat:
                    iid = np.random.randint(low=0, high=self.num_item)
                self.neg_data[i] = iid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.data[idx][1]
        if self.is_training:
            neg_data = self.neg_data
            item_j = neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i


class KMCLRData(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):
        super(KMCLRData, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.beh = beh
        self.behaviors_data = behaviors_data

        self.length = self.data.shape[0]
        self.neg_data = [None] * self.length
        self.pos_data = [None] * self.length

    def ng_sample(self):
        for i in range(self.length):
            self.neg_data[i] = [None] * len(self.beh)
            self.pos_data[i] = [None] * len(self.beh)

        for index in range(len(self.beh)):
            train_u, train_v = self.behaviors_data[index].nonzero()
            beh_dok = self.behaviors_data[index].todok()

            set_pos = np.array(list(set(train_v)))

            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None)
            self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)

            for i in range(self.length):

                uid = self.data[i][0]
                iid_neg = self.neg_data[i][index] = self.neg_data_index[i]
                iid_pos = self.pos_data[i][index] = self.pos_data_index[i]

                if (uid, iid_neg) in beh_dok:
                    while (uid, iid_neg) in beh_dok:
                        iid_neg = np.random.randint(low=0, high=self.num_item)
                        self.neg_data[i][index] = iid_neg
                    self.neg_data[i][index] = iid_neg

                if index == (len(self.beh) - 1):
                    self.pos_data[i][index] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[index].tocsr()[uid].data) == 0:
                        self.pos_data[i][index] = -1
                    else:
                        t_array = self.behaviors_data[index].tocsr()[uid].toarray()
                        pos_index = np.where(t_array != 0)[1]
                        iid_pos = np.random.choice(pos_index, size=1, replace=True, p=None)[0]
                        self.pos_data[i][index] = iid_pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.pos_data[idx]

        if self.is_training:
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i


class KGDataset(Dataset):
    def __init__(self, m_item, kg_path='./datasets/multi_behavior/' + configs['data']['name'] + '/kg.txt'):
        self.m_item = m_item
        kg_data = pd.read_csv(kg_path, sep=' ', names=['h', 'r', 't'], engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data(kg_data=self.kg_data)

    @property
    def entity_count(self):
        return self.kg_data['t'].max() + 2

    @property
    def relation_count(self):
        return self.kg_data['r'].max() + 2

    def get_kg_dict(self, item_num):
        entity_num = configs['model']['entity_num_per_item']
        i2es = dict()
        i2rs = dict()
        for item in range(item_num):
            rts = self.kg_dict.get(item, False)
            if rts:
                tails = list(map(lambda x: x[1], rts))
                relations = list(map(lambda x: x[0], rts))
                if (len(tails) > entity_num):
                    i2es[item] = torch.IntTensor(tails).to(configs['device'])[:entity_num]
                    i2rs[item] = torch.IntTensor(relations).to(configs['device'])[:entity_num]
                else:
                    tails.extend([self.entity_count] * (entity_num - len(tails)))
                    relations.extend([self.relation_count] * (entity_num - len(relations)))
                    i2es[item] = torch.IntTensor(tails).to(configs['device'])
                    i2rs[item] = torch.IntTensor(relations).to(configs['device'])
            else:
                i2es[item] = torch.IntTensor([self.entity_count] * entity_num).to(configs['device'])
                i2rs[item] = torch.IntTensor([self.relation_count] * entity_num).to(configs['device'])
        return i2es, i2rs

    def generate_kg_data(self, kg_data):
        kg_dict = collections.defaultdict(list)
        for row in kg_data.iterrows():
            h, r, t = row[1]
            if h >= self.m_item:
                continue
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads

    def __len__(self):
        return len(self.kg_dict)

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        raise NotImplementedError

    def getSparseGraph(self):
        raise NotImplementedError


class UIDataset(BasicDataset):
    def __init__(self, train_mat, path):
        self.split = configs['model']['A_split']
        self.folds = configs['model']['A_n_fold']
        self.n_user = train_mat.shape[0]
        self.m_item = train_mat.shape[1]
        self.path = path
        
        trainUser = train_mat.tocsr().tocoo().row
        trainItem = train_mat.tocsr().tocoo().col
        self.traindataSize = len(trainUser)
        
        self.trainUniqueUsers = np.array(list(set(trainUser)))
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        self.Graph = None

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item), dtype=np.float32)
        self._allPos = self.getUserPosItems(list(range(self.n_user)))

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(configs['device']))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        if self.Graph is None:

            # adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            # adj_mat = adj_mat.tolil()
            # R = self.UserItemNet.tolil()
            # adj_mat[:self.n_users, self.n_users:] = R
            # adj_mat[self.n_users:, :self.n_users] = R.T
            # adj_mat = adj_mat.todok()
            
            rows = self.UserItemNet.tocoo().row
            cols = self.UserItemNet.tocoo().col
            new_rows = np.concatenate([rows, cols + self.n_users], axis=0)
            new_cols = np.concatenate([cols + self.n_users, rows], axis=0)
            adj_mat = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.n_users + self.m_items, self.n_users + self.m_items]).tocsr().tocoo().todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(configs['device'])
                
        return self.Graph

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __len__(self):
        return self.traindataSize

    def __getitem__(self, idx):
        user = self.trainUser[idx]
        pos = random.choice(self._allPos[user])
        while True:
            neg = np.random.randint(0, self.m_item)
            if neg in self._allPos[user]:
                continue
            else:
                break
        return user, pos, neg


# ----Sampler----


class MMCLRNeighborSampler(object):
    def __init__(self, g, num_layers, neg_sample_num=1, is_eval=False):
        self.g = g
        self.num_layers = num_layers
        self.is_eval = is_eval
        self.neg_sample_num = neg_sample_num
        self.rng = random.Random(configs['train']['random_seed'])
        self.error_count = 0
        self.total = 0

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        block_src_nodes = []
        for layer in range(self.num_layers):
            frontier = dgl.in_subgraph(self.g, seeds)
            frontier = dgl.compact_graphs(frontier, always_preserve=seeds)
            seeds = frontier.srcdata[dgl.NID]
            blocks.insert(0, frontier)
            src_nodes = {}
            for ntype in frontier.ntypes:
                src_nodes[ntype] = frontier.nodes(ntype=ntype)
            block_src_nodes.insert(0, src_nodes)
        input_nodes = seeds

        return input_nodes, blocks, block_src_nodes

    def sample_neg_user(self, batch_users):
        batch_users = batch_users.tolist()
        neg_batch_users = []
        user_set = list(set(batch_users))

        for user in batch_users:
            neg_user = user
            while neg_user == user:
                neg_user_idx = self.rng.randint(0, len(user_set) - 1)
                neg_user = user_set[neg_user_idx]

            neg_batch_users.append(neg_user_idx)
        return torch.tensor(neg_batch_users)

    def sample_from_item_pairs(self, seq_tensors):
        neg_src = []
        pos_src = []
        pos_dst = []
        neg_dst = []
        batch_tensors = [[] for _ in range(len(seq_tensors[0]))]
        for seq in seq_tensors:
            user_id, masked_item_seq, pv_item_seq, cart_item_seq, fav_item_seq, pos, neg, b1, b2, b3, con_len, sampled_click, pos_buy_item_seq = seq

            for i, data in enumerate(seq):
                batch_tensors[i].append(data)
            if self.is_eval:

                neg_dst.append(neg)
                pos_dst.append(pos)
            else:
                masked = pos - neg
                pos_dst.append(pos[masked != 0])
                neg_dst.append(neg[masked != 0])
            pos_src.append(user_id.repeat(pos_dst[-1].shape[0]))
            neg_src.append(user_id.repeat(neg_dst[-1].shape[0]))

        batch_tensors = [torch.stack(tensors, dim=0) for tensors in batch_tensors]
        batch_tensors[0] = batch_tensors[0].reshape(-1)
        neg_user_ids = torch.tensor([0])
        pos_dst = torch.cat(pos_dst, axis=0)
        neg_dst = torch.cat(neg_dst, axis=0)
        neg_src = torch.cat(neg_src, axis=0)
        pos_src = torch.cat(pos_src, axis=0)

        pos_graph = dgl.heterograph({('user', 'buy', 'item'): (pos_src, pos_dst),})
        neg_graph = dgl.heterograph({('user', 'buy', 'item'): (neg_src, neg_dst),})
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        buy_items = torch.cat((pos_dst, neg_dst), dim=0)
        seeds = {'user': batch_tensors[0], 'item': torch.cat((pos_dst, neg_dst), dim=0)}
        input_nodes, blocks, block_src_nodes = None, None, None
        return input_nodes, pos_graph, neg_graph, blocks, block_src_nodes, batch_tensors, neg_user_ids
