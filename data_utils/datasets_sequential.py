from scipy.sparse import coo_matrix, dok_matrix
import torch.utils.data as data
from config.configurator import configs
import numpy as np
import torch
import dgl
import pandas as pd
from collections import defaultdict, Counter
import scipy.sparse as sp
from os import path
from sklearn.metrics.pairwise import cosine_similarity


def build_sim_graph(user_history_lists, mode):
    k = configs['model']['sim_group_k']
    graph_file = path.join(configs['data']['dir'], f"sim_graph_{k}_{mode}.bin")
    try:
        g = dgl.load_graphs(graph_file, [0])
        print("loading isim graph from DGL binary file...")
        return g[0][0]
    except:
        print("building isim graph...")
        row = []
        col = []
        for uid, item_seq in user_history_lists.items():
            seq_len = len(item_seq)
            col.extend(item_seq)
            row.extend([uid]*seq_len)
        row = np.array(row)
        col = np.array(col)
        # n_users, n_items
        cf_graph = sp.csr_matrix(([1]*len(row), (row, col)), shape=(
            configs['data']['user_num']+1, configs['data']['item_num']+1), dtype=np.float32)
        similarity = cosine_similarity(cf_graph.transpose())
        # filter topk connections
        sim_items_slices = []
        sim_weights_slices = []
        i = 0
        while i < similarity.shape[0]:
            similarity = similarity[i:, :]
            sim = similarity[:256, :]
            sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
            sim_weights = np.take_along_axis(sim, sim_items, axis=1)
            sim_items_slices.append(sim_items)
            sim_weights_slices.append(sim_weights)
            i = i + 256
        sim = similarity[256:, :]
        sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
        sim_weights = np.take_along_axis(sim, sim_items, axis=1)
        sim_items_slices.append(sim_items)
        sim_weights_slices.append(sim_weights)

        sim_items = np.concatenate(sim_items_slices, axis=0)
        sim_weights = np.concatenate(sim_weights_slices, axis=0)
        row = []
        col = []
        for i in range(len(sim_items)):
            row.extend([i]*len(sim_items[i]))
            col.extend(sim_items[i])
        values = sim_weights / sim_weights.sum(axis=1, keepdims=True)
        values = np.nan_to_num(values).flatten()
        adj_mat = sp.csr_matrix((values, (row, col)), shape=(
            configs['data']['item_num'] + 1, configs['data']['item_num'] + 1))
        g = dgl.from_scipy(adj_mat, 'w')
        g.edata['w'] = g.edata['w'].float()
        print("saving isim graph to binary file...")
        dgl.save_graphs(graph_file, [g])
        return g


def build_adj_graph(user_history_lists, mode):
    graph_file = path.join(configs['data']['dir'], f"adj_graph_{mode}.bin")
    user_edges_file = path.join(configs['data']['dir'], "user_edges.pkl.zip")
    try:
        g = dgl.load_graphs(graph_file, [0])
        user_edges = pd.read_pickle(user_edges_file)
        print("loading graph from DGL binary file...")
        return g[0][0], user_edges
    except:
        print("constructing DGL graph...")
        item_adj_dict = defaultdict(list)
        item_edges_of_user = dict()
        for uid, item_seq in user_history_lists.items():
            item_edges_a, item_edges_b = [], []
            seq_len = len(item_seq)
            for i in range(seq_len):
                if i > 0:
                    item_adj_dict[item_seq[i]].append(item_seq[i-1])
                    item_adj_dict[item_seq[i-1]].append(item_seq[i])
                    item_edges_a.append(item_seq[i])
                    item_edges_b.append(item_seq[i-1])
                if i+1 < seq_len:
                    item_adj_dict[item_seq[i]].append(item_seq[i+1])
                    item_adj_dict[item_seq[i+1]].append(item_seq[i])
                    item_edges_a.append(item_seq[i])
                    item_edges_b.append(item_seq[i+1])
            item_edges_of_user[uid] = (np.asarray(
                item_edges_a), np.asarray(item_edges_b))
        if mode == 'train':
            item_edges_of_user = pd.DataFrame.from_dict(
                item_edges_of_user, orient='index', columns=['item_edges_a', 'item_edges_b'])
            item_edges_of_user.to_pickle(user_edges_file)
        cols = []
        rows = []
        values = []
        for item in item_adj_dict:
            adj = item_adj_dict[item]
            adj_count = Counter(adj)
            rows.extend([item]*len(adj_count))
            cols.extend(adj_count.keys())
            values.extend(adj_count.values())

        adj_mat = sp.csr_matrix((values, (rows, cols)), shape=(
            configs['data']['item_num'] + 1, configs['data']['item_num'] + 1))
        adj_mat = adj_mat.tolil()
        adj_mat.setdiag(np.ones((configs['data']['item_num'] + 1,)))
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        g = dgl.from_scipy(norm_adj, 'w')
        g.edata['w'] = g.edata['w'].float()
        print("saving DGL graph to binary file...")
        dgl.save_graphs(graph_file, [g])
        if mode == 'train':
            return g, item_edges_of_user
        else:
            return g, None


class SequentialDataset(data.Dataset):
    def __init__(self, user_seqs, mode='train', user_seqs_aug=None):
        self.mode = mode
        self.max_seq_len = configs['model']['max_seq_len']
        self.user_history_lists = {user: seq for user,
                                   seq in zip(user_seqs["uid"], user_seqs["item_seq"])}
        if user_seqs_aug is not None:
            self.uids = user_seqs_aug["uid"]
            self.seqs = user_seqs_aug["item_seq"]
            self.last_items = user_seqs_aug["item_id"]
        else:
            self.uids = user_seqs["uid"]
            self.seqs = user_seqs["item_seq"]
            self.last_items = user_seqs["item_id"]

        if mode == 'test':
            self.test_users = self.uids
            self.user_pos_lists = np.asarray(
                self.last_items, dtype=np.int32).reshape(-1, 1).tolist()
        
        if configs['model']['name'] == 'dcrec_seq':
            self.adj_graph, self.user_edges = build_adj_graph(self.user_history_lists, mode)
            self.sim_graph = build_sim_graph(self.user_history_lists, mode)

    def _pad_seq(self, seq):
        if len(seq) >= self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        else:
            # pad at the head
            seq = [0] * (self.max_seq_len - len(seq)) + seq
        return seq

    def sample_negs(self):
        if 'neg_samp' in configs['data'] and configs['data']['neg_samp']:
            self.negs = []
            for i in range(len(self.uids)):
                u = self.uids[i]
                seq = self.user_history_lists[u]
                last_item = self.last_items[i]
                while True:
                    iNeg = np.random.randint(1, configs['data']['item_num'])
                    if iNeg not in seq and iNeg != last_item:
                        break
                self.negs.append(iNeg)
        else:
            pass

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        seq_i = self.seqs[idx]
        if self.mode == 'train' and 'neg_samp' in configs['data'] and configs['data']['neg_samp']:
            return self.uids[idx], torch.LongTensor(self._pad_seq(seq_i)), self.last_items[idx], self.negs[idx]
        else:
            return self.uids[idx], torch.LongTensor(self._pad_seq(seq_i)), self.last_items[idx]
