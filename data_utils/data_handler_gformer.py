import pickle
import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import networkx as nx
from config.configurator import configs

class DataHandlerGFormer:
    def __init__(self):
        self.device = configs['device']
        if configs['data']['name'] == 'yelp':
            predir = './datasets/sparse_yelp/'
        elif configs['data']['name'] == 'ifashion':
            predir = './datasets/ifashion/'
        elif configs['data']['name'] == 'lastfm':
            predir = './datasets/lastfm/'
        self.predir = predir
        self.trnfile = predir + 'train_mat.pkl'
        self.tstfile = predir + 'test_mat.pkl'

    def single_source_shortest_path_length_range(self, graph, node_range, cutoff):  # 最短路径算法
        dists_dict = {}
        for node in node_range:
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff=None)
        return dists_dict

    def get_random_anchorset(self):
        n = self.num_nodes
        annchorset_id = np.random.choice(n, size=configs['model']['anchor_set_num'], replace=False)
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(configs['data']['user_num'] + configs['data']['item_num']))

        rows = self.allOneAdj._indices()[0, :]
        cols = self.allOneAdj._indices()[1, :]

        rows = np.array(rows.cpu())
        cols = np.array(cols.cpu())

        edge_pair = list(zip(rows, cols))
        graph.add_edges_from(edge_pair)
        dists_array = np.zeros((len(annchorset_id), self.num_nodes))

        dicts_dict = self.single_source_shortest_path_length_range(graph, annchorset_id, None)
        for i, node_i in enumerate(annchorset_id):
            shortest_dist = dicts_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist != -1:
                    dists_array[i, j] = 1 / (dist + 1)
        self.dists_array = dists_array
        self.anchorset_id = annchorset_id #

    def preSelect_anchor_set(self):
        self.num_nodes = configs['data']['user_num'] + configs['data']['item_num']
        self.get_random_anchorset()

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, trainMat):

        a = sp.csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
        b = sp.csr_matrix((configs['data']['item_num'], configs['data']['item_num']))
        mat = sp.vstack([sp.hstack([a, trainMat]), sp.hstack([trainMat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def makeAllOne(self, torchAdj):
        idxs = torchAdj._indices()
        vals = t.ones_like(torchAdj._values())
        shape = torchAdj.shape
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def load_data(self):
        trnMat = self.loadOneFile(self.trnfile)
        tstMat = self.loadOneFile(self.tstfile)
        configs['data']['user_num'], configs['data']['item_num'] = trnMat.shape

        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.allOneAdj = self.makeAllOne(self.torchBiAdj)
        trnData = TrnData(trnMat)
        self.train_dataloader = dataloader.DataLoader(trnData, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.test_dataloader = dataloader.DataLoader(tstData, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.valid_dataloader = dataloader.DataLoader(tstData, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)


class TrnData(data.Dataset):
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


class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.test_users = tstUsrs
        self.user_pos_lists = tstLocs

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        return self.test_users[idx], np.reshape(self.csrmat[self.test_users[idx]].toarray(), [-1])
