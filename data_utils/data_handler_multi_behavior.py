import dgl
import torch
import pickle
import datetime
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import torch.utils.data as dataloader
from config.configurator import configs
from data_utils.datasets_multi_behavior import *
from models.multi_behavior.kmclr import KGModel, Contrast, BPRLoss


class DataHandlerMultiBehavior:
    def __init__(self):
        super(DataHandlerMultiBehavior, self).__init__()

        if configs['data']['name'] == 'ijcai_15':
            self.predir = './datasets/multi_behavior/ijcai_15/'
            self.behaviors = ['click', 'fav', 'cart', 'buy']
            self.beh_meta_path = ['buy', 'click_buy', 'click_fav_buy', 'click_fav_cart_buy']
        elif configs['data']['name'] == 'tmall':
            self.predir = './datasets/multi_behavior/tmall/'
            self.behaviors = ['pv', 'fav', 'cart', 'buy']
            self.beh_meta_path = ['buy', 'pv_buy', 'pv_fav_buy', 'pv_fav_cart_buy']
        elif configs['data']['name'] == 'retail_rocket':
            self.predir = './datasets/multi_behavior/retail_rocket/'
            self.behaviors = ['view', 'cart', 'buy']
            self.beh_meta_path = ['buy', 'view_buy', 'view_cart_buy']

        self.train_file = self.predir + 'train_mat_'
        self.val_file = self.predir + 'test_mat.pkl'
        self.test_file = self.predir + 'test_mat.pkl'

        if configs['model']['name'] == 'cml':
            self.meta_multi_single_file = self.predir + 'meta_multi_single_beh_user_index_shuffle'

    def _load_data(self):
        self.t_max = -1
        self.t_min = 0x7FFFFFFF
        self.time_number = -1

        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {}
        self.behaviors_data = {}
        for i in range(0, len(self.behaviors)):
            with open(self.train_file + self.behaviors[i] + '.pkl', 'rb') as fs:
                data = pickle.load(fs)
                self.behaviors_data[i] = 1*(data != 0)
                if data.get_shape()[0] > self.user_num:
                    self.user_num = data.get_shape()[0]
                if data.get_shape()[1] > self.item_num:
                    self.item_num = data.get_shape()[1]
                if data.data.max() > self.t_max:
                    self.t_max = data.data.max()
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()
                if self.behaviors[i] == configs['model']['target']:
                    self.train_mat = data
                    self.trainLabel = 1 * (self.train_mat != 0)
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))
        self.test_mat = pickle.load(open(self.test_file, 'rb'))
        self.userNum = self.behaviors_data[0].shape[0]
        self.itemNum = self.behaviors_data[0].shape[1]
        self._data2mat()
        if configs['model']['name'] == 'cml':
            self.meta_multi_single = pickle.load(open(self.meta_multi_single_file, 'rb'))
        elif configs['model']['name'] == 'hmgcr':
            self.beh_meta_path_data = {}
            self.beh_meta_path_mats = {}
            for i in range(0, len(self.beh_meta_path)):
                self.beh_meta_path_data[i] = 1*(pickle.load(open(self.train_file + self.beh_meta_path[i] + '.pkl', 'rb')) != 0)
            time = datetime.datetime.now()
            print("Start building: ", time)
            for i in range(0, len(self.behaviors_data)):
                self.beh_meta_path_mats[i] = self._get_use(self.beh_meta_path_data[i])
            time = datetime.datetime.now()
            print("End building: ", time)
        elif configs['model']['name'] == 'smbrec':
            self.beh_degree_list = []
            for i in range(len(self.behaviors)):
                self.beh_degree_list.append(torch.tensor(((self.behaviors_data[i] != 0) * 1).sum(axis=-1)).cuda())

    def _data2mat(self):
        time = datetime.datetime.now()
        print("Start building: ", time)
        for i in range(0, len(self.behaviors_data)):
            self.behaviors_data[i] = 1*(self.behaviors_data[i] != 0)
            self.behavior_mats[i] = self._get_use(self.behaviors_data[i])
        time = datetime.datetime.now()
        print("End building: ", time)

    def _get_use(self, behaviors_data):
        behavior_mats = {}
        behaviors_data = (behaviors_data != 0) * 1
        behavior_mats['A'] = self._matrix_to_tensor(self._normalize_adj(behaviors_data))
        behavior_mats['AT'] = self._matrix_to_tensor(self._normalize_adj(behaviors_data.T))
        behavior_mats['A_ori'] = None
        return behavior_mats

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        rowsum_diag = sp.diags(np.power(rowsum+1e-8, -0.5).flatten())
        colsum = np.array(adj.sum(0))
        colsum_diag = sp.diags(np.power(colsum+1e-8, -0.5).flatten())
        return rowsum_diag*adj*colsum_diag

    def _matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)
        shape = torch.Size(cur_matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()

    def load_data(self):
        self._load_data()
        configs['data']['user_num'], configs['data']['item_num'] = self.train_mat.shape
        test_data = AllRankTestData(self.test_mat, self.train_mat)
        self.test_dataloader = dataloader.DataLoader(test_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)

        if configs['model']['name'] == 'cml':
            train_u, train_v = self.train_mat.nonzero()
            train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1, 1))).tolist()
            train_dataset = CMLData(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
            self.train_dataloader = dataloader.DataLoader(train_dataset, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        elif configs['model']['name'] == 'hmgcr' or configs['model']['name'] == 'smbrec':
            train_dataset = PairwiseTrnData(self.trainLabel.tocoo())
            self.train_dataloader = dataloader.DataLoader(train_dataset, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        elif configs['model']['name'] == 'kmclr':
            train_u, train_v = self.train_mat.nonzero()
            train_data = np.hstack((train_u.reshape(-1, 1), train_v.reshape(-1, 1))).tolist()
            train_dataset = KMCLRData(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
            self.train_dataloader = dataloader.DataLoader(train_dataset, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
            self.test_dataloader = dataloader.DataLoader(test_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
            # kg handler
            with open(self.train_file + 'buy.pkl', 'rb') as f:
                self.buy_mat = pickle.load(f)
            self.raw_kg_dataset = UIDataset(train_mat=self.buy_mat, path=self.predir)
            self.kg_dataset = KGDataset(self.raw_kg_dataset.m_item)
            self.Kg_model = KGModel(self.raw_kg_dataset, self.kg_dataset).to(configs['device']).to(configs['device'])
            self.contrast_model = Contrast(self.Kg_model, configs['model']['kgc_temp'])
            self.kg_optimizer = optim.Adam(self.Kg_model.parameters(), lr=configs['model']['kg_lr'])
            self.bpr = BPRLoss(self.Kg_model, self.kg_optimizer)
        else:
            if configs['train']['loss'] == 'pairwise':
                trn_data = PairwiseTrnData(self.train_mat.tocoo())
            self.train_dataloader = dataloader.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)


class DataHandlerMF(DataHandlerMultiBehavior):
    def __init__(self):
        super(DataHandlerMF, self).__init__()

    def load_data(self):
        self._load_data()
        configs['data']['user_num'], configs['data']['item_num'] = self.train_mat.shape
        if configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(self.train_mat.tocoo())
        test_data = AllRankTestData(self.test_mat, self.train_mat)
        self.test_dataloader = dataloader.DataLoader(test_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = dataloader.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
