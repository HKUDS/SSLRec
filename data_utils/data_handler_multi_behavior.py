import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from math import ceil
import datetime

import scipy.sparse as sp
from scipy.sparse import *
import torch
import torch.utils.data as dataloader
from data_utils.datasets_multi_behavior import RecDatasetBeh, AllRankTstData
from config.configurator import configs


class DataHandlerMultiBehavior:
    def __init__(self):
        if configs['data']['name'] == 'ijcai_15':
            predir = './datasets/ijcai_15/'
            self.behaviors = ['click','fav', 'cart', 'buy']
            self.behaviors_SSL = ['click','fav', 'cart', 'buy']
        elif configs['data']['name'] == 'tmall':
            predir = './datasets/tmall/'
            self.behaviors_SSL = ['pv','fav', 'cart', 'buy']
            self.behaviors = ['pv','fav', 'cart', 'buy']
        elif configs['data']['name'] == 'retail_rocket':
            predir = './datasets/retail_rocket/'
            self.behaviors = ['view','cart', 'buy']
            self.behaviors_SSL = ['view','cart', 'buy']

        self.trn_file = predir + 'train_mat_'  # train_mat_buy.pkl 
        self.val_file = predir + 'test_mat.pkl'
        self.tst_file = predir + 'test_mat.pkl'
        self.meta_multi_single_file = predir + 'meta_multi_single_beh_user_index_shuffle'


    def _load_data(self):
        self.meta_multi_single = pickle.load(open(self.meta_multi_single_file, 'rb'))

        self.t_max = -1 
        self.t_min = 0x7FFFFFFF
        self.time_number = -1
 
        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {} 
        self.behaviors_data = {}
        for i in range(0, len(self.behaviors)):
            with open(self.trn_file + self.behaviors[i] + '.pkl', 'rb') as fs:  
                data = pickle.load(fs)
                self.behaviors_data[i] = data 

                if data.get_shape()[0] > self.user_num:  
                    self.user_num = data.get_shape()[0]  
                if data.get_shape()[1] > self.item_num:  
                    self.item_num = data.get_shape()[1]  

                if data.data.max() > self.t_max:
                    self.t_max = data.data.max()
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()

                if self.behaviors[i]==configs['model']['target']:  
                    self.trn_mat = data  
                    self.trainLabel = 1*(self.trn_mat != 0)  
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))  

        self.tst_mat = pickle.load(open(self.tst_file, 'rb'))

        self.userNum = self.behaviors_data[0].shape[0]
        self.itemNum = self.behaviors_data[0].shape[1]
        # self.behavior = None
        self.behavior_mats = self._data2mat()

    def _data2mat(self):
        time = datetime.datetime.now()
        print("Start building:  ", time)
        for i in range(0, len(self.behaviors_data)):
            self.behavior_mats[i] = self._get_use(self.behaviors_data[i])                  
        time = datetime.datetime.now()
        print("End building:", time)


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
        return adj
        return rowsum_diag*adj
        return adj*colsum_diag


    def _matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
        values = torch.from_numpy(cur_matrix.data)  
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  

    def load_data(self):  
        self._load_data()
        configs['data']['user_num'], configs['data']['item_num'] = self.trn_mat.shape
        tst_data = AllRankTstData(self.tst_mat, self.trn_mat)
        # self.torch_adj = self._make_torch_adj(self.trn_mat)  # TODO
        train_u, train_v = self.trn_mat.nonzero()
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = RecDatasetBeh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        self.train_dataloader = dataloader.DataLoader(train_dataset, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        self.test_dataloader = dataloader.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)



