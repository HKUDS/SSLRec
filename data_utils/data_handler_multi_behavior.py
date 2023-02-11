import torch
import torch.utils.data as data
import pickle
import numpy as np
import scipy.sparse as sp
from math import ceil
import datetime
import time
import random


from Params import args
# import graph_utils

import numpy as np
import scipy.sparse as sp
from scipy.sparse import *
import torch
from Params import args



class DataHandler(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):  
        super(DataHandler, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.behavior = beh
        self.behaviors_data = behaviors_data

        self.length = self.data.shape[0] 
        self.neg_data = [None]*self.length  
        self.pos_data = [None]*self.length  

        self.userNum = behaviors_data[0].shape[0]
        self.itemNum = behaviors_data[0].shape[1]
        # self.behavior = None
        self.behavior_mats = self.data2mat()


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


    def _matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
        values = torch.from_numpy(cur_matrix.data)  
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  


    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'  

        for i in range(self.length):
            self.neg_data[i] = [None]*len(self.behavior)
            self.pos_data[i] = [None]*len(self.behavior)

        for index in range(len(self.behavior)):
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

                if index == (len(self.behavior)-1):
                    self.pos_data[i][index] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[index][uid].data)==0:
                        self.pos_data[i][index] = -1
                    else:
                        t_array = self.behaviors_data[index][uid].toarray()
                        pos_index = np.where(t_array!=0)[1]
                        iid_pos = np.random.choice(pos_index, size = 1, replace=True, p=None)[0]   
                        self.pos_data[i][index] = iid_pos


