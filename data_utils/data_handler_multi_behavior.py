import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from math import ceil
import datetime

import scipy.sparse as sp
from scipy.sparse import *
import torch
import torch.utils.data as data
# import torch.utils.data as dataloader
from data_utils.datasets_general_cf import PairwiseTrnData, AllRankTstData
from config.configurator import configs
from data_utils.datasets_general_cf import PairwiseTrnData, AllRankTstData



class DataHandler:
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
        # self.trn_file = args.path + args.dataset + '/trn_'
        # self.tst_file = args.path + args.dataset + '/tst_int'     
        self.meta_multi_single_file = predir + 'meta_multi_single_beh_user_index_shuffle'


    def _load_data(self, file):
        self.meta_multi_single = pickle.load(open(self.meta_multi_single_file, 'rb'))

        self.t_max = -1 
        self.t_min = 0x7FFFFFFF
        self.time_number = -1
 
        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {} 
        self.behaviors = []
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

                if self.behaviors[i]==args.target:
                    self.trn_mat = data  
                    self.trainLabel = 1*(self.trn_mat != 0)  
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))  

        self.tst_mat = pickle.load(open(self.tst_file, 'rb'))

        self.userNum = self.behaviors_data[0].shape[0]
        self.itemNum = self.behaviors_data[0].shape[1]
        # self.behavior = None
        self.behavior_mats = self.data2mat()

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

    def load_data(self):
        #----raw version--------------------------------------------------------------------------------------------------------------------
        # self.trn_mat = self._load_one_mat(self.trn_file)
        # self.tst_mat = self._load_one_mat(self.tst_file)
        # self.trn_mat = trn_mat
        configs['data']['user_num'], configs['data']['item_num'] = self.trn_mat.shape
        self.torch_adj = self._make_torch_adj(self.trn_mat)  # TODO

        if configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(self.trn_mat)
        # elif configs['train']['loss'] == 'pointwise':
        # 	trn_data = PointwiseTrnData(trn_mat)
        tst_data = AllRankTstData(self.tst_mat, self.trn_mat)
        self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
        #----raw version--------------------------------------------------------------------------------------------------------------------

        #----my version--------------------------------------------------------------------------------------------------------------------
        #train_dataloader
        train_u, train_v = self.trn_mat.nonzero()
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = DataHandler.RecDataset_beh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

        #valid_dataloader

        # test_dataloader
        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)

        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        # tstUsrs = np.reshape(np.argwhere(data!=None), [-1])
        test_data = np.hstack((test_user.reshape(-1,1), test_item.reshape(-1,1))).tolist()
        # testbatch = np.maximum(1, args.batch * args.sampNum 
        test_dataset = DataHandler.RecDataset(test_data, self.item_num, self.trainMat, 0, False)  
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)  
        #----my version--------------------------------------------------------------------------------------------------------------------








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


