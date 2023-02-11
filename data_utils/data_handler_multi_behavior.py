import torch
import torch.utils.data as data
import pickle
import numpy as np
import scipy.sparse as sp
from math import ceil
import datetime

from Params import args
import graph_utils



class RecDataset(data.Dataset):
    def __init__(self, data, num_item, train_mat=None, num_ng=1, is_training=True):  
        super(RecDataset, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.train_mat = train_mat
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'  
        dok_trainMat = self.train_mat.todok()  
        length = self.data.shape[0]  
        self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)  

        for i in range(length):  #
            uid = self.data[i][0]
            iid = self.neg_data[i]
            if (uid, iid) in dok_trainMat:
                while (uid, iid) in dok_trainMat:  
                    iid = np.random.randint(low=0, high=self.num_item)
                    self.neg_data[i] = iid
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

    def getMatrix(self):
        pass
    
    def getAdj(self):
        pass
    
    def sampleLargeGraph(self):
   
    
        def makeMask():
            pass
    
        def updateBdgt():
            pass
    
        def sample():
            pass
    
    def constructData(self):
        pass




class RecDataset_beh(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):  
        super(RecDataset_beh, self).__init__()

        self.data = np.array(data)
        self.num_item = num_item
        self.is_training = is_training
        self.beh = beh
        self.behaviors_data = behaviors_data

        self.length = self.data.shape[0] 
        self.neg_data = [None]*self.length  
        self.pos_data = [None]*self.length  

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'  

        for i in range(self.length):
            self.neg_data[i] = [None]*len(self.beh)
            self.pos_data[i] = [None]*len(self.beh)

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

                if index == (len(self.beh)-1):
                    self.pos_data[i][index] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[index][uid].data)==0:
                        self.pos_data[i][index] = -1
                    else:
                        t_array = self.behaviors_data[index][uid].toarray()
                        pos_index = np.where(t_array!=0)[1]
                        iid_pos = np.random.choice(pos_index, size = 1, replace=True, p=None)[0]   
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

