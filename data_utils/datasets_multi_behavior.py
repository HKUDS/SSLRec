from scipy.sparse import coo_matrix, dok_matrix
import torch.utils.data as data
from config.configurator import configs
import numpy as np


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
                    if len(self.behaviors_data[index][uid].data) == 0:
                        self.pos_data[i][index] = -1
                    else:
                        t_array = self.behaviors_data[index][uid].toarray()
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
