import pickle
import numpy as np
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_sequential import SequentialDataset
import torch as t
import torch.utils.data as data
from os import path


class DataHandlerSequential:
    def __init__(self):
        if configs['data']['name'] == 'ml-20m':
            predir = './datasets/ml-20m_seq/'
        self.trn_file = path.join(predir, 'train.tsv')
        self.val_file = path.join(predir, 'test.tsv')
        self.tst_file = path.join(predir, 'test.tsv')
        self.max_seq_len = configs['data']['max_seq_len']

    def _read_tsv_to_padded_seqs(self, tsv_file):
        user_seqs = {"uid": [], "item_seq": [], "item_id": []}
        with open(tsv_file, 'r') as f:
            line = f.readline()
            while line:
                uid, seq, last_item = line.strip().split('\t')
                seq = seq.split(' ')
                seq = [int(i) for i in seq]
                if len(seq) >= self.max_seq_len:
                    seq = seq[:self.max_seq_len]
                else:
                    seq = seq + [0] * (self.max_seq_len - len(seq))
                user_seqs["uid"].append(int(uid))
                user_seqs["item_seq"].append(seq)
                user_seqs["item_id"].append(int(last_item))
                line = f.readline()
        return user_seqs
    
    def _set_statistics(self, user_seqs_train, user_seqs_test):
        user_num = max(max(user_seqs_train["uid"]), max(user_seqs_test["uid"])) + 1
        # item originally starts with 1
        item_num = max(max(user_seqs_train["item_id"]), max(user_seqs_test["item_id"]), max(user_seqs_train["item_seq"]), max(user_seqs_test["item_seq"]))
        configs['data']['user_num'] = user_num
        configs['data']['item_num'] = item_num
    
    def _seq_aug(self, user_seqs):
        user_seqs_new = {"uid": [], "item_seq": [], "item_id": []}
        for uid, seq, last_item in zip(user_seqs["uid"], user_seqs["item_seq"], user_seqs["item_id"]):
            for i in range(1, len(seq)):
                user_seqs_new["uid"].append(uid)
                user_seqs_new["item_seq"].append(seq[:i])
                user_seqs_new["item_id"].append(seq[i])
        return user_seqs_new

    def load_data(self):
        user_seqs_train = self._read_tsv_to_padded_seqs(self.trn_file)
        user_seqs_test = self._read_tsv_to_padded_seqs(self.tst_file)
        self._set_statistics(user_seqs_train, user_seqs_test)

        # seqeuntial augmentation: [1, 2, 3,] -> [1,2], [3]
        if configs['data']['seq_aug']:
            user_seqs_train = self._seq_aug(user_seqs_train)
            
        trn_data = SequentialDataset(user_seqs_train)
        tst_data = SequentialDataset(user_seqs_test)
        self.test_dataloader = data.DataLoader(
            tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(
            trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
