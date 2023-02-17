from scipy.sparse import coo_matrix, dok_matrix
import torch.utils.data as data
from config.configurator import configs
import numpy as np
import torch


class SequentialDataset(data.Dataset):
    def __init__(self, user_seqs, mode='train'):
        self.mode = mode
        self.max_seq_len = configs['model']['max_seq_len']
        self.uids = user_seqs["uid"]
        self.user_history_lists = {user: seq for user, seq in zip(self.uids, user_seqs["item_seq"])}
        self.last_items = user_seqs["item_id"]
        if mode == 'test':
            self.test_users = self.uids
            self.user_pos_lists = np.asarray(self.last_items, dtype=np.int32).reshape(-1, 1).tolist()

    def _pad_seq(self, seq):
        if len(seq) >= self.max_seq_len:
            seq = seq[:self.max_seq_len]
        else:
            # pad at the head
            seq = [0] * (self.max_seq_len - len(seq)) + seq
        return seq

    def sample_negs(self):
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

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        seq_i = self.user_history_lists[self.uids[idx]]
        if self.mode == 'train' and 'neg_samp' in configs['data'] and configs['data']['neg_samp']:
            return self.uids[idx], torch.LongTensor(self._pad_seq(seq_i)), self.last_items[idx], self.negs[idx]
        else:
            return self.uids[idx], torch.LongTensor(self._pad_seq(seq_i)), self.last_items[idx]
