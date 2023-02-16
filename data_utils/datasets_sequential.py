from scipy.sparse import coo_matrix, dok_matrix
import torch.utils.data as data
from config.configurator import configs
import numpy as np
import torch


class SequentialDataset(data.Dataset):
    def __init__(self, user_seqs, mode='train'):
        self.max_seq_len = configs['model']['max_seq_len']
        self.uids = user_seqs["uid"]
        self.padded_seqs = self._pad_seqs(user_seqs["item_seq"])
        self.last_items = user_seqs["item_id"]
        if mode == 'test':
            self.test_users = self.uids
            self.user_pos_lists = np.asarray(self.last_items, dtype=np.int32).reshape(-1, 1).tolist()

    def _pad_seqs(self, seqs):
        padded_seqs = []
        for seq in seqs:
            if len(seq) >= self.max_seq_len:
                seq = seq[:self.max_seq_len]
            else:
                # pad at the head
                seq = [0] * (self.max_seq_len - len(seq)) + seq
            padded_seqs.append(seq)
        return padded_seqs

    def sample_negs(self):
        return None

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        return self.uids[idx], torch.LongTensor(self.padded_seqs[idx]), torch.LongTensor([self.last_items[idx]])
