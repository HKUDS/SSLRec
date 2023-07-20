import torch.utils.data as data
from config.configurator import configs
import numpy as np


class KGTrainDataset(data.Dataset):
    def __init__(self, train_cf_pairs, train_user_dict) -> None:
        self.train_cf_pairs = train_cf_pairs
        self.train_user_dict = train_user_dict

    def sample_negs(self):
        self.negs = np.zeros(len(self.train_cf_pairs), dtype=np.int32)
        for i in range(len(self.train_cf_pairs)):
            u = self.train_cf_pairs[i][0]
            while True:
                neg_i = np.random.randint(configs['data']['item_num'])
                if neg_i not in self.train_user_dict[u]:
                    break
            self.negs[i] = neg_i

    def __len__(self):
        return len(self.train_cf_pairs)

    def __getitem__(self, idx):
        # u, i, neg_i
        return self.train_cf_pairs[idx][0], self.train_cf_pairs[idx][1], self.negs[idx]


class KGTestDataset(data.Dataset):
    def __init__(self, test_user_dict, train_user_dict) -> None:
        self.user_pos_lists = test_user_dict
        self.test_users = np.array(list(test_user_dict.keys()))
        self.user_history_lists = train_user_dict

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        return self.test_users[idx]


class KGTripletDataset(data.Dataset):
    def __init__(self, kg_triplets, kg_dict) -> None:
        self.kg_triplets = kg_triplets
        self.kg_dict = kg_dict

    def __len__(self):
        return len(self.kg_triplets)

    def _neg_sample_kg(self, h, r):
        while True:
            neg_t = np.random.randint(configs['data']['entity_num'])
            if (r, neg_t) not in self.kg_dict[h]:
                break
        return neg_t

    def __getitem__(self, idx):
        random_idx = np.random.randint(len(self.kg_triplets))
        h, r, t = self.kg_triplets[random_idx]
        neg_t = self._neg_sample_kg(h, r)
        return h, r, t, neg_t
