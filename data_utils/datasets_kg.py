import torch.utils.data as data
from config.configurator import configs
import numpy as np
import random
import torch

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


def sample_pos_triples_for_h(kg_dict, head, n_sample_pos_triples):
    pos_triples = kg_dict[head]
    n_pos_triples = len(pos_triples)

    sample_relations, sample_pos_tails = [], []
    while True:
        if len(sample_relations) == n_sample_pos_triples:
            break

        pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
        tail = pos_triples[pos_triple_idx][1]
        relation = pos_triples[pos_triple_idx][0]

        if relation not in sample_relations and tail not in sample_pos_tails:
            sample_relations.append(relation)
            sample_pos_tails.append(tail)
    return sample_relations, sample_pos_tails

def sample_neg_triples_for_h(kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
    pos_triples = kg_dict[head]

    sample_neg_tails = []
    while True:
        if len(sample_neg_tails) == n_sample_neg_triples:
            break

        tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
        if (relation, tail) not in pos_triples and tail not in sample_neg_tails:
            sample_neg_tails.append(tail)
    return sample_neg_tails

def generate_kg_batch(kg_dict, batch_size, highest_neg_idx):
    exist_heads = kg_dict.keys()
    if batch_size <= len(exist_heads):
        batch_head = random.sample(exist_heads, batch_size)
    else:
        batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for h in batch_head:
        relation, pos_tail = sample_pos_triples_for_h(kg_dict, h, 1)
        batch_relation += relation
        batch_pos_tail += pos_tail

        neg_tail = sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
        batch_neg_tail += neg_tail

    batch_head = torch.LongTensor(batch_head)
    batch_relation = torch.LongTensor(batch_relation)
    batch_pos_tail = torch.LongTensor(batch_pos_tail)
    batch_neg_tail = torch.LongTensor(batch_neg_tail)
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

