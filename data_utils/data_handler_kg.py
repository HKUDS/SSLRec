import torch
import torch.utils.data as data
import numpy as np
import scipy.sparse as sp
from config.configurator import configs
from os import path
from collections import defaultdict
from tqdm import tqdm
from .datasets_kg import KGTrainDataset, KGTestDataset, KGTripletDataset

class DataHandlerKG:
    def __init__(self) -> None:
        if configs['data']['name'] == 'mind':
            predir = './datasets/mind_kg/'
        elif configs['data']['name'] == 'amazon-book':
            predir = './datasets/amazon-book_kg/'
        elif configs['data']['name'] == 'last-fm':
            predir = './datasets/last-fm_kg/'
        configs['data']['dir'] = predir
        self.trn_file = path.join(predir, 'train.txt')
        self.val_file = path.join(predir, 'test.txt')
        self.tst_file = path.join(predir, 'test.txt') 
        self.kg_file = path.join(predir, 'kg_final.txt')   
        self.train_user_dict = defaultdict(list)
        self.test_user_dict = defaultdict(list)

    def _read_cf(self, file_name):
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(" ")]
            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])
        return np.array(inter_mat)
    
    def _collect_ui_dict(self, train_data, test_data):
        n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
        n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1
        configs['data']['user_num'] = n_users
        configs['data']['item_num'] = n_items

        for u_id, i_id in train_data:
            self.train_user_dict[int(u_id)].append(int(i_id))
        for u_id, i_id in test_data:
            self.test_user_dict[int(u_id)].append(int(i_id))

    def _read_triplets(self, file_name):
        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        can_triplets_np = np.unique(can_triplets_np, axis=0)

        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

        n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
        n_nodes = n_entities + configs['data']['user_num']
        n_relations = max(triplets[:, 1]) + 1

        configs['data']['entity_num'] = n_entities
        configs['data']['node_num'] = n_nodes
        configs['data']['relation_num'] = n_relations

        return triplets

    def _build_graphs(self, train_data, triplets):
        kg_dict = defaultdict(list)
        # h, t, r
        kg_edges = list()
        # u, i
        ui_edges = list()

        print("Begin to load interaction triples ...")
        for u_id, i_id in tqdm(train_data, ascii=True):
            ui_edges.append([u_id, i_id])

        print("Begin to load knowledge graph triples ...")
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            kg_edges.append([h_id, t_id, r_id])
            kg_dict[h_id].append((r_id, t_id))

        return kg_edges, ui_edges, kg_dict

    def _build_ui_mat(self, ui_edges):
        n_users = configs['data']['user_num']
        n_items = configs['data']['item_num']
        cf_edges = np.array(ui_edges)
        vals = [1.] * len(cf_edges)
        mat = sp.coo_matrix((vals, (cf_edges[:, 0], cf_edges[:, 1])), shape=(n_users, n_items))
        return mat
    
    def load_data(self):
        train_cf = self._read_cf(self.trn_file)
        test_cf = self._read_cf(self.tst_file)
        self._collect_ui_dict(train_cf, test_cf)
        kg_triplets = self._read_triplets(self.kg_file)
        self.kg_edges, ui_edges, self.kg_dict = self._build_graphs(train_cf, kg_triplets)
        self.ui_mat = self._build_ui_mat(ui_edges)

        test_data = KGTestDataset(self.test_user_dict)
        self.test_dataloader = data.DataLoader(test_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        train_data = KGTrainDataset(train_cf, self.train_user_dict)
        self.train_dataloader = data.DataLoader(train_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)

        if 'train_trans' in configs['model'] and configs['model']['train_trans']:
            triplet_data = KGTripletDataset(kg_triplets, self.kg_dict)
            # no shuffle because of randomness
            self.triplet_dataloader = data.DataLoader(triplet_data, batch_size=configs['train']['kg_batch_size'], shuffle=False, num_workers=0)
