import torch
import torch.utils.data as data
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from data_utils.datasets_general_cf import PairwiseTrnData, AllRankTstData
from data_utils.datasets_diff import DiffusionData
import numpy as np
import scipy.sparse as sp
from config.configurator import configs
from os import path
from collections import defaultdict
from tqdm import tqdm
from .datasets_kg import KGTrainDataset, KGTestDataset, KGTripletDataset, generate_kg_batch
import random


class DataHandlerKG:
	def __init__(self) -> None:
		if configs['data']['name'] == 'mind':
			predir = './datasets/kg/mind_kg/'
		elif configs['data']['name'] == 'alibaba-fashion':
			predir = './datasets/kg/alibaba-fashion_kg/'
		elif configs['data']['name'] == 'last-fm':
			predir = './datasets/kg/last-fm_kg/'

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
	
	def _read_cf_diff(self, file_name):
		inter_mat = list()
		lines = open(file_name, "r").readlines()
		for l in lines:
			tmps = l.strip()
			inters = [int(i) for i in tmps.split(" ")]
			u_id, pos_ids = inters[0], inters[1:]
			pos_ids = list(set(pos_ids))
			self.max_uid = max(self.max_uid, u_id)
			for i_id in pos_ids:
				inter_mat.append([u_id, i_id])
				self.max_iid = max(self.max_iid, i_id)
		return np.array(inter_mat)
	
	def _get_sp_mat(self, cf_data):
		ui_edges = list()
		for u_id, i_id in cf_data:
			ui_edges.append([u_id, i_id])
		ui_edges = np.array(ui_edges)
		vals = [1.] * len(ui_edges)
		mat = sp.coo_matrix((vals, (ui_edges[:, 0], ui_edges[:, 1])), shape=(self.max_uid+1, self.max_iid+1))
		return mat
	
	def _read_triplets_diff(self, file_name):
		can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
		can_triplets_np = np.unique(can_triplets_np, axis=0)
		
		inv_triplets_np = can_triplets_np.copy()
		inv_triplets_np[:, 0] = can_triplets_np[:, 2]
		inv_triplets_np[:, 2] = can_triplets_np[:, 0]
		inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
		triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

		n_relations = max(triplets[:, 1]) + 1
		
		configs['data']['relation_num'] = n_relations
		configs['data']['entity_num'] = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1		
		return triplets 
	
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
		configs['data']['triplet_num'] = len(triplets)

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
			# h,t,r
			kg_edges.append([h_id, t_id, r_id])
			kg_dict[h_id].append((r_id, t_id))

		return kg_edges, ui_edges, kg_dict

	def _build_graphs_diff(self, triplets):
		kg_dict = defaultdict(list)
		# h, t, r
		kg_edges = list()
		
		print("Begin to load knowledge graph triples ...")
		
		kg_counter_dict = {}
		
		for h_id, r_id, t_id in tqdm(triplets, ascii=True):
			if h_id not in kg_counter_dict.keys():
				kg_counter_dict[h_id] = set()
			if t_id not in kg_counter_dict[h_id]:
				kg_counter_dict[h_id].add(t_id)
			else:
				continue
			kg_edges.append([h_id, t_id, r_id])
			kg_dict[h_id].append((r_id, t_id))
			
		return kg_edges, kg_dict
	
	def buildKGMatrix(self, kg_edges):
		edge_list = []
		for h_id, t_id, r_id in kg_edges:
			edge_list.append((h_id, t_id))
		edge_list = np.array(edge_list)
		
		kgMatrix = sp.csr_matrix((np.ones_like(edge_list[:,0]), (edge_list[:,0], edge_list[:,1])), dtype='float64', shape=(configs['data']['entity_num'], configs['data']['entity_num']))
		
		return kgMatrix

	def _build_ui_mat(self, ui_edges):
		n_users = configs['data']['user_num']
		n_items = configs['data']['item_num']
		cf_edges = np.array(ui_edges)
		vals = [1.] * len(cf_edges)
		mat = sp.coo_matrix((vals, (cf_edges[:, 0], cf_edges[:, 1])), shape=(n_users, n_items))
		return mat
	
	def RelationDictBuild(self):
		relation_dict = {}
		for head in self.kg_dict:
			relation_dict[head] = {}
			for (relation, tail) in self.kg_dict[head]:
				relation_dict[head][tail] = relation
		return relation_dict
	
	def buildUIMatrix(self, mat):
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()
	
	def _normalize_adj(self, mat):
		"""Laplacian normalization for mat in coo_matrix

		Args:
			mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

		Returns:
			scipy.sparse.coo_matrix: normalized adjacent matrix
		"""
		# Add epsilon to avoid divide by zero
		degree = np.array(mat.sum(axis=-1)) + 1e-10
		d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
		d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
		d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
		return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

	def _make_torch_adj(self, mat):
		"""Transform uni-directional adjacent matrix in coo_matrix into bi-directional adjacent matrix in torch.sparse.FloatTensor

		Args:
			mat (coo_matrix): the uni-directional adjacent matrix

		Returns:
			torch.sparse.FloatTensor: the bi-directional matrix
		"""
		a = csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
		b = csr_matrix((configs['data']['item_num'], configs['data']['item_num']))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		# mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
		mat = self._normalize_adj(mat)

		# make torch tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])

	def load_data(self):
		if 'diff_model' in configs['model'] and configs['model']['diff_model'] == 1:
			self.max_uid, self.max_iid = 0, 0
			trn_cf = self._read_cf_diff(self.trn_file)
			tst_cf = self._read_cf_diff(self.tst_file)
			val_cf = self._read_cf_diff(self.val_file)
			trn_mat = self._get_sp_mat(trn_cf)
			tst_mat = self._get_sp_mat(tst_cf)
			val_mat = self._get_sp_mat(val_cf)
			self.trn_mat = trn_mat
			configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
			self.torch_adj = self._make_torch_adj(trn_mat)
			self.ui_matrix = self.buildUIMatrix(trn_mat)
			if configs['train']['loss'] == 'pairwise':
				trn_data = PairwiseTrnData(trn_mat)
			kg_triplets = self._read_triplets_diff(self.kg_file)
			self.kg_edges, self.kg_dict = self._build_graphs_diff(kg_triplets)
			self.kg_matrix = self.buildKGMatrix(self.kg_edges)
			self.diffusionData = DiffusionData(self.kg_matrix.A)
			self.diffusionLoader = data.DataLoader(self.diffusionData, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
			self.relation_dict = self.RelationDictBuild()
			val_data = AllRankTstData(val_mat, trn_mat)
			tst_data = AllRankTstData(tst_mat, trn_mat)
			self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
			self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
			self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
		else:
			train_cf = self._read_cf(self.trn_file)
			test_cf = self._read_cf(self.tst_file)
			self._collect_ui_dict(train_cf, test_cf)
			kg_triplets = self._read_triplets(self.kg_file)
			self.kg_edges, self.ui_edges, self.kg_dict = self._build_graphs(train_cf, kg_triplets)
			self.ui_mat = self._build_ui_mat(self.ui_edges)

			test_data = KGTestDataset(self.test_user_dict, self.train_user_dict)
			self.test_dataloader = data.DataLoader(test_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
			train_data = KGTrainDataset(train_cf, self.train_user_dict)
			self.train_dataloader = data.DataLoader(train_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)

			if 'train_trans' in configs['model'] and configs['model']['train_trans']:
				triplet_data = KGTripletDataset(kg_triplets, self.kg_dict)
				# no shuffle because of randomness
				self.triplet_dataloader = data.DataLoader(triplet_data, batch_size=configs['train']['kg_batch_size'], shuffle=False, num_workers=0)
	
	def generate_kg_batch(self):
		return generate_kg_batch(self.kg_dict, configs['train']['kg_batch_size'], configs['data']['entity_num'])
