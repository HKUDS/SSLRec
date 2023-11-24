import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_social import PairwiseTrnData, AllRankTstData, SocialPairwiseTrnData, DSLTrnData
import torch as t
import torch.utils.data as data
import dgl
from dgl import DGLGraph
import networkx as nx
from math import sqrt
from tqdm import tqdm
import os

class DataHandlerSocial:
	def __init__(self):
		if configs['data']['name'] == 'ciao':
			predir = './datasets/social/ciao/'
		elif configs['data']['name'] == 'epinions':
			predir = './datasets/social/epinions/'
		elif configs['data']['name'] == 'yelp':
			predir = './datasets/social/yelp/'
		elif configs['data']['name'] == 'lastfm':
			predir = './datasets/social/lastfm/'

		self.trn_file = predir + 'trn_mat.pkl'
		self.tst_file = predir + 'tst_mat.pkl'
		self.trust_file = predir + 'trust_mat.pkl'
		self.category_file = predir + 'category.pkl'
		if configs['model']['name'] == 'smin':
			self.metapath_file = predir + 'metapath.pkl'
			self.subgraph_file = predir + '2hop_ui_subgraph.pkl'
		if configs['model']['name'] == 'kcgn':
			self.trn_time_file = predir + 'trn_time.pkl'
			self.multi_item_adj_file = predir + 'multi_item_adj.pkl'
			self.uu_vv_graph_file = predir + 'uu_vv_graph.pkl'
			self.uu_subgraph_file = predir + 'uu_mat_subgraph.pkl'
			self.ii_subgraph_file = predir + 'ii_mat_subgraph.pkl'

	def _load_one_mat(self, file):
		"""Load one single adjacent matrix from file

		Args:
			file (string): path of the file to load

		Returns:
			scipy.sparse.coo_matrix: the loaded adjacent matrix
		"""
		with open(file, 'rb') as fs:
			mat = (pickle.load(fs) != 0).astype(np.float32)
		if type(mat) != coo_matrix:
			mat = coo_matrix(mat)
		return mat

	def _load(self, path):
		with open(path, 'rb') as fs:
			data = pickle.load(fs)
		return data

	def _save(self, data, path):
		with open(path, 'wb') as fs:
			pickle.dump(data, fs)

	def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
		"""Convert a scipy sparse matrix to a torch sparse tensor."""
		if type(sparse_mx) != sp.coo_matrix:
			sparse_mx = sparse_mx.tocoo().astype(np.float32)
		indices = t.from_numpy(
			np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
		values = t.from_numpy(sparse_mx.data.astype(np.float32))
		shape = t.Size(sparse_mx.shape)
		return t.sparse.FloatTensor(indices, values, shape).to(configs['device'])

	def _build_subgraph(self, mat, subnode):
		node_num = mat.shape[0]
		graph = nx.Graph(mat)
		subgraph_list = list(nx.connected_components(graph))
		subgraph_cnt = len(subgraph_list)
		node_subgraph = [-1 for i in range(node_num)]
		adj_mat = sp.dok_matrix((subgraph_cnt, node_num), dtype=int)
		node_list = []
		for i in range(subgraph_cnt):
			subgraph_id = i
			subgraph = subgraph_list[i]
			if len(subgraph) > subnode:
				node_list += list(subgraph)
			for node_id in subgraph:
				assert node_subgraph[node_id] == -1
				node_subgraph[node_id] = subgraph_id
				adj_mat[subgraph_id, node_id] = 1
		node_subgraph = np.array(node_subgraph)
		assert np.sum(node_subgraph == -1) == 0
		adj_mat = adj_mat.tocsr()
		subgraph = (node_subgraph, adj_mat, node_list)
		return subgraph_list, subgraph

	def _build_motif_induced_adjacency_matrix(self, trust_mat, trn_mat):
		S = trust_mat
		Y = trn_mat
		B = S.multiply(S.T)
		U = S - B
		C1 = (U.dot(U)).multiply(U.T)
		A1 = C1 + C1.T
		C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
		A2 = C2 + C2.T
		C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
		A3 = C3 + C3.T
		A4 = (B.dot(B)).multiply(B)
		C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
		A5 = C5 + C5.T
		A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
		A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
		A8 = (Y.dot(Y.T)).multiply(B)
		A9 = (Y.dot(Y.T)).multiply(U)
		A9 = A9+A9.T
		A10  = Y.dot(Y.T)-A8-A9
		#addition and row-normalization
		H_s = sum([A1, A2, A3, A4, A5, A6, A7])
		H_s = H_s.multiply(1.0 / H_s.sum(axis=1).reshape(-1, 1))
		H_j = sum([A8, A9])
		H_j = H_j.multiply(1.0 / H_j.sum(axis=1).reshape(-1, 1))
		H_p = A10
		H_p = H_p.multiply(H_p > 1)
		H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))
		return [H_s, H_j, H_p]

	def _build_joint_adjacency(self, trn_mat):
		indices = t.from_numpy(
			np.vstack((trn_mat.row, trn_mat.col)).astype(np.int64))
		udegree = np.array(trn_mat.sum(axis=-1)).flatten()
		idegree = np.array(trn_mat.sum(axis=0)).flatten()
		values = t.Tensor([float(e) / sqrt(udegree[trn_mat.row[i]]) / sqrt(idegree[trn_mat.col[i]])
					for i, e in enumerate(trn_mat.data)])
		shape = t.Size(trn_mat.shape)
		norm_adj = t.sparse.FloatTensor(indices, values, shape)
		return norm_adj

	def _gen_metapath(self, trn_mat, trust_mat, category_mat, category_dict):
		"""
			adjust rate according to dataset.
		"""
		trn_mat = trn_mat.tocsr()
		user_num, item_num = trn_mat.shape
		
		uu_mat = (trust_mat.T + trust_mat) + sp.eye(user_num).tocsr()
		uu_mat = (uu_mat != 0)

		uiu_mat = sp.dok_matrix((user_num, user_num))
		for i in tqdm(range(user_num), desc='Creating uiu matrix'):
			data = trn_mat[i].toarray()
			_, iid = np.where(data != 0)
			uid_list, _ = np.where(np.sum(trn_mat[:, iid]!=0, axis=1) != 0)
			uid_list2 = np.random.choice(uid_list, size=int(uid_list.size*0.3), replace=False)
			uid_list2 = uid_list2.tolist()
			tmp = [i] * len(uid_list2)
			uiu_mat[tmp, uid_list2] = 1
		uiu_mat = uiu_mat.tocsr()
		uiu_mat = uiu_mat + uiu_mat.T + sp.eye(user_num).tocsr()
		uiu_mat = (uiu_mat != 0)

		uitiu_mat = sp.dok_matrix((user_num, user_num))
		for i in tqdm(range(user_num), desc='Creating uitiu matrix'):
			data = trn_mat[i].toarray()
			_, iid = np.where(data != 0)
			_, typeid_list = np.where(np.sum(category_mat[iid] != 0, axis=0) != 0)
			typeid_set = set(typeid_list.tolist())
			for typeid in typeid_set:
				iid_list = category_dict[typeid]
				uid_list, _ = np.where(np.sum(trn_mat[:, iid_list]!=0, axis=1) != 0)
				uid_list2 = np.random.choice(uid_list, size=int(uid_list.size*0.0003), replace=False)
				uid_list2 = uid_list2.tolist()
				tmp = [i]*len(uid_list2)
				uitiu_mat[tmp, uid_list2] = 1
		uitiu_mat = uitiu_mat.tocsr()
		uitiu_mat = uitiu_mat + uitiu_mat.T + sp.eye(user_num).tocsr()  
		uitiu_mat = (uitiu_mat != 0)

		iti_mat = sp.dok_matrix((item_num, item_num))
		for i in tqdm(range(item_num), desc='Creating iti matrix'):
			data = category_mat[i].toarray()
			_, typeid_list = np.where(data != 0)
			item_list, _ = np.where(np.sum(category_mat[:, typeid_list] != 0, axis=1) != 0)
			item_list2 = np.random.choice(item_list, size=int(item_list.size*0.002), replace=False)
			item_list2 = item_list2.tolist()
			tmp = [i]*len(item_list2)
			iti_mat[tmp, item_list2] = 1
		iti_mat = iti_mat.tocsr()
		iti_mat = iti_mat + iti_mat.T + sp.eye(item_num).tocsr() 
		iti_mat = (iti_mat != 0)

		iui_mat = sp.dok_matrix((item_num, item_num))
		trn_mat_T = trn_mat.T
		for i in tqdm(range(item_num), desc='Creating iui matrix'):
			data = trn_mat_T[i].toarray()
			_, uid = np.where(data != 0)
			iid_list, _ = np.where(np.sum(trn_mat_T[:, uid] != 0, axis=1) != 0)
			iid_list2 = np.random.choice(iid_list, size=int(iid_list.size*0.25), replace=False)
			iid_list2 = iid_list2.tolist()
			tmp = [i]*len(iid_list2)
			iui_mat[tmp, iid_list2] = 1
		iui_mat = iui_mat.tocsr()
		iui_mat = iui_mat + iui_mat.T + sp.eye(item_num).tocsr() 
		iui_mat = (iui_mat != 0)

		metapath = {}
		metapath['UU'] = uu_mat
		metapath['UIU'] = uiu_mat
		metapath['UITIU'] = uitiu_mat
		metapath['ITI'] = iti_mat
		metapath['IUI'] = iui_mat

		return metapath

	def _gen_subgraph(self, trn_mat, metapath, k_hop=2):
		uu_mat = metapath['UU']
		iti_mat = metapath['ITI']
		user_num, item_num = trn_mat.shape
		ui_num = user_num + item_num

		ui_subgraph = sp.dok_matrix((ui_num, ui_num))
		u_list, v_list = trn_mat.row, trn_mat.col
		ui_subgraph[u_list, user_num+v_list] = 1
		ui_subgraph[user_num+v_list, u_list] = 1
		uu_mat = uu_mat.tocoo()
		u_list, v_list = uu_mat.row, uu_mat.col
		ui_subgraph[u_list, v_list] = 1
		iti_mat = iti_mat.tocoo()
		u_list, v_list = iti_mat.row, iti_mat.col
		u_list = np.random.choice(u_list, size=int(u_list.size*0.02),replace=False)
		v_list = np.random.choice(v_list, size=int(v_list.size*0.02),replace=False)
		ui_subgraph[user_num+u_list, user_num+v_list] = 1
		ui_subgraph = ui_subgraph.tocsr()

		ui_mat = ui_subgraph.copy()
		if k_hop > 1:
			for i in tqdm(range(ui_num), desc='Creating ui subgraph'):
				data = ui_mat[i].toarray()
				_, id_list = np.where(data!=0)
				tmp = k_hop - 1
				while tmp > 0:
					_, id_list = np.where(np.sum(ui_mat[id_list,:],axis=0) > 10)
					ui_subgraph[i, id_list] = 1
					tmp = tmp -1
		ui_subgraph = (ui_subgraph !=0)
		data = (ui_mat, ui_subgraph)

		return data

	def _create_category_dict(self, category_mat):
		category_dict = {}
		category_data = category_mat.toarray()
		for i in range(category_data.shape[0]):
			iid = i
			item_type_list = np.where(category_data[i])[0]
			for typeid in item_type_list:
				if typeid in category_dict:
					category_dict[typeid].append(iid)
				else:
					category_dict[typeid] = [iid]
		return category_dict

	def _create_multiitem_user_adj(self, trn_mat, trn_time):
		rating_class = np.unique(trn_mat.data).size
		user_num, item_num = trn_mat.shape
		multi_adj = sp.lil_matrix((rating_class*item_num, user_num), dtype=int)
		trn_mat = trn_mat.tocoo()
		uid_list, iid_list, r_list = trn_mat.row, trn_mat.col, trn_mat.data

		for i in range(uid_list.size):
			uid = uid_list[i]
			iid = iid_list[i]
			r = r_list[i]
			multi_adj[iid*rating_class+r-1, uid] = trn_time[uid, iid]
			assert trn_time[uid, iid] != 0

		a = sp.csr_matrix((multi_adj.shape[1], multi_adj.shape[1]))
		b = sp.csr_matrix((multi_adj.shape[0], multi_adj.shape[0]))
		multi_adj2 = sp.vstack([sp.hstack([a, multi_adj.T]), sp.hstack([multi_adj,b])])
		return multi_adj2.tocsr()

	def _gen_uu_vv_graph(self, trn_mat, trust_mat, category_mat, category_dict):
		user_num, item_num = trn_mat.shape
		assert category_mat.shape[0] == trn_mat.shape[1]
		mat = (trust_mat.T + trust_mat) + sp.eye(user_num)
		uu_mat = (mat != 0) * 1

		iti_mat = sp.dok_matrix((item_num, item_num))
		category_mat = category_mat.toarray()
		for i in tqdm(range(category_mat.shape[0]), desc='Creating iti matrix'):
			item_type_list = np.where(category_mat[i])[0]
			for item_type in item_type_list:					
				item_list = category_dict[item_type]
				item_list = np.array(item_list)
				if item_list.size < 100:
					rate = 0.1
				elif item_list.size < 1000:
					rate = 0.01
				else:
					rate = 0.001
				item_list2 = np.random.choice(item_list, size=int(item_list.size*rate/2), replace=False)
				item_list2 = item_list2.tolist()
				tmp = [i for _ in range(len(item_list2))]
				iti_mat[tmp, item_list2] = 1
		
		iti_mat = iti_mat.tocsr()
		iti_mat = iti_mat + iti_mat.T + sp.eye(item_num)
		iti_mat = (iti_mat != 0)*1

		uu_vv_graph = {}
		uu_vv_graph['UU'] = uu_mat
		uu_vv_graph['II'] = iti_mat
		return uu_vv_graph

	def _normalize_adj(self, mat):
		"""Laplacian normalization for mat in coo_matrix

		Args:
			mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

		Returns:
			scipy.sparse.coo_matrix: normalized adjacent matrix
		"""
		degree = np.array(mat.sum(axis=-1))
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
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])

	def _make_torch_uu_adj(self, mat):
		mat = (mat != 0) * 1.0
		# mat = (mat+ sp.eye(mat.shape[0])) * 1.0
		mat = self._normalize_adj(mat)
		
		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])

	def load_data(self):
		trn_mat = self._load(self.trn_file)
		tst_mat = self._load(self.tst_file)
		trust_mat = self._load(self.trust_file)
		category_mat = self._load(self.category_file)
		self.trn_mat = trn_mat
		self.trust_mat = trust_mat
		category_dict = self._create_category_dict(category_mat)
		configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
		
		if configs['train']['loss'] == 'pairwise':
			trn_data = PairwiseTrnData(trn_mat)
		# elif configs['train']['loss'] == 'pointwise':
		# 	trn_data = PointwiseTrnData(trn_mat)
		tst_data = AllRankTstData(tst_mat, trn_mat)

		self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
		self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)

		if configs['model']['name'] == 'smin':
			if configs['data']['clear']:
				if os.path.exists(self.metapath_file):
					os.remove(self.metapath_file)
			if os.path.exists(self.metapath_file):
				metapath = self._load(self.metapath_file)
			else:
				metapath = self._gen_metapath(trn_mat, trust_mat, category_mat, category_dict)
				self._save(metapath, self.metapath_file)

			if configs['data']['clear']:
				if os.path.exists(self.subgraph_file):
					os.remove(self.subgraph_file)
			if os.path.exists(self.subgraph_file):
				subgraph = self._load(self.subgraph_file)
			else:
				subgraph = self._gen_subgraph(trn_mat, metapath, configs['model']['k_hop_num'])
				self._save(subgraph, self.subgraph_file)

			uu_graph = dgl.from_scipy(metapath['UU'], device=configs['device'])
			uiu_graph = dgl.from_scipy(metapath['UIU'], device=configs['device'])
			uitiu_graph = dgl.from_scipy(metapath['UITIU'], device=configs['device'])

			iti_graph = dgl.from_scipy(metapath['ITI'], device=configs['device'])
			iui_graph = dgl.from_scipy(metapath['IUI'], device=configs['device'])

			graph_dict={}
			graph_dict['uu'] = uu_graph
			graph_dict['uiu'] = uiu_graph
			graph_dict['uitiu'] = uitiu_graph
			graph_dict['iui'] = iui_graph
			graph_dict['iti'] = iti_graph

			print('user metapath: ' + configs['model']['user_graph_indx'])
			user_graph_list = configs['model']['user_graph_indx'].split('_')
			self.user_graph = []
			for i in range(len(user_graph_list)):
				self.user_graph.append(graph_dict[user_graph_list[i]])
			
			print('item metapath: ' + configs['model']['item_graph_indx'])
			item_graph_list = configs['model']['item_graph_indx'].split('_')
			self.item_graph = []
			for i in range(len(item_graph_list)):
				self.item_graph.append(graph_dict[item_graph_list[i]])

			del graph_dict, uu_graph, uiu_graph, uitiu_graph, iui_graph, iti_graph

			(self.ui_graph_adj, self.ui_subgraph_adj) = subgraph
			self.ui_subgraph_adj_tensor = self._sparse_mx_to_torch_sparse_tensor(self.ui_subgraph_adj)
			self.ui_subgraph_adj_norm =t.from_numpy(np.sum(self.ui_subgraph_adj,axis=1)).float().to(configs['device'])
			self.ui_graph = DGLGraph(self.ui_graph_adj).to(configs['device'])
		
		elif configs['model']['name'] == 'kcgn':
			trn_time = self._load(self.trn_time_file)
			if configs['data']['clear']:
				if os.path.exists(self.multi_item_adj_file):
					os.remove(self.multi_item_adj_file)
			if os.path.exists(self.multi_item_adj_file):
				multi_adj_time = self._load(self.multi_item_adj_file)
			else:
				multi_adj_time = self._create_multiitem_user_adj(trn_mat, trn_time)
				self._save(multi_adj_time, self.multi_item_adj_file)

			if configs['data']['clear']:
				if os.path.exists(self.uu_vv_graph_file):
					os.remove(self.uu_vv_graph_file)
			if os.path.exists(self.uu_vv_graph_file):
				uu_vv_graph = self._load(self.uu_vv_graph_file)
			else:
				uu_vv_graph = self._gen_uu_vv_graph(trn_mat, trust_mat, category_mat, category_dict)
				self._save(uu_vv_graph, self.uu_vv_graph_file)

			uu_mat = uu_vv_graph['UU'].astype(bool)
			ii_mat = uu_vv_graph['II'].astype(bool)
		
			uu_mat_edge_src, uu_mat_edge_dst = uu_mat.nonzero()
			self.uu_graph = dgl.graph(data=(uu_mat_edge_src, uu_mat_edge_dst),
							idtype=t.int32,
							num_nodes=uu_mat.shape[0],
							device=configs['device'])

			ii_mat_edge_src, ii_mat_edge_dst = ii_mat.nonzero()
			self.ii_graph = dgl.graph(data=(ii_mat_edge_src, ii_mat_edge_dst),
							idtype=t.int32,
							num_nodes=ii_mat.shape[0],
							device=configs['device'])
			
			if configs['data']['clear']:
				if os.path.exists(self.uu_subgraph_file):
					os.remove(self.uu_subgraph_file)
			if os.path.exists(self.uu_subgraph_file):
				uu_subgraph = self._load(self.uu_subgraph_file)
			else:
				_, uu_subgraph = self._build_subgraph(uu_mat, configs['model']['subnode'])
				self._save(uu_subgraph, self.uu_subgraph_file)

			if configs['data']['clear']:
				if os.path.exists(self.ii_subgraph_file):
					os.remove(self.ii_subgraph_file)
			if os.path.exists(self.ii_subgraph_file):
				ii_subgraph = self._load(self.ii_subgraph_file)
			else:
				_, ii_subgraph = self._build_subgraph(ii_mat, configs['model']['subnode'])
				self._save(ii_subgraph, self.ii_subgraph_file)

			self.uu_node_subgraph, self.uu_subgraph_adj, self.uu_dgi_node = uu_subgraph
			self.ii_node_subgraph, self.ii_subgraph_adj, self.ii_dgi_node = ii_subgraph

			self.uu_subgraph_adj_tensor = self._sparse_mx_to_torch_sparse_tensor(self.uu_subgraph_adj)
			self.uu_subgraph_adj_norm = t.from_numpy(np.sum(self.uu_subgraph_adj, axis=1)).float().to(configs['device'])

			self.ii_subgraph_adj_tensor = self._sparse_mx_to_torch_sparse_tensor(self.ii_subgraph_adj)
			self.ii_subgraph_adj_norm = t.from_numpy(np.sum(self.ii_subgraph_adj, axis=1)).float().to(configs['device'])

			self.uu_dgi_node_mask = np.zeros(configs['data']['user_num'])
			self.uu_dgi_node_mask[self.uu_dgi_node] = 1
			self.uu_dgi_node_mask = t.from_numpy(self.uu_dgi_node_mask).float().to(configs['device'])

			self.ii_dgi_node_mask = np.zeros(configs['data']['item_num'])
			self.ii_dgi_node_mask[self.ii_dgi_node] = 1
			self.ii_dgi_node_mask = t.from_numpy(self.ii_dgi_node_mask).float().to(configs['device'])

			print('time process')
			print('time step = %.1f hour' % (configs['model']['time_step']))
			time_step = 3600 * float(configs['model']['time_step'])
			row, col = multi_adj_time.nonzero()
			multi_data = multi_adj_time.data
			min_utc = multi_data.min()
			multi_data = ((multi_data - min_utc) / time_step).astype(int) + 2
			assert np.sum(row == col) == 0
			multi_adj_time_norm = sp.coo_matrix((multi_data, (row, col)), dtype=int, shape=multi_adj_time.shape).tocsr()
			self.max_time = multi_adj_time_norm.max() + 1
			print("max time = %d"%(self.max_time))
			num = multi_adj_time_norm.shape[0]
			multi_adj_time_norm = multi_adj_time_norm + sp.eye(num)
			print("uv graph link num = %d"%(multi_adj_time_norm.nnz))

			edge_src, edge_dst = multi_adj_time_norm.nonzero()
			time_seq = multi_adj_time_norm.tocoo().data
			self.time_seq_tensor = t.from_numpy(time_seq.astype(float)).long().to(configs['device'])

			self.rating_class = np.unique(trn_mat.data).size

			self.uv_g = dgl.graph(data=(edge_src, edge_dst),
									idtype=t.int32,
									num_nodes=multi_adj_time_norm.shape[0],
									device=configs['device'])

		elif configs['model']['name'] == 'mhcn':
			M_matrices = self._build_motif_induced_adjacency_matrix(trust_mat, trn_mat)
			H_s = M_matrices[0]
			self.H_s = self._sparse_mx_to_torch_sparse_tensor(H_s)
			H_j = M_matrices[1]
			self.H_j = self._sparse_mx_to_torch_sparse_tensor(H_j)
			H_p = M_matrices[2]
			self.H_p = self._sparse_mx_to_torch_sparse_tensor(H_p)
			self.R = self._build_joint_adjacency(trn_mat).to(configs['device'])

		elif configs['model']['name'] == 'dcrec':
			self.torch_adj = self._make_torch_adj(trn_mat)
			self.torch_uu_adj = self._make_torch_uu_adj(trust_mat)
		
		elif configs['model']['name'] == 'dsl':
			self.torch_adj = self._make_torch_adj(trn_mat)
			self.torch_uu_adj = self._make_torch_uu_adj(trust_mat)
			trust_mat = trust_mat.tocoo()
			social_trn_data = SocialPairwiseTrnData(trust_mat)
			dsl_trn_data = DSLTrnData(trn_data, social_trn_data)
			self.train_dataloader = data.DataLoader(dsl_trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)