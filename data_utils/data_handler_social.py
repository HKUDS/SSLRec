import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_social import PairwiseTrnData, AllRankTstData
import torch as t
import torch.utils.data as data
import dgl
from dgl import DGLGraph
import networkx as nx

class DataHandlerSocial:
	def __init__(self):
		if configs['data']['name'] == 'ciao':
			predir = './datasets/social_ciao/'
		elif configs['data']['name'] == 'epinions':
			predir = './datasets/social_epinions/'
		elif configs['data']['name'] == 'yelp':
			predir = './datasets/social_yelp/'
		elif configs['data']['name'] == 'lastfm':
			predir = './datasets/social_lastfm/'
		self.trn_file = predir + 'trn_mat.pkl'
		self.tst_file = predir + 'tst_mat.pkl'
		if configs['model']['name'] == 'smin':
			self.metapath_file = predir + 'metapath.pkl'
			self.subgraph_file = predir + '2hop_ui_subgraph.pkl'
		if configs['model']['name'] == 'kcgn':
			self.multi_item_adj_file = predir + 'multi_item_adj.pkl'
			self.uu_vv_graph_file = predir + 'uu_vv_graph.pkl'
			self.uu_subgraph_file = predir + 'uu_mat_subgraph.pkl'
			self.ii_subgraph_file = predir + 'ii_mat_subgraph.pkl'
		if configs['model']['name'] == 'mhcn':
			self.trust_file = predir + 'trust.pkl'

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

	def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
		"""Convert a scipy sparse matrix to a torch sparse tensor."""
		if type(sparse_mx) != sp.coo_matrix:
			sparse_mx = sparse_mx.tocoo().astype(np.float32)
		indices = t.from_numpy(
			np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
		values = t.from_numpy(sparse_mx.data)
		shape = t.Size(sparse_mx.shape)
		return t.sparse.FloatTensor(indices, values, shape)

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
		return subgraph_list, node_subgraph, adj_mat, node_list

	def _build_motif_induced_adjacency_matrix(self, trust_mat, trn_mat):
		S = trust_mat
		Y = trn_mat
		self.user_adjacency = Y.tocsr()
		self.item_adjacency = Y.T.tocsr()
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
		H_s = sum([A1,A2,A3,A4,A5,A6,A7])
		H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1))
		H_j = sum([A8,A9])
		H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
		H_p = A10
		H_p = H_p.multiply(H_p>1)
		H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))

		return [H_s,H_j,H_p]

	def load_data(self):
		with open(self.trn_file, 'rb') as fs:
			trn_mat = pickle.load(fs)
		tst_mat = self._load_one_mat(self.tst_file)
        
		configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
		
		if configs['train']['loss'] == 'pairwise':
			trn_data = PairwiseTrnData(trn_mat)
		# elif configs['train']['loss'] == 'pointwise':
		# 	trn_data = PointwiseTrnData(trn_mat)
		tst_data = AllRankTstData(tst_mat, trn_mat)
		self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
		self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)

		if configs['model']['name'] == 'smin':
			with open(self.metapath_file, 'rb') as fs:
				metapath = pickle.load(fs)
			with open(self.subgraph_file, 'rb') as fs:
				subgraph = pickle.load(fs)

			uu_graph = dgl.from_scipy(metapath['UU'], device=t.device('cuda'))
			uiu_graph = dgl.from_scipy(metapath['UIU'], device=t.device('cuda'))
			uitiu_graph = dgl.from_scipy(metapath['UITIU'], device=t.device('cuda'))

			iti_graph = dgl.from_scipy(metapath['ITI'], device=t.device('cuda'))
			iui_graph = dgl.from_scipy(metapath['IUI'], device=t.device('cuda'))

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
			self.ui_subgraph_adj_tensor = self._sparse_mx_to_torch_sparse_tensor(self.ui_subgraph_adj).cuda()
			self.ui_subgraph_adj_norm =t.from_numpy(np.sum(self.ui_subgraph_adj,axis=1)).float().cuda()
			self.ui_graph = DGLGraph(self.ui_graph_adj).to(t.device('cuda'))
		
		elif configs['model']['name'] == 'kcgn':
			with open(self.multi_item_adj_file, 'rb') as fs:
				multi_adj_time = pickle.load(fs)
			with open(self.uu_vv_graph_file, 'rb') as fs:
				uu_vv_graph = pickle.load(fs)
			uu_mat = uu_vv_graph['UU'].astype(bool)
			ii_mat = uu_vv_graph['II'].astype(bool)
		
			uu_mat_edge_src, uu_mat_edge_dst = uu_mat.nonzero()
			self.uu_graph = dgl.graph(data=(uu_mat_edge_src, uu_mat_edge_dst),
							idtype=t.int32,
							num_nodes=uu_mat.shape[0],
							device=t.device('cuda'))

			ii_mat_edge_src, ii_mat_edge_dst = ii_mat.nonzero()
			self.ii_graph = dgl.graph(data=(ii_mat_edge_src, ii_mat_edge_dst),
							idtype=t.int32,
							num_nodes=ii_mat.shape[0],
							device=t.device('cuda'))
			
			# _, self.uu_node_subgraph, self.uu_subgraph_adj, self.uu_dgi_node = self._build_subgraph(uu_mat, configs['model']['subnode'])
			# save_data = (self.uu_node_subgraph, self.uu_subgraph_adj, self.uu_dgi_node)
			# with open(self.uu_subgraph_file, 'wb') as fs:
			# 	pickle.dump(save_data, fs)

			# _, self.ii_node_subgraph, self.ii_subgraph_adj, self.ii_dgi_node = self._build_subgraph(ii_mat, configs['model']['subnode'])
			# save_data = (self.ii_node_subgraph, self.ii_subgraph_adj, self.ii_dgi_node)
			# with open(self.ii_subgraph_file, 'wb') as fs:
			# 	pickle.dump(save_data, fs)

			with open(self.uu_subgraph_file, 'rb') as fs:
				self.uu_node_subgraph, self.uu_subgraph_adj, self.uu_dgi_node = pickle.load(fs)
			with open(self.ii_subgraph_file, 'rb') as fs:
				self.ii_node_subgraph, self.ii_subgraph_adj, self.ii_dgi_node = pickle.load(fs)

			self.uu_subgraph_adj_tensor = self._sparse_mx_to_torch_sparse_tensor(self.uu_subgraph_adj).cuda()
			self.uu_subgraph_adj_norm = t.from_numpy(np.sum(self.uu_subgraph_adj, axis=1)).float().cuda()

			self.ii_subgraph_adj_tensor = self._sparse_mx_to_torch_sparse_tensor(self.ii_subgraph_adj).cuda()
			self.ii_subgraph_adj_norm = t.from_numpy(np.sum(self.ii_subgraph_adj, axis=1)).float().cuda()

			self.uu_dgi_node_mask = np.zeros(configs['data']['user_num'])
			self.uu_dgi_node_mask[self.uu_dgi_node] = 1
			self.uu_dgi_node_mask = t.from_numpy(self.uu_dgi_node_mask).float().cuda()

			self.ii_dgi_node_mask = np.zeros(configs['data']['item_num'])
			self.ii_dgi_node_mask[self.ii_dgi_node] = 1
			self.ii_dgi_node_mask = t.from_numpy(self.ii_dgi_node_mask).float().cuda()

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
			self.time_seq_tensor = t.from_numpy(time_seq.astype(float)).long().cuda()

			self.rating_class = np.unique(trn_mat.data).size

			self.uv_g = dgl.graph(data=(edge_src, edge_dst),
									idtype=t.int32,
									num_nodes=multi_adj_time_norm.shape[0],
									device=t.device('cuda'))

		elif configs['model']['name'] == 'mhcn':
			trust_mat = self._load_one_mat(self.trust_file)
			self.M_matrices = self._build_motif_induced_adjacency_matrix(trust_mat, trn_mat)
