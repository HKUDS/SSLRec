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

class DataHandlerSocial:
	def __init__(self):
		if configs['data']['name'] == 'ciao':
			predir = './datasets/social_ciao/'
		elif configs['data']['name'] == 'epinions':
			predir = './datasets/social_epinions/'
		elif configs['data']['name'] == 'yelp':
			predir = './datasets/social_yelp/'
		self.trn_file = predir + 'trn_mat.pkl'
		self.tst_file = predir + 'tst_mat.pkl'
		self.metapath_file = predir + 'metapath.pkl'
		self.subgraph_file = predir + '2hop_ui_subgraph.pkl'

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

	def load_data(self):
		trn_mat = self._load_one_mat(self.trn_file)
		tst_mat = self._load_one_mat(self.tst_file)
		with open(self.metapath_file, 'rb') as fs:
			metapath = pickle.load(fs)
		with open(self.subgraph_file, 'rb') as fs:
			subgraph = pickle.load(fs)
        
		configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
		
		if configs['train']['loss'] == 'pairwise':
			trn_data = PairwiseTrnData(trn_mat)
		# elif configs['train']['loss'] == 'pointwise':
		# 	trn_data = PointwiseTrnData(trn_mat)
		tst_data = AllRankTstData(tst_mat, trn_mat)
		self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
		self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)

		uu_graph = dgl.from_scipy(metapath['UU'])
		uiu_graph = dgl.from_scipy(metapath['UIU'])
		uitiu_graph = dgl.from_scipy(metapath['UITIU'])

		iti_graph = dgl.from_scipy(metapath['ITI'])
		iui_graph = dgl.from_scipy(metapath['IUI'])

		graph_dict={}
		graph_dict['uu'] = uu_graph
		graph_dict['uiu'] = uiu_graph
		graph_dict['uitiu'] = uitiu_graph
		graph_dict['iui'] = iui_graph
		graph_dict['iti'] = iti_graph

		print('user metapath: ' + configs['model']['user_graph_indx'])
		user_graph_list = configs['model']['user_graph_indx'].split('_')
		user_graph = []
		for i in range(len(user_graph_list)):
			user_graph.append(graph_dict[user_graph_list[i]])
		
		print('item metapath: ' + configs['model']['item_graph_indx'])
		item_graph_list = configs['model']['item_graph_indx'].split('_')
		item_graph = []
		for i in range(len(item_graph_list)):
			item_graph.append(graph_dict[item_graph_list[i]])

		del graph_dict, uu_graph, uiu_graph, uitiu_graph, iui_graph, iti_graph

		if configs['model']['informax'] == 1:
			(ui_graph_adj, ui_subgraph_adj) = subgraph
			ui_subgraph_adj_Tensor = self._sparse_mx_to_torch_sparse_tensor(ui_subgraph_adj).cuda()
			ui_subgraph_adj_norm =t.from_numpy(np.sum(ui_subgraph_adj,axis=1)).float().cuda()
			ui_graph = DGLGraph(ui_graph_adj)
		
		

			