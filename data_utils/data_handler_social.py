import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_social import PairwiseTrnData, AllRankTstData
import torch as t
import torch.utils.data as data

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
		