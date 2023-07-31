import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_general_cf import PairwiseTrnData, AllRankTstData, PairwiseWEpochFlagTrnData
import torch as t
import torch.utils.data as data

class DataHandlerGeneralCF:
	def __init__(self):
		if configs['data']['name'] == 'yelp':
			predir = './datasets/general_cf/sparse_yelp/'
		elif configs['data']['name'] == 'gowalla':
			predir = './datasets/general_cf/sparse_gowalla/'
		elif configs['data']['name'] == 'amazon':
			predir = './datasets/general_cf/sparse_amazon/'
		self.trn_file = predir + 'train_mat.pkl'
		self.val_file = predir + 'valid_mat.pkl'
		self.tst_file = predir + 'test_mat.pkl'

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
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])
	
	def load_data(self):
		trn_mat = self._load_one_mat(self.trn_file)
		tst_mat = self._load_one_mat(self.tst_file)
		val_mat = self._load_one_mat(self.val_file)

		self.trn_mat = trn_mat
		configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
		self.torch_adj = self._make_torch_adj(trn_mat)

		if configs['train']['loss'] == 'pairwise':
			trn_data = PairwiseTrnData(trn_mat)
		elif configs['train']['loss'] == 'pairwise_with_epoch_flag':
			trn_data = PairwiseWEpochFlagTrnData(trn_mat)
		# elif configs['train']['loss'] == 'pointwise':
		# 	trn_data = PointwiseTrnData(trn_mat)

		val_data = AllRankTstData(val_mat, trn_mat)
		tst_data = AllRankTstData(tst_mat, trn_mat)
		self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
		self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
		self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)