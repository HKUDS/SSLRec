import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
from config.configurator import configs
from utils.time_logger import log# MARK
from utils.datasets import PairwiseTrnData, PointwiseTrnData, AllRankTstData
import torch as t
import torch.utils.data as data

class DataHandler:
	def __init__(self):
		if configs.data == 'yelp':
			predir = './Datasets/sparse_yelp/'
		elif configs.data == 'gowalla':
			predir = './Datasets/sparse_gowalla/'
		elif configs.data == 'amazon':
			predir = './Datasets/sparse_amazon/'
		self.trn_file = predir + 'trn_mat.pkl'
		self.val_file = predir + 'val_mat.pkl'
		self.tst_file = predir + 'tst_mat.pkl'

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
		degree = np.array(mat.sum(axis=-1))
		d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
		d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
		d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
		return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
	
	def _make_torch_adj(self, mat):
		"""Transform uni-directional adjacent matrix in coo_matrix into bi-directional adjacent matrix in torch.sparse.FloatTensor (with self-loop)

		Args:
			mat (coo_matrix): the uni-directional adjacent matrix

		Returns:
			torch.sparse.FloatTensor: the bi-directional matrix with self-loop
		"""
		a = csr_matrix((configs.user_num, configs.user_num))
		b = csr_matrix((configs.item, configs.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0# MARK
		mat = self._normalize_adj(mat)

		# make torch tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()
	
	def load_data(self):
		trn_mat = self._load_one_mat(self.trn_file)
		tst_mat = self._load_one_mat(self.tst_file)
		self.trn_mat = trn_mat
		configs.user_num, configs.item_num = trn_mat.shape
		self.torch_adj = self._make_torch_adj(trn_mat)

		if configs.loss == 'pairwise':
			trn_data = PairwiseTrnData(trn_mat)
		elif configs.loss == 'pointwise':
			trn_data = PointwiseTrnData(trn_mat)
		tst_data = AllRankTstData(tst_mat, trn_mat)
		self.tst_loader = data.DataLoader(tst_data, batch_size=configs.tst_batch_size, shuffle=False, num_workers=0)
		self.trn_loader = data.DataLoader(trn_data, batch_size=configs.batch_size, shuffle=True, num_workers=0)