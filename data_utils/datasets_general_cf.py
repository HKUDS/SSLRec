from scipy.sparse import coo_matrix, dok_matrix
import torch.utils.data as data
from config.configurator import configs
import numpy as np

class PairwiseTrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
	
	def sample_negs(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(configs['data']['item_num'])
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg
	
	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class PairwiseWEpochFlagTrnData(PairwiseTrnData):
	def __init__(self, coomat):
		super(PairwiseWEpochFlagTrnData, self).__init__(coomat)
		self.epoch_flag_counter = -1
		self.epoch_period = configs['model']['epoch_period']
	
	def __getitem__(self, idx):
		flag = 0
		if self.epoch_flag_counter == -1:
			flag = 1
			self.epoch_flag_counter = 0
		if idx == 0:
			self.epoch_flag_counter += 1
			if self.epoch_flag_counter % self.epoch_period == 0:
				flag = 1
		anc, pos, neg = super(PairwiseWEpochFlagTrnData, self).__getitem__(idx)
		return anc, pos, neg, flag

class AllRankTstData(data.Dataset):
	def __init__(self, coomat, trn_mat):
		self.csrmat = (trn_mat.tocsr() != 0) * 1.0

		user_pos_lists = [list() for i in range(coomat.shape[0])]
		# user_pos_lists = set()
		test_users = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			user_pos_lists[row].append(col)
			test_users.add(row)
		self.test_users = np.array(list(test_users))
		self.user_pos_lists = user_pos_lists
	
	def __len__(self):
		return len(self.test_users)
	
	def __getitem__(self, idx):
		pck_user = self.test_users[idx]
		pck_mask = self.csrmat[pck_user].toarray()
		pck_mask = np.reshape(pck_mask, [-1])
		return pck_user, pck_mask
