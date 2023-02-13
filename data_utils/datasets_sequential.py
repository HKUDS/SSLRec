from scipy.sparse import coo_matrix, dok_matrix
import torch.utils.data as data
from config.configurator import configs
import numpy as np

class SequentialDataset(data.Dataset):
	def __init__(self, user_seqs):
		self.uids = user_seqs["uid"]
		self.seqs = user_seqs["item_seq"]
		self.last_items = user_seqs["item_id"]
	
	def sample_negs(self):
		return None
	
	def __len__(self):
		return len(self.uids)

	def __getitem__(self, idx):
		return self.uids[idx], self.seqs[idx], self.last_items[idx]