import torch as t
from torch import nn
from config.configurator import configs


class BaseModel(nn.Module):
	def __init__(self, data_handler):
		super(BaseModel, self).__init__()

		# put data_handler.xx you need into self.xx

		# put hyperparams you need into self.xx
		self.user_num = configs['data']['user_num']
		self.item_num = configs['data']['item_num']
		self.embedding_size = configs['model']['embedding_size']

		# initialize parameters
	
	# suggest to return embeddings
	def forward(self):
		pass

	def cal_loss(self, batch_data):
		"""return losses and weighted loss to training

		Args:
			batch_data (tuple): a batch of training samples already in cuda
		
		Return:
			loss (0-d torch.Tensor): the overall weighted loss
			losses (dict): dict for specific terms of losses for printing
		"""
		pass
	
	def _mask_predict(self, full_preds, train_mask):
		return full_preds * (1 - train_mask) - 1e8 * train_mask
	
	def full_predict(self, batch_data):
		"""return all-rank predictions to evaluation process, should call _mask_predict for masking the training pairs

		Args:
			batch_data (tuple): data in a test batch, e.g. batch_users, train_mask
		
		Return:
			full_preds (torch.Tensor): a [test_batch_size * item_num] prediction tensor
		"""
		pass