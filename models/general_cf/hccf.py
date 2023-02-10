import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
model_config = configs['model']

class HCCF(BaseModel):
	def __init__(self, data_handler):
		super(HCCF, self).__init__(data_handler)

		self.adj = data_handler.torch_adj

		self.layer_num = model_config['layer_num']
		self.reg_weight = model_config['reg_weight']
		self.cl_weight = model_config['cl_weight']
		self.hyper_num = model_config['hyper_num']

		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
		self.hgnn_layer = HGNNLayer()
		self.user_hyper_embeds = nn.Parameter(init(t.empty(self.embedding_size, self.hyper_num)))
		self.item_hyper_embeds = nn.Parameter(init(t.empty(self.embedding_size, self.hyper_num)))

class HGNNLayer(nn.Module):
	def __init__(self):
		super(HGNNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=model_config['leaky'])
	
	def forward(self, adj, embeds):
		hids = self.act(adj.T @ embeds)
		embeds = self.act(adj @ hids)
		return embeds