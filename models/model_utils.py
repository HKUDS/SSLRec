import torch as t
from torch import nn
from config.configurator import configs

class SpAdjEdgeDrop(nn.Module):
	def __init__(self):
		super(SpAdjEdgeDrop, self).__init__()

	def forward(self, adj, keep_rate):
		if keep_rate == 1.0:
			return adj
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = (t.rand(edgeNum) + keep_rate).floor().type(t.bool)
		newVals = vals[mask]# / keep_rate
		newIdxs = idxs[:, mask]
		return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class NodeDrop(nn.Module):
	def __init__(self):
		super(NodeDrop, self).__init__()

	def forward(self, embeds, keep_rate):
		if keep_rate == 1.0:
			return embeds
		data_config = configs['data']
		node_num = data_config['user_num'] + data_config['item_num']
		mask = (t.rand(node_num) + keep_rate).floor().view([-1, 1])
		return embeds * mask