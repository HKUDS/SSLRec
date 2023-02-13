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

class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = "both"
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None):
        graph = graph.local_var()

        if self._norm == 'both':
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1) # outdegree of nodes
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1) # (n, 1)
            norm = th.reshape(norm, shp) # (n, 1)
            # feat = feat * norm

        if weight is not None:
            if self.weight is not None:
                raise DGLError('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = th.matmul(feat, weight)
            feat = feat * norm
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_u(u='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
        else:
            # aggregate first then mult W
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_u(u='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            if weight is not None:
                rst = th.matmul(rst, weight)

        if self._norm != 'none':
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = th.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
        if self._activation is not None:
            rst = self._activation(rst)

        return rst

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 activation):
        super(GCN, self).__init__()
        self.g = g
        self.layer = GraphConv(in_feats, n_hidden, weight=False, activation=activation)

    def forward(self, features):
        h = features
        h = self.layer(self.g, h)
        return h
