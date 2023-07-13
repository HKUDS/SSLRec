import math
import torch as t
from torch import nn
from torch.nn import init
import dgl.function as fn
import torch.nn.functional as F
from config.configurator import configs

"""
Graph Related Augmentation
"""
class EdgeDrop(nn.Module):
    """ Drop edges in a graph.
    """
    def __init__(self, resize_val=False):
        super(EdgeDrop, self).__init__()
        self.resize_val = resize_val

    def forward(self, adj, keep_rate):
        """
        :param adj: torch_adj in data_handler
        :param keep_rate: ratio of preserved edges
        :return: adjacency matrix after dropping edges
        """
        if keep_rate == 1.0: return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (t.rand(edgeNum) + keep_rate).floor().type(t.bool)
        newVals = vals[mask] / (keep_rate if self.resize_val else 1.0)
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class NodeDrop(nn.Module):
    """ Drop nodes in a graph
        It is implemented by replace the embeddings of dropped nodes with random embeddings.
    """
    def __init__(self):
        super(NodeDrop, self).__init__()

    def forward(self, embeds, keep_rate):
        """
        :param embeds: the embedding matrix of nodes in the graph
        :param keep_rate: ratio of preserved nodes
        :return: the embeddings matrix after dropping nodes
        """
        if keep_rate == 1.0: return embeds
        data_config = configs['data']
        node_num = data_config['user_num'] + data_config['item_num']
        mask = (t.rand(node_num) + keep_rate).floor().view([-1, 1])
        return embeds * mask


"""
Feature-based Augmentation
"""
def perturb_embedding(embeds, eps):
    """
    :param embeds: embedding matrix
    :param eps: hyperparameters that control the degree of perturbation
    :return: perturbed embedding matrix
    """
    noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * eps
    embeds = embeds + noise
    return embeds