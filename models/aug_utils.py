import math
import torch as t
from torch import nn
from torch.nn import init
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
    """ Drop nodes in a graph.
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

class AdaptiveMask(nn.Module):
    """ Adaptively masking edges with learned weight (used in DCCF)
    """
    def __init__(self, head_list, tail_list, matrix_shape):
        """
        :param head_list: list of id about head nodes
        :param tail_list: list of id about tail nodes
        :param matrix_shape: shape of the matrix
        """
        super(AdaptiveMask, self).__init__()
        self.head_list = head_list
        self.tail_list = tail_list
        self.matrix_shape = matrix_shape

    def forward(self, head_embeds, tail_embeds):
        """
        :param head_embeds: embeddings of head nodes
        :param tail_embeds: embeddings of tail nodes
        :return: indices and values (representing a augmented graph in torch_sparse fashion)
        """
        import torch_sparse
        head_embeddings = t.nn.functional.normalize(head_embeds)
        tail_embeddings = t.nn.functional.normalize(tail_embeds)
        edge_alpha = (t.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(row=self.head_list, col=self.tail_list, value=edge_alpha, sparse_sizes=self.matrix_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = t.stack([self.head_list, self.tail_list], dim=0)
        G_values = D_scores_inv[self.head_list] * edge_alpha
        return G_indices, G_values

class SvdDecomposition(nn.Module):
    """ Utilize SVD to decompose matrix (used in LightGCL)
    """
    def __init__(self, svd_q):
        super(SvdDecomposition, self).__init__()
        self.svd_q = svd_q

    def forward(self, adj):
        """
        :param adj: torch sparse matrix
        :return: matrices obtained by SVD decomposition
        """
        svd_u, s, svd_v = t.svd_lowrank(adj, q=self.svd_q)
        u_mul_s = svd_u @ t.diag(s)
        v_mul_s = svd_v @ t.diag(s)
        del s
        return svd_u.T, svd_v.T, u_mul_s, v_mul_s

"""
Feature-based Augmentation
"""
class EmbedDrop(nn.Module):
    """ Drop embeddings by nn.Dropout
    """
    def __init__(self, p=0.2):
        super(EdgeDrop, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, embeds):
        """
        :param embeds: embedding matrix
        :return: embedding matrix after dropping
        """
        embeds = self.dropout(embeds)
        return embeds

class EmbedPerturb(nn.Module):
    """ Perturb embeddings
    """
    def __init__(self, eps):
        super(EmbedPerturb, self).__init__()
        self.eps = eps

    def forward(self, embeds):
        """ Perturbing embeddings with noise
        :param embeds: embedding matrix
        :return: perturbed embedding matrix
        """
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        embeds = embeds + noise
        return embeds

class KMeansClustering(nn.Module):
    """ Use KMeans to calculate cluster centers of embeddings (used in NCL)
    """
    def __init__(self, cluster_num, embedding_size):
        super(KMeansClustering, self).__init__()
        self.cluster_num = cluster_num
        self.embedding_size = embedding_size

    def forward(self, embeds):
        """
        :param embeds: embedding matrix
        :return: cluster information obtained by KMeans
        """
        centroids = t.rand([self.cluster_num, self.embedding_size]).cuda()
        ones = t.ones([embeds.shape[0], 1]).cuda()
        for i in range(1000):
            dists = (embeds.view([-1, 1, self.embedding_size]) - centroids.view([1, -1, self.embedding_size])).square().sum(-1)
            _, idxs = t.min(dists, dim=1)
            newCents = t.zeros_like(centroids)
            newCents.index_add_(0, idxs, embeds)
            clustNums = t.zeros([centroids.shape[0], 1]).cuda()
            clustNums.index_add_(0, idxs, ones)
            centroids = newCents / (clustNums + 1e-6)
        return centroids, idxs, clustNums
