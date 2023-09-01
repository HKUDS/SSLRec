import math
import random
from models.base_model import BaseModel
from models.model_utils import TransformerLayer, TransformerEmbedding
import numpy as np
import torch
from torch import nn
from config.configurator import configs
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from copy import deepcopy

def cal_kl_1(target, input):
    target[target<1e-8] = 1e-8
    target = torch.log(target + 1e-8)
    input = torch.log_softmax(input + 1e-8, dim=0)
    return F.kl_div(input, target, reduction='batchmean', log_target=True)

class CLLayer(torch.nn.Module):
    def __init__(self, num_hidden: int, tau: float = 0.5):
        super().__init__()
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        def f(x): return torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def vanilla_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)
        pos_pairs = f(self.sim(z1, z2)).diag()
        neg_pairs = f(self.sim(z1, z2)).sum(1)
        return -torch.log(1e-8 + pos_pairs / neg_pairs)

    def vanilla_loss_with_one_negative(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)
        pos_pairs = f(self.sim(z1, z2)).diag()
        neg_pairs = f(self.sim(z1, z2))
        rand_pairs = torch.randperm(neg_pairs.size(1))
        neg_pairs = neg_pairs[torch.arange(
            0, neg_pairs.size(0)), rand_pairs] + neg_pairs.diag()
        return -torch.log(pos_pairs / neg_pairs)

    def grace_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                   mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        return ret


def graph_dual_neighbor_readout(g: dgl.DGLGraph, aug_g: dgl.DGLGraph, node_ids, features):
    _, all_neighbors = g.out_edges(node_ids)
    all_nbr_num = g.out_degrees(node_ids)
    _, foreign_neighbors = aug_g.out_edges(node_ids)
    for_nbr_num = aug_g.out_degrees(node_ids)
    all_neighbors = [set(t.tolist())
                     for t in all_neighbors.split(all_nbr_num.tolist())]
    foreign_neighbors = [set(t.tolist())
                         for t in foreign_neighbors.split(for_nbr_num.tolist())]
    # sample foreign neighbors
    for i, nbrs in enumerate(foreign_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            foreign_neighbors[i] = set(nbrs)
    civil_neighbors = [all_neighbors[i]-foreign_neighbors[i]
                       for i in range(len(all_neighbors))]
    # sample civil neighbors
    for i, nbrs in enumerate(civil_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            civil_neighbors[i] = set(nbrs)
    for_lens = [len(t) for t in foreign_neighbors]
    cv_lens = torch.tensor([len(t)
                           for t in civil_neighbors], dtype=torch.int16)
    zero_indicies = (cv_lens == 0).nonzero().view(-1).tolist()
    cv_lens = cv_lens[cv_lens > 0].tolist()
    foreign_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.long) for s in foreign_neighbors])
    civil_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.long) for s in civil_neighbors])
    cv_feats = features[civil_neighbors].split(cv_lens)
    cv_feats = [t.mean(dim=0) for t in cv_feats]
    # insert zero vector for zero-length neighbors
    if len(zero_indicies) > 0:
        for i in zero_indicies:
            cv_feats.insert(i, torch.zeros_like(features[0]))
    for_feats = features[foreign_neighbors].split(for_lens)
    for_feats = [t.mean(dim=0) for t in for_feats]
    return torch.stack(cv_feats, dim=0), torch.stack(for_feats, dim=0)


def graph_augment(g: dgl.DGLGraph, user_ids, user_edges):
    # Augment the graph with the item sequence, deleting co-occurrence edges in the batched sequences
    # generating indicies like: [1,2] [2,3] ... as the co-occurrence rel.
    # indexing edge data using node indicies and delete them
    # for edge weights, delete them from the raw data using indexed edges
    user_ids = user_ids.cpu().numpy()
    node_indicies_a = np.concatenate(
        user_edges.loc[user_ids, "item_edges_a"].to_numpy())
    node_indicies_b = np.concatenate(
        user_edges.loc[user_ids, "item_edges_b"].to_numpy())
    node_indicies_a = torch.from_numpy(
        node_indicies_a).to(g.device)
    node_indicies_b = torch.from_numpy(
        node_indicies_b).to(g.device)
    edge_ids = g.edge_ids(node_indicies_a, node_indicies_b)

    aug_g: dgl.DGLGraph = deepcopy(g)
    # The features for the removed edges will be removed accordingly.
    aug_g.remove_edges(edge_ids)
    return aug_g


def graph_dropout(g: dgl.DGLGraph, keep_prob):
    # Firstly mask selected edge values, returns the true values along with the masked graph.
    origin_edge_w = g.edata['w']

    drop_size = int((1-keep_prob) * g.num_edges())
    random_index = torch.randint(
        0, g.num_edges(), (drop_size,), device=g.device)
    mask = torch.zeros(g.num_edges(), dtype=torch.uint8,
                       device=g.device).bool()
    mask[random_index] = True
    g.edata['w'].masked_fill_(mask, 0)

    return origin_edge_w, g


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0.7):
        super(GCN, self).__init__()
        self.dropout_prob = dropout_prob
        self.layer = GraphConv(in_dim, out_dim, weight=False,
                               bias=False, allow_zero_in_degree=False)

    def forward(self, graph, feature):
        graph = dgl.add_self_loop(graph)
        origin_w, graph = graph_dropout(graph, 1-self.dropout_prob)
        embs = [feature]
        for i in range(2):
            feature = self.layer(graph, feature, edge_weight=graph.edata['w'])
            F.dropout(feature, p=0.2, training=self.training)
            embs.append(feature)
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        # recover edge weight
        graph.edata['w'] = origin_w
        return final_emb


class DCRec_seq(BaseModel):

    def __init__(self, data_handler):
        super(DCRec_seq, self).__init__(data_handler)
        self.data_handler = data_handler
        self.device = configs['device']
        self.item_num = configs['data']['item_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['model']['max_seq_len']
        # load parameters info
        self.n_layers = configs['model']['n_layers']
        self.n_heads = configs['model']['n_heads']
        self.emb_size = configs['model']['embedding_size']
        # the dimensionality in feed-forward layer
        self.inner_size = 4 * self.emb_size
        self.dropout_rate = configs['model']['dropout_rate']
        self.batch_size = configs['train']['batch_size']

        self.weight_mean = configs['model']['weight_mean']
        self.kl_weight = configs['model']['kl_weight']
        self.cl_lambda = configs['model']['cl_lambda']
        # load dataset info
        # define layers and loss
        self.emb_layer = TransformerEmbedding(
            self.item_num + 1, self.emb_size, self.max_len)

        self.transformer_layers = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.inner_size, self.dropout_rate) for _ in range(self.n_layers)])

        self.loss_func = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(self.dropout_rate)

        self.layernorm = nn.LayerNorm(self.emb_size, eps=1e-12)

        # Contrastive Learning
        self.contrastive_learning_layer = CLLayer(
            self.emb_size, tau=configs['model']['cl_temp'])

        # Fusion Attn
        self.attn_weights = nn.Parameter(
            torch.Tensor(self.emb_size, self.emb_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.emb_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)

        # Global Graph Learning
        self.item_adjgraph = data_handler.train_dataloader.dataset.adj_graph.to(self.device)
        self.user_edges = data_handler.train_dataloader.dataset.user_edges
        self.item_simgraph = data_handler.train_dataloader.dataset.sim_graph.to(self.device)
        self.graph_dropout = configs["model"]["graph_dropout_prob"]

        self.gcn = GCN(self.emb_size, self.emb_size, self.graph_dropout)

        self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _subgraph_agreement(self, aug_g, adj_graph_emb, adj_graph_emb_last_items, last_items):
        # here it firstly removes items of the sequence in the cooccurrence graph, and then performs the gnn aggregation, and finally calculates the item-wise agreement score.
        aug_output_seq = self.gcn_forward(g=aug_g)[last_items]
        civil_nbr_ro, foreign_nbr_ro = graph_dual_neighbor_readout(
            self.item_adjgraph, aug_g, last_items, adj_graph_emb)

        view1_sim = F.cosine_similarity(
            adj_graph_emb_last_items, aug_output_seq, eps=1e-12)
        view2_sim = F.cosine_similarity(
            adj_graph_emb_last_items, foreign_nbr_ro, eps=1e-12)
        view3_sim = F.cosine_similarity(
            civil_nbr_ro, foreign_nbr_ro, eps=1e-12)
        agreement = (view1_sim+view2_sim+view3_sim)/3
        agreement = torch.sigmoid(agreement)
        agreement = (agreement - agreement.min()) / \
            (agreement.max() - agreement.min())
        agreement = (self.weight_mean / agreement.mean()) * agreement
        return agreement

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, task_label=False):
        """Generate bidirectional attention mask for multi-head attention."""
        if task_label:
            label_pos = torch.ones((item_seq.size(0), 1), device=self.device)
            item_seq = torch.cat((label_pos, item_seq), dim=1)
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def gcn_forward(self, g=None):
        item_emb = self.emb_layer.token_emb.weight
        item_emb = self.dropout(item_emb)
        light_out = self.gcn(g, item_emb)
        return self.layernorm(light_out+item_emb)

    def forward(self, batch_seqs):
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x = self.emb_layer(batch_seqs)
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        return x[:, -1, :]  # [B H]

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_pos_items = batch_data
        last_items = batch_seqs[:, -1].view(-1)
        # graph view
        masked_g = self.item_adjgraph
        aug_g = graph_augment(self.item_adjgraph, batch_user, self.user_edges)
        adj_graph_emb = self.gcn_forward(masked_g)
        sim_graph_emb = self.gcn_forward(self.item_simgraph)
        adj_graph_emb_last_items = adj_graph_emb[last_items]
        sim_graph_emb_last_items = sim_graph_emb[last_items]

        seq_output = self.forward(batch_seqs)
        aug_seq_output = self.forward(batch_seqs)
        # First-stage CL, providing CL weights
        # CL weights from augmentation
        mainstream_weights = self._subgraph_agreement(
            aug_g, adj_graph_emb, adj_graph_emb_last_items, last_items)
        # filtering those len=1, set weight=0.5
        seq_lens = batch_seqs.ne(0).sum(dim=1)
        mainstream_weights[seq_lens == 1] = 0.5

        expected_weights_distribution = torch.normal(self.weight_mean, 0.1, size=mainstream_weights.size()).to(self.device)
        # kl_loss = self.config["kl_weight"] * cal_kl(expected_weights_distribution.sort()[0], mainstream_weights.sort()[0])

        # apply log_softmax for input
        kl_loss = self.kl_weight * cal_kl_1(expected_weights_distribution.sort()[0], mainstream_weights.sort()[0])

        personlization_weights = mainstream_weights.max() - mainstream_weights

        # contrastive learning
        cl_loss_adj = self.contrastive_learning_layer.vanilla_loss(
            aug_seq_output, adj_graph_emb_last_items)
        cl_loss_a2s = self.contrastive_learning_layer.vanilla_loss(
            adj_graph_emb_last_items, sim_graph_emb_last_items)
        cl_loss = (self.cl_lambda * (mainstream_weights *
                                     cl_loss_adj + personlization_weights * cl_loss_a2s)).mean()
        # Fusion After CL
        # 3, N_mask, dim
        mixed_x = torch.stack(
            (seq_output, adj_graph_emb[last_items], sim_graph_emb[last_items]), dim=0)
        weights = (torch.matmul(
            mixed_x, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
        # 3, N_mask, 1
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_output = (mixed_x*score).sum(0)
        # [item_num, H]
        test_item_emb = self.emb_layer.token_emb.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits+1e-8, batch_pos_items)

        loss_dict = {
            "loss": loss.item(),
            "cl_loss": cl_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        return loss + cl_loss + kl_loss, loss_dict

    def full_predict(self, batch_data):
        batch_user, batch_seqs, _ = batch_data
        seq_output = self.forward(batch_seqs)
        last_items = batch_seqs[:, -1].view(-1)

        # graph view
        adj_graph = self.data_handler.test_dataloader.dataset.adj_graph.to(self.device)
        sim_graph = self.data_handler.test_dataloader.dataset.sim_graph.to(self.device)
        iadj_graph_output_raw = self.gcn_forward(adj_graph)
        iadj_graph_output_seq = iadj_graph_output_raw[last_items]
        isim_graph_output_seq = self.gcn_forward(sim_graph)[last_items]
        # 3, N_mask, dim
        mixed_x = torch.stack(
            (seq_output, iadj_graph_output_seq, isim_graph_output_seq), dim=0)
        weights = (torch.matmul(
            mixed_x, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
        # 3, N_mask, 1
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_output = (mixed_x*score).sum(0)

        test_item_emb = self.emb_layer.token_emb.weight  # [num, H]
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores