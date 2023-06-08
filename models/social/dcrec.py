import torch as t
from torch import nn
import torch.nn.functional as F
import random
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop, MLP

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class DcRec(BaseModel):
    def __init__(self, data_handler):
        super(DcRec, self).__init__(data_handler)

        self.adj = data_handler.torch_adj
        self.uu_adj = data_handler.torch_uu_adj

        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']
        self.keep_rate = configs['model']['keep_rate']
        self.cross_weight = configs['model']['cross_weight']
        self.domain_weight = configs['model']['domain_weight']
        self.tau = configs['model']['tau']
        
        self.ui_user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.uu_user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.ui_item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        
        self.gcn = nn.ModuleList()
        for i in range(self.layer_num):
            self.gcn.append(GCNLayer(self.embedding_size))
            
        self.ui_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.uu_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.is_training = True
        
    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)
	
    def _lightgcn(self, adj, user_embeds, item_embeds):
        embeds = t.concat([user_embeds, item_embeds], axis=0)
        embeds_list = [embeds]
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list) / len(embeds_list)
        return embeds[:self.user_num], embeds[self.user_num:]

    def _gcn(self, adj, embeds):
        embeds_list = [embeds]
        for layer in self.gcn:
            embeds = F.relu(layer(adj, embeds_list[-1]))
            embeds_list.append(embeds)
        embeds = sum(embeds_list) / len(embeds_list)
        return embeds

    # TODO: efficiency
    def edge_adding(self, adj, p=0.5):
        """
        Perform edge adding.

        Args:
            p: the probability of adding an edge.
        """
        mask = t.rand(adj.shape) < p
        adj = adj + mask.to_sparse().to(configs['device'])
        return adj

    # TODO: efficiency
    def edge_dropout(self, adj, p=0.5):
        """
        Perform edge dropout.
        
        Args:
            adj: the input adjacency matrix (sparse or dense).
            p: the probability of dropping an edge.
        """
        mask = torch.bernoulli(torch.full(adj.shape, 1 - p, device=adj.device))
        adj = adj * mask.to_sparse()
        return adj
    
    # TODO: implement node dropout.
    def node_dropout(self, adj, p=0.5):
        """
        Perform node dropout.
        
        Args:
            p: the probability of dropping a node.
        """

        pass

    def graph_augment(self, adj):
        switch = random.sample(range(2), k=2)
        if switch[0] == 0:
            adj1 = self.edge_adding(adj)
        elif switch[0] == 1:
            adj1 = self.edge_dropout(adj)
        elif switch[0] == 2:
            adj1 = self.node_dropout(adj)

        if switch[1] == 0:
            adj2 = self.edge_adding(adj)
        elif switch[1] == 1:
            adj2 = self.edge_dropout(adj)
        elif switch[1] == 2:
            adj2 = self.node_dropout(adj)

        return adj1, adj2

    def edge_dropper(self, adj, p=0.5):
        keep_rate = 1 - p
        if keep_rate == 0.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edge_num = vals.size()
        mask = (t.rand(edge_num) + keep_rate).floor().type(t.bool)
        new_vals = vals[mask]
        new_idxs = idxs[:, mask]
        return t.sparse.FloatTensor(new_idxs, new_vals, adj.shape)

    def forward(self, adj, uu_adj, keep_rate):
        if not self.is_training:
            return self.final_user_embeds, self.final_item_embeds

        # adj1, adj2 = self.graph_augment(adj)
        # uu_adj1, uu_adj2 = self.graph_augment(uu_adj)
        adj1 = self.edge_dropper(adj)
        adj2 = self.edge_dropper(adj)
        uu_adj1 = self.edge_dropper(uu_adj)
        uu_adj2 = self.edge_dropper(uu_adj)

        ui_user_embeds, ui_item_embeds = self._lightgcn(adj, self.ui_user_embeds, self.ui_item_embeds)
        ui_user_embeds1, ui_item_embeds1 = self._lightgcn(adj1, self.ui_user_embeds, self.ui_item_embeds)
        ui_user_embeds2, ui_item_embeds2 = self._lightgcn(adj2, self.ui_user_embeds, self.ui_item_embeds)
        uu_user_embeds1 = self._gcn(uu_adj1, self.uu_user_embeds)
        uu_user_embeds2 = self._gcn(uu_adj2, self.uu_user_embeds)

        ui_user_embeds1 = F.relu(self.ui_linear(ui_user_embeds1))
        ui_user_embeds2 = F.relu(self.ui_linear(ui_user_embeds2))
        uu_user_embeds1 = F.relu(self.uu_linear(uu_user_embeds1))
        uu_user_embeds2 = F.relu(self.uu_linear(uu_user_embeds2))
        
        self.final_user_embeds = ui_user_embeds
        self.final_item_embeds = ui_item_embeds
        return ui_user_embeds, ui_item_embeds, ui_user_embeds1, ui_item_embeds1, ui_user_embeds2, ui_item_embeds2, uu_user_embeds1, uu_user_embeds2

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return t.mm(z1, z2.t())

    def semi_loss(self, z1, z2, batch_size):
        # f = lambda x: t.exp(x / self.tau)
        # refl_sim = f(self.sim(z1, z1))
        # between_sim = f(self.sim(z1, z2))
        # return -t.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: t.exp(x / self.tau)
        indices = t.arange(0, num_nodes).to(z1.device)
        losses = []

        # for i in range(num_batches):
        for i in range(3): # TODO: out of memory problem
            mask = indices[i * batch_size: (i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1)) # [b, n]
            between_sim = f(self.sim(z1[mask], z2)) # [b, n]
            loss = -t.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag()
                                / (refl_sim.sum(1) + between_sim.sum(1)
                                - refl_sim[:, i * batch_size: (i + 1) * batch_size].diag()))
            losses.append(loss)
        
        return t.cat(losses)

    def gca_loss(self, z1, z2, mean=True, batch_size=1024):
        l1 = self.semi_loss(z1, z2, batch_size)
        l2 = self.semi_loss(z2, z1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds, ui_user_embeds1, ui_item_embeds1, ui_user_embeds2, ui_item_embeds2, uu_user_embeds1, uu_user_embeds2 = self.forward(self.adj, self.uu_adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)

        cross_loss = self.gca_loss(uu_user_embeds1, ui_user_embeds1) + self.gca_loss(uu_user_embeds1, ui_user_embeds2)\
                        + self.gca_loss(uu_user_embeds2, ui_user_embeds1) + self.gca_loss(uu_user_embeds2, ui_user_embeds2)
        cross_loss *= self.cross_weight
        i_loss = self.gca_loss(ui_user_embeds1, ui_user_embeds2) + self.gca_loss(ui_item_embeds1, ui_item_embeds2)
        s_loss = self.gca_loss(uu_user_embeds1, uu_user_embeds2)
        domain_loss = self.domain_weight * (i_loss + s_loss)
        reg_loss = self.reg_weight * reg_pick_embeds([anc_embeds, pos_embeds, neg_embeds])
        loss = bpr_loss + reg_loss + domain_loss + cross_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'domain_loss': domain_loss, 'cross_loss': cross_loss}
        return loss, losses
        
    def full_predict(self, batch_data):
        ret = self.forward(self.adj, self.uu_adj, 1.0)
        user_embeds, item_embeds = ret[:2]
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

class GCNLayer(nn.Module):
    def __init__(self, embedding_size, bias=False):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(init(t.empty(embedding_size, embedding_size)))
    
    def forward(self, adj ,x):
        output = t.spmm(adj, x)
        return output