import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

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
        self.tau = configs['model']['tau']
        
        self.ui_user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.uu_user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.ui_item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        
        self.edge_dropper1 = SpAdjEdgeDrop()
        self.edge_dropper2 = SpAdjEdgeDrop()
        self.edge_dropper3 = SpAdjEdgeDrop()
        self.edge_dropper4 = SpAdjEdgeDrop()
        self.gcn = nn.ModuleList()
        for i in range(self.layer_num):
            self.gcn.append(GCNLayer(self.embedding_size, bias=False))
        self.ui_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.uu_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size, self.embedding_size)
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
        
    def forward(self, adj, uu_adj, keep_rate):
        if not self.is_training:
            return self.final_user_embeds, self.final_item_embeds

        adj1 = self.edge_dropper1(adj, keep_rate)
        adj2 = self.edge_dropper2(adj, keep_rate)
        uu_adj1 = self.edge_dropper3(uu_adj, keep_rate)
        uu_adj2 = self.edge_dropper4(uu_adj, keep_rate)

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
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: t.exp(x / self.tau)
        indices = t.arange(0, num_nodes).to(configs['device'])
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i+1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            losses.append(-t.log(between_sim[:, i*batch_size: (i+1) * batch_size].diag()
                                / (refl_sim.sum(1) + between_sim.sum(1)
                                    - refl_sim[:, i * batch_size: (i+1) *batch_size].diag())))
        return t.cat(losses)

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def gca_loss(self, z1, z2, mean=True, batch_size=configs['train']['batch_size']):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2, batch_size)
        l2 = self.semi_loss(h2, h1, batch_size)

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
        cross_loss *= self.lambda2
        i_loss = self.gca_loss(ui_user_embeds1, ui_user_embeds2) + self.gca_loss(ui_item_embeds1, ui_item_embeds2)
        s_loss = self.gca_loss(uu_user_embeds1, uu_user_embeds2)
        domain_loss = self.lambda1 * (i_loss + s_loss)
        reg_loss = self.reg_weight * reg_pick_embeds([anc_embeds, pos_embeds, neg_embeds])
        loss = bpr_loss + reg_loss + domain_loss + cross_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'domain_loss': domain_loss, 'cross_loss': cross_loss}
        return loss, losses
        
    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, self.uu_adj, 1.0)
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
        if bias:
            self.bias = nn.Parameter(init(t.empty(embedding_size)))
        else:
            self.bias = self.register_parameter('bias', None)
    
    def forward(self, adj ,x):
        support = t.mm(x, self.weight)
        output = t.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
