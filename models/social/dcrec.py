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

        self.adj = data_handler.adj

        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']
        self.keep_rate = configs['model']['keep_rate']
        
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        
        self.edge_dropper = SpAdjEdgeDrop()
        self.edge_dropper1 = SpAdjEdgeDrop()
        self.edge_dropper2 = SpAdjEdgeDrop()
        self.edge_dropper3 = SpAdjEdgeDrop()
        self.edge_dropper4 = SpAdjEdgeDrop()
        self.is_training = True
        
    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)
	
    def _lightgcn(self, adj, user_embeds, item_embeds):
        embeds = t.concat([user_embeds, item_embeds], axis=0)
        embeds_list = [embeds]
        adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list) / len(embeds_list)
        return embeds[:self.user_num], embeds[self.user_num:]

    def _gcn(self, uu_adj, user_embeds):
        pass
        
    def forward(self, adj, uu_adj, keep_rate):
        if not self.is_training:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

        adj1 = self.edge_dropper1(adj, keep_rate)
        adj2 = self.edge_dropper2(adj, keep_rate)
        uu_adj1 = self.edge_dropper3(adj, keep_rate)
        uu_adj2 = self.edge_dropper4(adj, keep_rate)

        user_embeds1, item_embeds1 = self._lightgcn(adj1, self.user_embeds, self.item_embeds)
        user_embeds2, item_embeds2 = self._lightgcn(adj2, self.user_embeds, self.item_embeds)
        user_embeds1 = self._gcn(uu_adj1, self.user_embeds)
        user_embeds2 = self._gcn(uu_adj2, self.user_embeds)
        
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]
        
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.uu_adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
        reg_loss = self.reg_weight * reg_pick_embeds([anc_embeds, pos_embeds, neg_embeds])
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
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