import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
from models.multi_behavior.bert_modules.bert import BERT
from models.multi_behavior.bert_modules.utils import LayerNorm
from models.multi_behavior.bert_modules.embedding.token import TokenEmbedding
from models.multi_behavior.utils import layer   
from models.multi_behavior.utils import tools

from config.configurator import configs
from models.base_model import BaseModel


# from baseline_model import EHCFLyaer,MRIG,MBGCN,MGNN,MBGMN
import math
class MMCLR(BaseModel):
    def __init__(self, data_handler):
        super(MMCLR, self).__init__(data_handler)
        self.data_handler = data_handler
        self.userEmbeddingLayer =torch.nn.Embedding(configs['model']['user_size'], configs['model']['embedding_size'],padding_idx=0)
        self.itemEmbeddingLayer =torch.nn.Embedding(configs['model']['item_size'], configs['model']['embedding_size'],padding_idx=0)
        # self.itemEmbeddingLayer_SEQ=torch.nn.Embedding(configs['model']['item_size'], configs['model']['embedding_size'],padding_idx=0)
        self.seq_layer=layer.SequenceLayer(self.itemEmbeddingLayer)
        self.graph_layer=layer.GraphLayer(userEmbeddingLayer=self.userEmbeddingLayer, itemEmbeddingLayer=self.itemEmbeddingLayer)
        self.BPRLoss=PR_loss_for_bert()
        self.norm=torch.nn.LayerNorm(configs['model']['embedding_size'],elementwise_affine=False)
        self.graph_user_project=torch.nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size'])
        self.graph_item_project=torch.nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size'])
        self.seq_user_project=torch.nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size'])
        self.seq_item_project=torch.nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size'])
        self.layernorm=torch.nn.LayerNorm(configs['model']['embedding_size'],elementwise_affine=False)
        self.item_fusion=torch.nn.Sequential(
            torch.nn.Linear(configs['model']['embedding_size']*2,configs['model']['embedding_size']),
            torch.nn.ReLU(True),
             torch.nn.LayerNorm(configs['model']['embedding_size']),
            torch.nn.Linear(configs['model']['embedding_size'],configs['model']['embedding_size']),
        )
        self.user_fusion=torch.nn.Sequential(
            torch.nn.Linear(configs['model']['embedding_size']*2,configs['model']['embedding_size']),
            torch.nn.ReLU(True),
            torch.nn.LayerNorm(configs['model']['embedding_size']),
            torch.nn.Linear(configs['model']['embedding_size'],configs['model']['embedding_size'])
        )
        self.fc_cl_graph=torch.nn.Sequential(
            torch.nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size']*2),
             torch.nn.ReLU(True),
            torch.nn.Linear(configs['model']['embedding_size']*2, configs['model']['embedding_size'])
        )
        self.fc_cl_seq=torch.nn.Sequential(
            torch.nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size']*2),
             torch.nn.ReLU(True),
            torch.nn.Linear(configs['model']['embedding_size']*2, configs['model']['embedding_size'])
        )
        self.fc_graph=torch.nn.Sequential(
            torch.nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size']*2),
           
             torch.nn.ReLU(True),
            torch.nn.Linear(configs['model']['embedding_size']*2, configs['model']['embedding_size']),
        )
        self.fc_sequence=torch.nn.Sequential(
            torch.nn.Linear(configs['model']['embedding_size'], configs['model']['embedding_size']*2),
            torch.nn.ReLU(True),
            torch.nn.Linear(configs['model']['embedding_size']*2, configs['model']['embedding_size']),
        )
        self.metric=torch.nn.BCELoss()
        #self.f=open('/data/wuyq/MMCLR/res.graph','w')
       

    def inner_CL(self,graph_buy_emb,graph_click_emb,seq_buy_emb,seq_click_emb,user_ids,have_diff_behaviors):
        graph_buy_emb=self.fc_cl_graph(graph_buy_emb)
        graph_click_emb=self.fc_cl_graph(graph_click_emb)
        seq_click_emb=self.fc_cl_seq(seq_click_emb)
        seq_buy_emb=self.fc_cl_seq(seq_buy_emb)
        mask=(user_ids.unsqueeze(0)==user_ids.unsqueeze(1))
        gb2sb=(graph_buy_emb*seq_buy_emb).sum(-1)
        gb2sc=(graph_buy_emb*seq_click_emb).sum(-1)
        sb2gb=(graph_buy_emb*seq_buy_emb).sum(-1)
        sb2gc=(seq_buy_emb*graph_click_emb).sum(-1)
        scores1=gb2sb-gb2sc
        scores2=sb2gb-sb2gc
        have_diff_mask=have_diff_behaviors!=0
        scores1=scores1[have_diff_mask]
        scores1=scores1.sigmoid()
        scores2=scores2[have_diff_mask]
        scores2=scores2.sigmoid()
        scores=torch.cat((scores1,scores2),dim=0)
        
        loss=self.metric(scores,torch.ones_like(scores))
        return loss
    def constrat(self,seq_user_embs,graph_user_embs,user_ids):
        graph_user_embs=self.fc_graph(graph_user_embs)
        seq_user_embs=self.fc_sequence(seq_user_embs)
        scores=torch.matmul(graph_user_embs,seq_user_embs.T)
        scores=scores
        mask=(user_ids.unsqueeze(0)==user_ids.unsqueeze(1))
        mask=~mask
        self_scores=torch.diag(scores).reshape(-1,1)
        scores=(self_scores-scores).sigmoid()
        scores=scores[mask]
        losses=self.metric(scores,torch.ones_like(scores,dtype=torch.float32))
        return losses

    def forward(self,graph_data,seq_data,is_eval,is_aaa=False):
        user_id,masked_item_seq,pv_item_seq,cart_item_seq,fav_item_seq,pos,neg,b1,b2,sampled_view,have_cart,sampled_click,have_constra=seq_data
        input_nodes,pos_graph,neg_graph,blocks,block_src_nodes=graph_data
        if not is_eval:
            blocks=self.data_handler.g
        else:
            blocks=self.data_handler.test_g
        ce_loss=torch.tensor([0.0],device=configs['train']['device'])
        graph_cl_loss=torch.tensor([0.0],device=configs['train']['device'])
        cross_constra_loss=torch.tensor([0.0],device=configs['train']['device'])
        graph_inner_loss=torch.tensor([0.0],device=configs['train']['device'])
        inner_cl_loss=torch.tensor([0.0],device=configs['train']['device'])
        # graph_user_embedding,graph_item_embedding,graph_neg_item_embedding=self.graph_layer(input_nodes,pos_graph,neg_graph,blocks,block_src_nodes,is_eval)
        seqUID2graphUID={}
        graph_UIDs=[]
        for user_id in seq_data[0].tolist():
            if user_id not in seqUID2graphUID:
                seqUID2graphUID[user_id]=len(seqUID2graphUID)
            graph_UIDs.append(seqUID2graphUID[user_id])
        sampled_view=sampled_view.tolist() ## b3 store the behavior of b1 and b2 to constra
        have_cart=have_cart.squeeze(-1)
        have_constra=have_constra.squeeze(-1)
        if not is_eval:
            pos,neg=pos.flatten(),neg.flatten()
            mask_index=((pos-neg)!=0)
            if configs['model']['mode']!='graph':
                seq_user_embedding,seq_item_embedding,seq_neg_item_embedding,seq_inner_loss,ce_loss,seq_click_emb,seq_sampled_click_emb=self.seq_layer(seq_data,is_eval)
                seq_item_embedding=seq_item_embedding[mask_index]
                seq_user_embedding=seq_user_embedding[mask_index]
                seq_neg_item_embedding=seq_neg_item_embedding[mask_index]
            if configs['model']['mode']!='sequence':
                user_id=seq_data[0].repeat(configs['model']['max_seq_len'],1).t().flatten()
                user4seq=user_id[mask_index]
                pos4seq=pos[mask_index]
                neg4seq=neg[mask_index]
                graph_user_embedding,graph_item_embedding,graph_neg_item_embedding,graph_cl_loss,graph_click_emb,graph_inner_loss,graph_sampled_click_emb=self.graph_layer(blocks,block_src_nodes,constra_b=sampled_view,have_cart=have_cart,seq_tensor=seq_data)

                
            mutliView_user_embedding=torch.cat((seq_user_embedding,graph_user_embedding),dim=-1)  #[383, 64]  [383, 64]
            mutliView_item_embedding=torch.cat((seq_item_embedding,graph_item_embedding),dim=-1)  #[383, 64]  [383, 64]
            mutliView_neg_item_embedding=torch.cat((seq_neg_item_embedding,graph_neg_item_embedding),dim=-1)
            multiview_sampled_click_embedding=torch.cat((seq_sampled_click_emb,graph_sampled_click_emb),dim=-1)
            mutliView_user_embedding=self.user_fusion(mutliView_user_embedding)
            mutliView_item_embedding=self.item_fusion(mutliView_item_embedding)
            mutliView_neg_item_embedding=self.item_fusion(mutliView_neg_item_embedding)
            multiview_sampled_click_embedding=self.item_fusion(multiview_sampled_click_embedding)
            cross_constra_loss=self.constrat(seq_user_embs=seq_user_embedding,graph_user_embs=graph_user_embedding, user_ids=user4seq)
            inner_cl_loss=self.inner_CL(graph_buy_emb=graph_click_emb[0],graph_click_emb=graph_click_emb[1],seq_buy_emb=seq_click_emb[0], seq_click_emb=seq_click_emb[1], user_ids=seq_data[0], have_diff_behaviors=have_constra)
 
            pos_score=torch.mul(mutliView_user_embedding,mutliView_item_embedding).sum(-1)
            neg_score=torch.mul(mutliView_user_embedding,mutliView_neg_item_embedding).sum(-1)

            score=((pos_score-neg_score)).sigmoid()
            link_loss=self.metric(score,torch.ones_like(score))
            mask=sampled_click[sampled_click!=-1]
            mask=mask>0
            masked_mul_user_emb=mutliView_user_embedding[mask]
            multiview_sampled_click_embedding=multiview_sampled_click_embedding[mask]
            masked_pos_score=pos_score[mask]
            buy_click_loss=torch.tensor([0.0],device=configs['train']['device'])
            click_score=torch.mul(masked_mul_user_emb,multiview_sampled_click_embedding).sum(-1)
            click_unclick_score=(click_score-neg_score[mask]).sigmoid()
            buy_click_score=(masked_pos_score-click_score).sigmoid()
            if configs['model']['clamp']!=0:
                buy_click_score=buy_click_score.clamp(max=0.7)
            buy_click_loss=self.metric(buy_click_score,torch.ones_like(buy_click_score))
            buy_click_loss=self.metric(click_unclick_score,torch.ones_like(click_unclick_score))+buy_click_loss
            loss=configs['model']['main_weight']*link_loss+ce_loss*configs['model']['seq_cons_weight']+graph_cl_loss*configs['model']['graph_cons_weight']+cross_constra_loss*configs['model']['cross_cons_weight']+inner_cl_loss*configs['model']['inner_loss_weight']+buy_click_loss*configs['model']['buy_click_weight']
            return loss,link_loss,ce_loss,graph_cl_loss,cross_constra_loss,graph_inner_loss,buy_click_loss
        else:
            if configs['model']['mode']!='graph':
                seq_user_embedding,seq_item_embedding,seq_neg_item_embedding,seq_inner_loss,ce_loss,seq_click_emb,seq_sampled_click_emb=self.seq_layer(seq_data,is_eval)
                # seq_user_embedding,seq_item_embedding,seq_neg_item_embedding=self.seq_user_project(seq_user_embedding),self.seq_item_project(seq_item_embedding),self.seq_item_project(seq_neg_item_embedding)
            if configs['model']['mode']!='sequence' :
                pos,neg=pos,neg
                user_id=seq_data[0]
                pos4seq=pos
                user4seq=user_id
                neg4seq=neg
         
                graph_user_embedding,graph_item_embedding,graph_neg_item_embedding,graph_cl_loss,graph_click_emb,graph_inner_loss,graph_sampled_click_emb=self.graph_layer(blocks,constra_b=sampled_view,have_cart=have_cart,seq_tensor=seq_data,is_eval=is_eval)

            mutliView_user_embedding=torch.cat((seq_user_embedding,graph_user_embedding),dim=-1)
            mutliView_item_embedding=torch.cat((seq_item_embedding,graph_item_embedding),dim=-1)
            multiview_sampled_click_embedding=torch.cat((seq_sampled_click_emb,graph_sampled_click_emb),dim=1)
            mutliView_neg_item_embedding=torch.cat((seq_neg_item_embedding,graph_neg_item_embedding),dim=-1)
            multiview_sampled_click_embedding=self.item_fusion(multiview_sampled_click_embedding)
            mutliView_user_embedding=self.user_fusion(mutliView_user_embedding)
            mutliView_item_embedding=self.item_fusion(mutliView_item_embedding)
            mutliView_neg_item_embedding=self.item_fusion(mutliView_neg_item_embedding)
            cross_constra_loss=self.constrat(seq_user_embs=seq_user_embedding,graph_user_embs=graph_user_embedding, user_ids=user4seq)
            inner_cl_loss=self.inner_CL(graph_buy_emb=graph_click_emb[0],graph_click_emb=graph_click_emb[1],seq_buy_emb=seq_click_emb[0], seq_click_emb=seq_click_emb[1], user_ids=seq_data[0], have_diff_behaviors=have_constra)    

            mutliView_neg_item_embedding=mutliView_neg_item_embedding.reshape(-1,configs['train']['neg_sample_num'],mutliView_neg_item_embedding.shape[-1])
            neg_score=torch.bmm(mutliView_neg_item_embedding,mutliView_user_embedding.unsqueeze(-1)).squeeze(-1)
            pos_score=torch.mul(mutliView_user_embedding,mutliView_item_embedding).sum(-1).unsqueeze(-1)
            
            score=((pos_score-neg_score)).sigmoid()
            point_j=torch.cat((pos_score,neg_score),dim=1)
            link_loss=self.metric(score,torch.ones_like(score))
            mask=sampled_click!=-1
            mutliView_user_embedding=mutliView_user_embedding.unsqueeze(1).repeat(1,50,1)
            auc_pos_scores=pos_score.squeeze(1).tolist()
            pos_score=pos_score.repeat(1,50)
            pos_score=pos_score[mask]
            mutliView_user_embedding=mutliView_user_embedding[mask]
            mask=sampled_click[sampled_click!=-1]
            mask=mask>0
            masked_mul_user_emb=mutliView_user_embedding[mask]
            multiview_sampled_click_embedding=multiview_sampled_click_embedding[mask]
            masked_pos_score=pos_score[mask] ## B  
            click_score=torch.mul(masked_mul_user_emb,multiview_sampled_click_embedding).sum(-1)
            buy_click_score=(masked_pos_score-click_score).sigmoid()
            buy_click_loss=self.metric(buy_click_score,torch.ones_like(buy_click_score))
            loss=configs['model']['main_weight']*link_loss+ce_loss*configs['model']['seq_cons_weight']+graph_cl_loss*configs['model']['graph_cons_weight']+cross_constra_loss*configs['model']['cross_cons_weight']+inner_cl_loss*configs['model']['inner_loss_weight']+buy_click_loss*configs['model']['buy_click_weight']
            return loss,link_loss,ce_loss,graph_cl_loss,cross_constra_loss,graph_inner_loss,buy_click_loss,point_j

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.userEmbeddingLayer.weight, self.itemEmbeddingLayer.weight #todo veision2
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


class PR_loss_for_bert(torch.nn.Module):
    def __init__(self):
        super (PR_loss_for_bert,self).__init__()
        self.criterion = torch.nn.BCELoss(reduction='none')
    
    def forward(self,masked_item_sequence,point_i,point_j,is_eval=False):
        # print(point_i.shape,point_j.shape)
        distance=(point_i-point_j).sigmoid()
        distance=distance.flatten()
        distance = self.criterion(distance, torch.ones_like(distance, dtype=torch.float32))
        if is_eval:
            loss=torch.sum(distance)/distance.shape[0]
        else:
            mip_mask = (masked_item_sequence == configs['model']['mask_id']).float()
            loss = torch.sum(distance * mip_mask.flatten())/mip_mask.sum()
        return loss
