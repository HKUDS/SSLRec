import numpy as np
from numpy import random
import pickle
from scipy.sparse import csr_matrix
import math
import gc
import time
import random
import datetime

import torch as t
import torch.nn as nn
import torch.utils.data as dataloader
import torch.nn.functional as F
from torch.nn import init

from tqdm import tqdm
from .trainer import *
from models import *
from config.configurator import configs


class CMLTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(CMLTrainer, self).__init__(data_handler, logger)
        self.meta_weight_net = MetaWeightNet(len(self.data_handler.behaviors)).cuda()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])


        self.opt = t.optim.AdamW(model.parameters(), lr = configs['optimizer']['lr'], weight_decay = configs['optimizer']['opt_weight_decay'])
        self.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = configs['optimizer']['meta_lr'], weight_decay=configs['optimizer']['meta_opt_weight_decay'])
        self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, configs['optimizer']['opt_base_lr'], configs['optimizer']['opt_max_lr'], step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, configs['optimizer']['meta_opt_base_lr'], configs['optimizer']['meta_opt_max_lr'], step_size_up=2, step_size_down=3, mode='triangular', gamma=0.98, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)

    def train_epoch(self, model, epoch_idx):
        train_loader = self.data_handler.train_dataloader        
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample()

        print("end_ng_samp:  ", time)
        
        epoch_loss = 0
    
        #prepare
        self.behavior_loss_list = [None]*len(self.data_handler.behaviors)      
        self.user_id_list = [None]*len(self.data_handler.behaviors) 
        self.item_id_pos_list = [None]*len(self.data_handler.behaviors) 
        self.item_id_neg_list = [None]*len(self.data_handler.behaviors) 
        self.meta_start_index = 0
        self.meta_end_index = self.meta_start_index + configs['train']['meta_batch']  

        #epoch
        cnt = 0
        for user, item_i, item_j in tqdm(train_loader):  

            user = user.long().cuda()
            self.user_step_index = user

            self.meta_user = t.as_tensor(self.data_handler.meta_multi_single[self.meta_start_index:self.meta_end_index]).cuda()  
            
            if self.meta_end_index == self.data_handler.meta_multi_single.shape[0]:
                self.meta_start_index = 0  
            else:
                self.meta_start_index = (self.meta_start_index + configs['train']['meta_batch']) % (self.data_handler.meta_multi_single.shape[0] - 1)
            self.meta_end_index = min(self.meta_start_index + configs['train']['meta_batch'], self.data_handler.meta_multi_single.shape[0])


            #round one
            meta_behavior_loss_list = [None]*len(self.data_handler.behaviors) 
            meta_user_index_list = [None]*len(self.data_handler.behaviors)   
            meta_model = CML(self.data_handler).cuda()
            meta_opt = t.optim.AdamW(meta_model.parameters(), lr = configs['optimizer']['lr'], weight_decay = configs['optimizer']['opt_weight_decay'])   
            meta_model.load_state_dict(model.state_dict())
            meta_user_embed, meta_item_embed, meta_user_embeds, meta_item_embeds = meta_model()
            for index in range(len(self.data_handler.behaviors) ):
                not_zero_index = np.where(item_i[index].cpu().numpy()!=-1)[0]
                self.user_id_list[index] = user[not_zero_index].long().cuda()
                meta_user_index_list[index] = self.user_id_list[index]
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()
                meta_userEmbed = meta_user_embed[self.user_id_list[index]]
                meta_posEmbed = meta_item_embed[self.item_id_pos_list[index]]
                meta_negEmbed = meta_item_embed[self.item_id_neg_list[index]]
                meta_pred_i, meta_pred_j = 0, 0
                meta_pred_i, meta_pred_j = self._innerProduct(meta_userEmbed, meta_posEmbed, meta_negEmbed)
                meta_behavior_loss_list[index] = - (meta_pred_i.view(-1) - meta_pred_j.view(-1)).sigmoid().log()
            meta_infoNCELoss_list, SSL_user_step_index = self._SSL(meta_user_embeds, meta_item_embeds, meta_user_embed, meta_item_embed, self.user_step_index)
            meta_infoNCELoss_list_weights, meta_behavior_loss_list_weights = self.meta_weight_net(\
                                                                         meta_infoNCELoss_list, \
                                                                         meta_behavior_loss_list, \
                                                                         SSL_user_step_index, \
                                                                         meta_user_index_list, \
                                                                         meta_user_embeds, \
                                                                         meta_user_embed)
            
            for i in range(len(self.data_handler.behaviors) ):
                meta_infoNCELoss_list[i] = (meta_infoNCELoss_list[i]*meta_infoNCELoss_list_weights[i]).sum()
                meta_behavior_loss_list[i] = (meta_behavior_loss_list[i]*meta_behavior_loss_list_weights[i]).sum()   
            meta_bprloss = sum(meta_behavior_loss_list) / len(meta_behavior_loss_list)
            meta_infoNCELoss = sum(meta_infoNCELoss_list) / len(meta_infoNCELoss_list)
            meta_regLoss = (t.norm(meta_userEmbed) ** 2 + t.norm(meta_posEmbed) ** 2 + t.norm(meta_negEmbed) ** 2)            
            meta_model_loss = (meta_bprloss + configs['train']['reg'] * meta_regLoss + configs['train']['beta']*meta_infoNCELoss) / configs['train']['batch_size']
            meta_opt.zero_grad(set_to_none=True)
            self.meta_opt.zero_grad(set_to_none=True)
            meta_model_loss.backward()
            nn.utils.clip_grad_norm_(self.meta_weight_net.parameters(), max_norm=20, norm_type=2)
            nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=20, norm_type=2)
            meta_opt.step()
            self.meta_opt.step()
   
            #round two
            behavior_loss_list = [None]*len(self.data_handler.behaviors) 
            user_index_list = [None]*len(self.data_handler.behaviors)   
            user_embed, item_embed, user_embeds, item_embeds = meta_model()
            for index in range(len(self.data_handler.behaviors) ):
                user_id, item_id_pos, item_id_neg = self._sampleTrainBatch(t.as_tensor(self.meta_user), self.data_handler.behaviors_data[i])
                user_index_list[index] = user_id
                userEmbed = user_embed[user_id]
                posEmbed = item_embed[item_id_pos]
                negEmbed = item_embed[item_id_neg]
                pred_i, pred_j = self._innerProduct(userEmbed, posEmbed, negEmbed)
                behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()  
            self.infoNCELoss_list, SSL_user_step_index = self._SSL(user_embeds, item_embeds, user_embed, item_embed, self.meta_user)
            infoNCELoss_list_weights, behavior_loss_list_weights = self.meta_weight_net(\
                                                                         self.infoNCELoss_list, \
                                                                         behavior_loss_list, \
                                                                         SSL_user_step_index, \
                                                                         user_index_list, \
                                                                         user_embeds, \
                                                                         user_embed)
            for i in range(len(self.data_handler.behaviors) ):
                self.infoNCELoss_list[i] = (self.infoNCELoss_list[i]*infoNCELoss_list_weights[i]).sum()
                behavior_loss_list[i] = (behavior_loss_list[i]*behavior_loss_list_weights[i]).sum()   

            bprloss = sum(behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(self.infoNCELoss_list) / len(self.infoNCELoss_list)
            round_two_regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)
            meta_loss = 0.5 * (bprloss + configs['train']['reg'] * round_two_regLoss  + configs['train']['beta']*infoNCELoss) / configs['train']['batch_size'] 
            self.meta_opt.zero_grad()
            meta_loss.backward()
            nn.utils.clip_grad_norm_(self.meta_weight_net.parameters(), max_norm=20, norm_type=2)
            self.meta_opt.step()



            #round three
            user_embed, item_embed, user_embeds, item_embeds = model()
            for index in range(len(self.data_handler.behaviors) ):
                userEmbed = user_embed[self.user_id_list[index]]
                posEmbed = item_embed[self.item_id_pos_list[index]]
                negEmbed = item_embed[self.item_id_neg_list[index]]
                pred_i, pred_j = 0, 0
                pred_i, pred_j = self._innerProduct(userEmbed, posEmbed, negEmbed)
                self.behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()  
            infoNCELoss_list, SSL_user_step_index = self._SSL(user_embeds, item_embeds, user_embed, item_embed, self.user_step_index)
            with t.no_grad():
                infoNCELoss_list_weights, behavior_loss_list_weights = self.meta_weight_net(\
                                                                            infoNCELoss_list, \
                                                                            self.behavior_loss_list, \
                                                                            SSL_user_step_index, \
                                                                            self.user_id_list, \
                                                                            user_embeds, \
                                                                            user_embed)
            for i in range(len(self.data_handler.behaviors) ):
                infoNCELoss_list[i] = (infoNCELoss_list[i]*infoNCELoss_list_weights[i]).sum()
                self.behavior_loss_list[i] = (self.behavior_loss_list[i]*behavior_loss_list_weights[i]).sum()  
            bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(infoNCELoss_list) / len(infoNCELoss_list)
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)
            loss = (bprloss + configs['train']['reg'] * regLoss + configs['train']['beta']*infoNCELoss) / configs['train']['batch_size']
            epoch_loss = epoch_loss + loss.item()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()

            cnt+=1

        return epoch_loss, user_embed, item_embed, user_embeds, item_embeds


    def _innerProduct(self, u, i, j):  
        pred_i = t.sum(t.mul(u,i), dim=1)*configs['model']['inner_product_mult']  
        pred_j = t.sum(t.mul(u,j), dim=1)*configs['model']['inner_product_mult']  
        return pred_i, pred_j

    def _negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def _sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds.cpu()].toarray()
        batch = len(batIds)
        user_id = [] 
        item_id_pos = [] 
        item_id_neg = [] 
 
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            sampNum = min(configs['train']['sampNum'], len(posset))  
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self._negSamp(temLabel[i], sampNum, labelMat.shape[1])

            for j in range(sampNum):
                user_id.append(batIds[i].item())
                item_id_pos.append(poslocs[j].item()) 
                item_id_neg.append(neglocs[j])
                cur += 1

        return t.as_tensor(np.array(user_id)).cuda(), t.as_tensor(np.array(item_id_pos)).cuda(), t.as_tensor(np.array(item_id_neg)).cuda() 



    def _SSL(self, user_embeddings, item_embeddings, target_user_embeddings, target_item_embeddings, user_step_index):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[t.randperm(embedding.size()[0])]  
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[t.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,t.randperm(corrupted_embedding.size()[1])]  
            return corrupted_embedding
        def score(x1, x2):
            return t.sum(t.mul(x1, x2), 1)

        def neg_sample_pair(x1, x2, τ = 0.05):  
            for i in range(x1.shape[0]):
                index_set = set(np.arange(x1.shape[0]))
                index_set.remove(i)
                index_set_neg = t.as_tensor(np.array(list(index_set))).long().cuda()  

                x_pos = x1[i].repeat(x1.shape[0]-1, 1)
                x_neg = x2[index_set]  
                
                if i==0:
                    x_pos_all = x_pos
                    x_neg_all = x_neg
                else:
                    x_pos_all = t.cat((x_pos_all, x_pos), 0)
                    x_neg_all = t.cat((x_neg_all, x_neg), 0)
            x_pos_all = t.as_tensor(x_pos_all)  #[9900, 100]
            x_neg_all = t.as_tensor(x_neg_all)  #[9900, 100]  

            return x_pos_all, x_neg_all

        def one_neg_sample_pair_index(i, step_index, embedding1, embedding2):

            index_set = set(np.array(step_index))
            index_set.remove(i.item())
            neg2_index = t.as_tensor(np.array(list(index_set))).long().cuda()

            neg1_index = t.ones((2,), dtype=t.long)
            neg1_index = neg1_index.new_full((len(index_set),), i)

            neg_score_pre = t.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze())
            return neg_score_pre

        def multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2):  #small, big, target, beh: [100], [1024], [31882, 16], [31882, 16]

            index_set = set(np.array(step_index.cpu()))
            batch_index_set = set(np.array(batch_index.cpu()))
            neg2_index_set = index_set - batch_index_set                         #beh
            neg2_index = t.as_tensor(np.array(list(neg2_index_set))).long().cuda()  #[910]
            neg2_index = t.unsqueeze(neg2_index, 0)                              #[1, 910]
            neg2_index = neg2_index.repeat(len(batch_index), 1)                  #[100, 910]
            neg2_index = t.reshape(neg2_index, (1, -1))                          #[1, 91000]
            neg2_index = t.squeeze(neg2_index)                                   #[91000]
                                                                                 #target
            neg1_index = batch_index.long().cuda()     #[100]
            neg1_index = t.unsqueeze(neg1_index, 1)                              #[100, 1]
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))               #[100, 910]
            neg1_index = t.reshape(neg1_index, (1, -1))                          #[1, 91000]           
            neg1_index = t.squeeze(neg1_index)                                   #[91000]

            neg_score_pre = t.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze().view(len(batch_index), -1), -1)  #[91000,1]==>[91000]==>[100, 910]==>[100]
            return neg_score_pre  #[100]

        def compute(x1, x2, neg1_index=None, neg2_index=None, τ = 0.05):  #[1024, 16], [1024, 16]

            if neg1_index!=None:
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]

            N = x1.shape[0]  
            D = x1.shape[1]

            x1 = x1
            x2 = x2

            scores = t.exp(t.div(t.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1)+1e-8))  #[1024, 1]
            
            return scores
        def single_infoNCE_loss_simple(embedding1, embedding2):
            pos = score(embedding1, embedding2)  #[100]
            neg1 = score(embedding2, row_column_shuffle(embedding1))  
            one = t.cuda.FloatTensor(neg1.shape[0]).fill_(1)  #[100]
            # one = zeros = t.ones(neg1.shape[0])
            con_loss = t.sum(-t.log(1e-8 + t.sigmoid(pos))-t.log(1e-8 + (one - t.sigmoid(neg1))))  
            return con_loss

        #use_less    
        def single_infoNCE_loss(embedding1, embedding2):
            N = embedding1.shape[0]
            D = embedding1.shape[1]

            pos_score = compute(embedding1, embedding2).squeeze()  #[100, 1]

            neg_x1, neg_x2 = neg_sample_pair(embedding1, embedding2)  #[9900, 100], [9900, 100]
            neg_score = t.sum(compute(neg_x1, neg_x2).view(N, (N-1)), dim=1)  #[100]  
            con_loss = -t.log(1e-8 +t.div(pos_score, neg_score))   
            con_loss = t.mean(con_loss)  
            return max(0, con_loss)

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index):  #target, beh
            N = step_index.shape[0]
            D = embedding1.shape[1]

            pos_score = compute(embedding1[step_index], embedding2[step_index]).squeeze()  #[1024]
            neg_score = t.zeros((N,), dtype = t.float64).cuda()  #[1024]

            #-------------------------------------------------multi version-----------------------------------------------------
            steps = int(np.ceil(N / configs['train']['SSL_batch']))  
            for i in range(steps):
                st = i * configs['train']['SSL_batch']
                ed = min((i+1) * configs['train']['SSL_batch'], N)
                batch_index = step_index[st: ed]

                neg_score_pre = multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2)
                if i ==0:
                    neg_score = neg_score_pre
                else:
                    neg_score = t.cat((neg_score, neg_score_pre), 0)
            #-------------------------------------------------multi version-----------------------------------------------------

            con_loss = -t.log(1e-8 +t.div(pos_score, neg_score+1e-8))  #[1024]/[1024]==>1024


            assert not t.any(t.isnan(con_loss))
            assert not t.any(t.isinf(con_loss))

            return t.where(t.isnan(con_loss), t.full_like(con_loss, 0+1e-8), con_loss)

        user_con_loss_list = []
        item_con_loss_list = []

        SSL_len = int(user_step_index.shape[0]/10)
        user_step_index = t.as_tensor(np.random.choice(user_step_index.cpu(), size=SSL_len, replace=False, p=None)).cuda()

        for i in range(len(self.data_handler.behaviors)):

            user_con_loss_list.append(single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index))

        user_con_losss = t.stack(user_con_loss_list, dim=0)  

        return user_con_loss_list, user_step_index  #4*[1024]
