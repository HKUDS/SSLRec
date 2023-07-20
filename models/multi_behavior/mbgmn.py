import os
import numpy as np
from time import time
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_pick_embeds
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop


class MBGMN(BaseModel):
    def __init__(self, data_handler):
        super(MBGMN, self).__init__(data_handler)

        self.data_handler = data_handler
        self.n_users = configs['data']['user_num']
        self.n_items = configs['data']['item_num']
        self.embedding_dim = configs['model']['embedding_size']
        self.act = nn.LeakyReLU(0.1)
        self.adj = data_handler.behavior_mats[len(data_handler.behaviors)-1]['A']

        uEmbed0 = nn.Embedding(self.n_users, self.embedding_dim//2)
        iEmbed0 = nn.Embedding(self.n_items, self.embedding_dim//2)
        behEmbeds = nn.Embedding(len(self.data_handler.behaviors), self.embedding_dim//2)
        self.ulat = [None]*(len(self.data_handler.behaviors)+1)
        self.ilat = [None]*(len(self.data_handler.behaviors)+1)
        nn.init.xavier_uniform_(uEmbed0.weight)
        nn.init.xavier_uniform_(iEmbed0.weight)
        nn.init.xavier_uniform_(behEmbeds.weight)
        self.uEmbed0 = uEmbed0.weight 
        self.iEmbed0 = iEmbed0.weight
        self.behEmbeds = behEmbeds.weight

        self.metaForSpecialize_linear_u = nn.Linear(int(3*(self.embedding_dim/2)), int(self.embedding_dim/2))
        self.metaForSpecialize_linear_i = nn.Linear(int(3*(self.embedding_dim/2)), int(self.embedding_dim/2))
        self.metaForSpecialize_linear_u1 = nn.Linear(int(self.embedding_dim/2), int(configs['model']['rank'] * (self.embedding_dim/2)))
        self.metaForSpecialize_linear_i1 = nn.Linear(int(self.embedding_dim/2), int(configs['model']['rank'] * (self.embedding_dim/2)))
        self.metaForSpecialize_linear_u2 = nn.Linear(int(self.embedding_dim/2), int(configs['model']['rank'] * (self.embedding_dim/2)))
        self.metaForSpecialize_linear_i2 = nn.Linear(int(self.embedding_dim/2), int(configs['model']['rank'] * (self.embedding_dim/2)))
        self.predMeta_FC1 = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        self.predMeta_FC2 = nn.Linear(self.embedding_dim * 3, self.embedding_dim * 3)
        self.predMeta_FC3 = nn.Linear(self.embedding_dim * 3, self.embedding_dim * 3 * self.embedding_dim)
        self.predMeta_FC4 = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        self.predMeta_FC5 = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        self.Q = nn.Parameter(torch.Tensor(configs['model']['embedding_size'], configs['model']['embedding_size']))
        nn.init.xavier_uniform_(self.Q)

    def forward(self):
      for beh in range(len(self.data_handler.behaviors)):
        params = self.metaForSpecialize(self.uEmbed0, self.iEmbed0, self.behEmbeds[beh], [self.data_handler.behavior_mats[beh]['A']], [self.data_handler.behavior_mats[beh]['AT']])
        behUEmbed0, behIEmbed0 = self.specialize(self.uEmbed0, self.iEmbed0, params)
        ulats = [behUEmbed0]
        ilats = [behIEmbed0]
        for i in range(configs['model']['layer_num']):
          ulat = self.messagePropagate(ilats[-1], self.data_handler.behavior_mats[beh]['A'], ulats[-1])
          ilat = self.messagePropagate(ulats[-1], self.data_handler.behavior_mats[beh]['AT'], ilats[-1])
          ulats.append(ulat + ulats[-1])
          ilats.append(ilat + ilats[-1])
        self.ulat[beh] = torch.stack(ulats).sum(0)
        self.ilat[beh] = torch.stack(ilats).sum(0)

      params = self.metaForSpecialize(self.uEmbed0, self.iEmbed0, self.behEmbeds[-1])
      behUEmbed0, behIEmbed0 = self.specialize(self.uEmbed0, self.iEmbed0, params)
      ulats = [behUEmbed0]
      ilats = [behIEmbed0]
      for i in range(configs['model']['layer_num']):
        ubehLats = []
        ibehLats = []
        for beh in range(len(self.data_handler.behaviors)):
          ulat = self.messagePropagate(ilats[-1], self.data_handler.behavior_mats[beh]['A'], ulats[-1])
          ilat = self.messagePropagate(ulats[-1], self.data_handler.behavior_mats[beh]['AT'], ilats[-1])
          ubehLats.append(ulat)
          ibehLats.append(ilat)
        ulat = torch.stack(self.lightSelfAttention(ubehLats, len(self.data_handler.behaviors), self.embedding_dim, configs['model']['att_head'] )).sum(0)
        ilat = torch.stack(self.lightSelfAttention(ibehLats, len(self.data_handler.behaviors), self.embedding_dim, configs['model']['att_head'] )).sum(0)
        ulats.append(ulat)
        ilats.append(ilat)
      self.ulat[-1] = torch.stack(ulats).sum(0)
      self.ilat[-1] = torch.stack(ilats).sum(0)
      return self.ulat[-1], self.ilat[-1]


    def messagePropagate(self, lats, adj, lats2):
      return self.act(torch.spmm(adj, lats))

    def metaForSpecialize(self, uEmbed, iEmbed, behEmbed, adjs=None, tpAdjs=None):
      rank = configs['model']['rank']
      uNeighbor = iNeighbor = 0
      if adjs!=None:
        for i in range(len(adjs)):
          uNeighbor += torch.spmm(adjs[i], iEmbed)
          iNeighbor += torch.spmm(tpAdjs[i], uEmbed)
      else:
        for i in range(len(self.data_handler.behaviors)):
          uNeighbor += torch.spmm(self.data_handler.behavior_mats[i]['A'], iEmbed)
          iNeighbor += torch.spmm(self.data_handler.behavior_mats[i]['AT'], uEmbed)
      ubehEmbed = behEmbed.repeat(uEmbed.shape[0], 1) * torch.ones_like(uEmbed)
      ibehEmbed = behEmbed.repeat(iEmbed.shape[0], 1) * torch.ones_like(iEmbed)  
      uMetaLat = self.act(self.metaForSpecialize_linear_u(torch.cat((ubehEmbed, uEmbed, uNeighbor), dim=-1)))
      iMetaLat = self.act(self.metaForSpecialize_linear_i(torch.cat((ibehEmbed, iEmbed, iNeighbor), dim=-1)))
      uW1 = torch.reshape(self.act(self.metaForSpecialize_linear_u1(uMetaLat)), (-1, int(self.embedding_dim/2), rank))
      uW2 = torch.reshape(self.act(self.metaForSpecialize_linear_u2(uMetaLat)), (-1, rank, int(self.embedding_dim/2)))
      iW1 = torch.reshape(self.act(self.metaForSpecialize_linear_i1(iMetaLat)), (-1, int(self.embedding_dim/2), rank))
      iW2 = torch.reshape(self.act(self.metaForSpecialize_linear_i2(iMetaLat)), (-1, rank, int(self.embedding_dim/2)))
      params = {'uW1': uW1, 'uW2': uW2, 'iW1': iW1, 'iW2': iW2}
      return params

    def specialize(self, uEmbed, iEmbed, params):
      retUEmbed = torch.sum( uEmbed.unsqueeze(-1) * params['uW1'], dim=1)
      retUEmbed = torch.sum( retUEmbed.unsqueeze(-1) * params['uW2'], dim=1)
      retUEmbed = torch.cat([retUEmbed, uEmbed], axis=-1)
      retIEmbed = torch.sum( iEmbed.unsqueeze(-1) * params['iW1'], dim=1)
      retIEmbed = torch.sum( retIEmbed.unsqueeze(-1) * params['iW2'], dim=1)
      retIEmbed = torch.cat([retIEmbed, iEmbed], dim=-1)
      return retUEmbed, retIEmbed



    def lightSelfAttention(self, localReps, number, inpDim, numHeads):
      rspReps = torch.reshape(torch.stack(localReps, axis=1), (-1, inpDim))
      tem = rspReps @ self.Q
      q = torch.reshape(tem, (-1, number, 1, numHeads, inpDim//numHeads))
      k = torch.reshape(tem, (-1, 1, number, numHeads, inpDim//numHeads))
      v = torch.reshape(rspReps, (-1, 1, number, numHeads, inpDim//numHeads))
      att = F.softmax(torch.sum(q * k, axis=-1, keepdims=True) / torch.sqrt(torch.tensor(inpDim/numHeads)), dim=2)
      attval = torch.reshape(torch.sum(att * v, dim=2), (-1, number, inpDim))
      rets = [None] * number
      for i in range(number):
        tem1 = torch.reshape(attval[:,i], (-1, inpDim))  
        rets[i] = tem1 + localReps[i]
      return rets



    def metaForPredict(self, src_ulat, src_ilat, tgt_ulat, tgt_ilat):
      self.embedding_dim = self.embedding_dim

      src_ui = self.act(self.predMeta_FC1( torch.cat((src_ulat * src_ilat, src_ulat, src_ilat), dim=-1) ))  # N*3->N
      tgt_ui = self.act(self.predMeta_FC1( torch.cat((tgt_ulat * tgt_ilat, tgt_ulat, tgt_ilat), dim=-1) ))  # N*3->N
      metalat = self.act(self.predMeta_FC2( torch.cat((src_ui * tgt_ui, src_ui, tgt_ui), dim=-1) ))  # N*3->N*3    [5084, 48] 
      w1 = torch.reshape(self.act(self.predMeta_FC3(metalat)) , (-1, self.embedding_dim * 3, self.embedding_dim))  #N*3->N*3*N    [5084, 48, 16]
      b1 = torch.reshape(self.act(self.predMeta_FC4(metalat)) , (-1, 1, self.embedding_dim))  # N*3->N  [15252, 1, 16]
      w2 = torch.reshape(self.act(self.predMeta_FC5(metalat)) , (-1, self.embedding_dim, 1))  # N*3->N  [5084, 16, 1]
      params = {
        'w1': w1,
        'b1': b1,
        'w2': w2
      }
      return params

    def _predict(self, ulat, ilat, params):
      predEmbed = torch.unsqueeze(torch.cat((ulat * ilat, ulat, ilat), dim=-1), dim=1)
      predEmbed = self.act(predEmbed @ params['w1'] + params['b1']) 
      preds = torch.squeeze(predEmbed @ params['w2'])
      return preds

    def predict(self, src, tgt):
      uids = self.uids[tgt]
      iids = self.iids[tgt]
      src_ulat = self.ulat[src][uids]
      src_ilat = self.ilat[src][iids]
      tgt_ulat = self.ulat[tgt][uids]
      tgt_ilat = self.ilat[tgt][iids]
      predParams = self.metaForPredict(src_ulat, src_ilat, tgt_ulat, tgt_ilat)
      return self._predict(src_ulat, src_ilat, predParams) * configs['model']['mult']
    
    def cal_loss(self, uids, iids):
        self.uids, self.iids = uids, iids 
        self.is_training = True
        user_embeds, item_embeds = self.forward()
      
        self.preLoss = 0
        for src in range(len(self.data_handler.behaviors) + 1):
          for tgt in range(len(self.data_handler.behaviors)):
            preds = self.predict(src, tgt)
            sampNum = len(self.uids[tgt]) // 2
            posPred = preds[:sampNum]
            negPred = preds[sampNum:]   
            self.preLoss += torch.mean(torch.maximum(torch.tensor(0.0).clone().detach(),  torch.tensor(1.0 - (posPred - negPred)).clone().detach()) )
            if src == self.data_handler.behaviors and tgt == self.data_handler.behaviors - 1:
              self.targetPreds = preds
        self.regLoss = configs['train']['reg'] * reg_pick_embeds([user_embeds, item_embeds])  # better version
        self.loss = self.preLoss + self.regLoss
        return self.loss


    def _regularize(self, names=None, method='L2'):
      ret = 0
      if method == 'L1':
        if names != None:
          for name in names:
            ret += torch.sum(torch.abs(getParam(name)))
        else:
          for name in regParams:
            ret += torch.sum(torch.abs(regParams[name]))
      elif method == 'L2':
        if names != None:
          for name in names:
            ret += torch.sum(torch.square(getParam(name)))
        else:
          for name in regParams:
            ret += torch.sum(torch.square(regParams[name]))
      return ret



    def full_predict(self, batch_data):  
        user_embeds, item_embeds = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

