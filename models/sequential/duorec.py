import math
import random
from models.base_model import BaseModel
from models.model_utils import TransformerLayer, TransformerEmbedding
import numpy as np
import torch
from torch import nn
from config.configurator import configs


class DuoRec(BaseModel):
    def __init__(self, data_handler):
        super(DuoRec, self).__init__(data_handler)
        self.data_handler = data_handler
        self.item_num = configs['data']['item_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['model']['max_seq_len']
        self.mask_token = self.item_num + 1
        # load parameters info
        self.n_layers = configs['model']['n_layers']
        self.n_heads = configs['model']['n_heads']
        self.emb_size = configs['model']['embedding_size']
        # the dimensionality in feed-forward layer
        self.inner_size = 4 * self.emb_size
        self.dropout_rate = configs['model']['dropout_rate']

        self.batch_size = configs['train']['batch_size']
        self.lmd_sem = configs['model']['lmd_sem']
        self.tau = configs['model']['tau']

        self.emb_layer = TransformerEmbedding(
            self.item_num + 2, self.emb_size, self.max_len)

        self.transformer_layers = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.inner_size, self.dropout_rate) for _ in range(self.n_layers)])

        self.loss_func = nn.CrossEntropyLoss()

        self.mask_default = self._mask_correlated_samples(
            batch_size=self.batch_size)
        self.cl_loss_func = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)
        self.same_target_index = self._semantic_augmentation(data_handler)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _semantic_augmentation(self, data_handler):
        same_target_index = {}
        train_last_items = np.asarray(data_handler.train_dataloader.dataset.last_items, dtype=np.int32)
        soted_indices = np.argsort(train_last_items)
        train_last_items = train_last_items[soted_indices]
        pre_item_id = train_last_items[0]
        pre_idx = 0
        for idx, item_id in enumerate(train_last_items):
            if item_id != pre_item_id:
                last_group_indices = soted_indices[pre_idx:idx]
                if len(last_group_indices) > 20:
                    sampled_same_id = np.random.choice(last_group_indices, 20, replace=False)
                else:
                    sampled_same_id = last_group_indices
                if len(sampled_same_id) > 0:
                    same_target_index[pre_item_id] = sampled_same_id
                pre_item_id = item_id
                pre_idx = idx
        return same_target_index

    def _pad_seq(self, seq):
        if len(seq) >= self.max_len:
            seq = seq[-self.max_len:]
        else:
            # pad at the head
            seq = [0] * (self.max_len - len(seq)) + seq
        return seq

    def _duorec_aug(self, batch_seqs, batch_last_items):
        # last_items = batch_seqs[:, -1].tolist()
        last_items = batch_last_items.tolist()
        train_seqs = self.data_handler.train_dataloader.dataset.seqs
        sampled_pos_seqs = []
        for i, item in enumerate(last_items):
            if item in self.same_target_index:
                sampled_seq_idx = np.random.choice(self.same_target_index[item])
                sampled_pos_seqs.append(train_seqs[sampled_seq_idx])
            else:
                sampled_pos_seqs.append(batch_seqs[i].tolist())
        # padding 0 at the left
        sampled_pos_seqs = [self._pad_seq(seq) for seq in sampled_pos_seqs]
        sampled_pos_seqs = torch.tensor(sampled_pos_seqs, dtype=torch.long, device=batch_seqs.device)
        return sampled_pos_seqs

    def _mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def _info_nce(self, z_i, z_j, temp, batch_size):
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self._mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)
        return info_nce_loss

    def forward(self, batch_seqs):
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x = self.emb_layer(batch_seqs)
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        output = x[:, -1, :]
        return output  # [B H]

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_last_items = batch_data
        seq_output = self.forward(batch_seqs)

        test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_func(logits, batch_last_items)

        # NCE
        seq_output1 = self.forward(batch_seqs)
        sem_aug_seqs = self._duorec_aug(batch_seqs, batch_last_items)
        seq_output2 = self.forward(sem_aug_seqs)

        cl_loss = self.lmd_sem * self._info_nce(
            seq_output1, seq_output2, temp=self.tau, batch_size=seq_output1.shape[0])

        loss_dict = {
            'rec_loss': loss.item(),
            'cl_loss': cl_loss.item(),
        }
        return loss + cl_loss, loss_dict

    def full_predict(self, batch_data):
        batch_user, batch_seqs, _ = batch_data
        logits = self.forward(batch_seqs)
        test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
        scores = torch.matmul(logits, test_item_emb.transpose(0, 1))
        return scores
