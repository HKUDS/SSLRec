from torch import nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.model_utils import TransformerLayer
from config.configurator import configs
import random
import torch


class BERTEmbLayer(nn.Module):
    def __init__(self, item_num, emb_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token_emb = nn.Embedding(item_num, emb_size, padding_idx=0)
        self.position_emb = nn.Embedding(max_len, emb_size)
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size

    def forward(self, batch_seqs):
        batch_size = batch_seqs.size(0)
        pos_emb = self.position_emb.weight.unsqueeze(
            0).repeat(batch_size, 1, 1)
        x = self.token_emb(batch_seqs) + pos_emb
        return self.dropout(x)


class BERT4Rec(BaseModel):
    def __init__(self, data_handler):
        super(BERT4Rec, self).__init__(data_handler)
        self.item_num = configs['data']['item_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['model']['max_seq_len']
        self.mask_token = self.item_num + 1

        self.emb_layer = BERTEmbLayer(
            self.item_num + 1, self.emb_size, self.max_len)

        self.n_layers = configs['model']['n_layers']
        self.n_heads = configs['model']['n_heads']
        self.dropout_rate = configs['model']['dropout_rate']
        self.mask_prob = configs['model']['mask_prob']

        self.transformer_layers = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.emb_size * 4, self.dropout_rate) for _ in range(self.n_layers)])

        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.out_fc = nn.Linear(self.emb_size, self.item_num + 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _transform_train_seq(self, batch_seqs, batch_last_items):
        device = batch_seqs.device
        seqs = torch.cat([batch_seqs, batch_last_items], dim=1)
        seqs = seqs.tolist()
        masked_seqs = []
        masked_items = []
        for seq in seqs:
            masked_seq = []
            masked_item = []
            for item in seq:
                if item == 0:
                    masked_seq.append(0)
                    masked_item.append(0)
                    continue
                prob = random.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob
                    if prob < 0.8:
                        masked_seq.append(self.mask_token)
                    elif prob < 0.9:
                        masked_seq.append(random.randint(1, self.item_num))
                    else:
                        masked_seq.append(item)
                    masked_item.append(item)
                else:
                    masked_seq.append(item)
                    masked_item.append(0)
            masked_seqs.append(masked_seq)
            masked_items.append(masked_item)
        masked_seqs = torch.tensor(masked_seqs, device=device, dtype=torch.long)[:, -self.max_len:]
        masked_items = torch.tensor(masked_items, device=device, dtype=torch.long)[:, -self.max_len:]
        return masked_seqs, masked_items

    def _transform_test_seq(self, batch_seqs):
        batch_mask_token = torch.LongTensor(
            [self.mask_token] * batch_seqs.size(0)).unsqueeze(1)
        seqs = torch.cat([batch_seqs, batch_mask_token], dim=1)
        return seqs[:, -self.max_len:]

    def forward(self, batch_seqs):
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x = self.emb_layer(batch_seqs)
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        return x

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_last_items = batch_data
        masked_seqs, masked_items = self._transform_train_seq(
            batch_seqs, batch_last_items)
        # B, T, E
        logits = self.forward(masked_seqs)
        logits = self.out_fc(logits)
        # B, T, E -> B*T, E
        logits = logits.view(-1, logits.size(-1))
        loss = self.loss_func(logits, masked_items.view(-1))
        return loss

    def full_predict(self, batch_data):
        batch_user, batch_seqs, _ = batch_data
        masked_seqs = self._transform_test_seq(batch_seqs)
        scores = self.forward(masked_seqs)
        scores = self.out_fc(scores)
        scores = scores[:, -1, :]
        return scores
