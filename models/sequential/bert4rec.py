from torch import nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.model_utils import TransformerLayer, TransformerEmbedding
from config.configurator import configs
import random
import torch

class BERT4Rec(BaseModel):
    def __init__(self, data_handler):
        super(BERT4Rec, self).__init__(data_handler)
        self.item_num = configs['data']['item_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['model']['max_seq_len']
        self.mask_token = self.item_num + 1

        self.emb_layer = TransformerEmbedding(
            self.item_num + 2, self.emb_size, self.max_len)

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
        for seq in seqs: # for each seq
            masked_seq = []
            masked_item = []
            for item in seq: # for each item
                if item == 0: # ignore 0 idx (padding)
                    masked_seq.append(0)
                    masked_item.append(0)
                    continue
                prob = random.random()
                if prob < self.mask_prob: # mask
                    prob /= self.mask_prob
                    if prob < 0.8:
                        masked_seq.append(self.mask_token)
                    elif prob < 0.9: # replace
                        masked_seq.append(random.randint(1, self.item_num)) # both include
                    else: # keep
                        masked_seq.append(item)
                    masked_item.append(item)
                else: # not mask
                    masked_seq.append(item) # keep
                    masked_item.append(0) # 0 represent no item
            masked_seqs.append(masked_seq)
            masked_items.append(masked_item)
        masked_seqs = torch.tensor(masked_seqs, device=device, dtype=torch.long)[:, -self.max_len:]
        masked_items = torch.tensor(masked_items, device=device, dtype=torch.long)[:, -self.max_len:]
        return masked_seqs, masked_items

    def _transform_test_seq(self, batch_seqs):
        batch_mask_token = torch.LongTensor(
            [self.mask_token] * batch_seqs.size(0)).unsqueeze(1).to(batch_seqs.device)
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
            batch_seqs, batch_last_items.unsqueeze(1))
        # B, T, E
        logits = self.forward(masked_seqs) # [b, l]
        logits = self.out_fc(logits) # [b, l, n+1]
        # B, T, E -> B*T, E
        logits = logits.view(-1, logits.size(-1)) # [b*l, n+1]
        loss = self.loss_func(logits, masked_items.reshape(-1))
        loss_dict = {'rec_loss': loss.item()}
        return loss, loss_dict

    def full_predict(self, batch_data):
        batch_user, batch_seqs, _ = batch_data
        masked_seqs = self._transform_test_seq(batch_seqs)
        scores = self.forward(masked_seqs)
        scores = self.out_fc(scores)
        scores = scores[:, -1, :]
        return scores