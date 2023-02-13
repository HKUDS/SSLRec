from torch import nn
import torch.nn.functional as F
from models.base_model import BaseModel
from config.configurator import configs

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
        pos_emb = self.position_emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.token_emb(batch_seqs) + pos_emb
        return self.dropout(x)

class BERT4Rec(BaseModel):
    def __init__(self, data_handler):
        super(BERT4Rec, self).__init__(data_handler)
        self.item_num = configs['data']['item_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['data']['max_seq_len']

        self.emb_layer = BERTEmbLayer(self.item_num, self.emb_size, self.max_len)