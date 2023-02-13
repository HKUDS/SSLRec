from torch import nn as nn

from models.multi_behavior.bert_modules.embedding import BERTEmbedding
from models.multi_behavior.bert_modules.transformer import TransformerBlock
# from utils import fix_random_seed_as

from config.configurator import configs


class BERT(nn.Module):
    def __init__(self):
        super().__init__()

        # fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = configs['model']['max_seq_len']
       
        num_items = configs['model']['item_size']
        n_layers = configs['model']['bert_layer']
        heads = configs['model']['bert_num_heads']
        vocab_size =num_items
        hidden = configs['model']['embedding_size']
        self.hidden = hidden
        dropout = configs['model']['bert_dropout']

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x,token_embedding):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x,token_embedding)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass
