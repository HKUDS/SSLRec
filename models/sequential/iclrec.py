import math
import random
from models.base_model import BaseModel
from models.model_utils import TransformerLayer, TransformerEmbedding
import numpy as np
import torch
from torch import nn
from config.configurator import configs
import faiss


class PCLoss(nn.Module):
    """ Reference: https://github.com/salesforce/PCL/blob/018a929c53fcb93fd07041b1725185e1237d2c0e/pcl/builder.py#L168
    """

    def __init__(self, temperature, device, contrast_mode="all"):
        super(PCLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.criterion = NCELoss(temperature, device)

    def forward(self, batch_sample_one, batch_sample_two, intents):
        """
        features: 
        intents: num_clusters x batch_size x hidden_dims
        """
        # instance contrast with prototypes
        mean_pcl_loss = 0
        pos_one_compare_loss = self.criterion(
            batch_sample_one, intents, intent_ids=None)
        pos_two_compare_loss = self.criterion(
            batch_sample_two, intents, intent_ids=None)
        mean_pcl_loss += pos_one_compare_loss
        mean_pcl_loss += pos_two_compare_loss
        mean_pcl_loss /= 2
        return mean_pcl_loss


class NCELoss(nn.Module):

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two, intent_ids=None):
        sim11 = torch.matmul(
            batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(
            batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(
            batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        # avoid contrast against positive intents
        if intent_ids is not None:
            intent_ids = intent_ids.contiguous().view(-1, 1)
            mask_11_22 = torch.eq(
                intent_ids, intent_ids.T).long().to(self.device)
            sim11[mask_11_22 == 1] = float("-inf")
            sim22[mask_11_22 == 1] = float("-inf")
            eye_metrix = torch.eye(d, dtype=torch.long).to(self.device)
            mask_11_22[eye_metrix == 1] = 0
            sim12[mask_11_22 == 1] = float("-inf")
        else:
            mask = torch.eye(d, dtype=torch.long).to(self.device)
            sim11[mask == 1] = float("-inf")
            sim22[mask == 1] = float("-inf")

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = "cpu"
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid
        
        index = faiss.IndexFlatL2(hidden_size)
        # res = faiss.StandardGpuResources()
        # res.noTempMemory()
        # cfg = faiss.GpuIndexFlatConfig()
        # cfg.useFloat16 = False
        # cfg.device = self.gpu_id
        # index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(
            self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        # for each sample, find cluster distance and assignments
        D, I = self.index.search(x, 1)
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]


class ICLRec(BaseModel):
    def __init__(self, data_handler):
        super(ICLRec, self).__init__(data_handler)
        self.device = configs['device']
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
        self.cl_weight = configs['model']['cl_weight']
        self.intent_cl_weight = configs['model']['intent_cl_weight']
        self.tau = configs['model']['tau']

        self.emb_layer = TransformerEmbedding(
            self.item_num + 2, self.emb_size, self.max_len)

        self.transformer_layers = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.inner_size, self.dropout_rate) for _ in range(self.n_layers)])

        self.loss_func = nn.CrossEntropyLoss()

        self.mask_default = self.mask_correlated_samples(
            batch_size=self.batch_size)

        self.cl_criterion = NCELoss(self.tau, self.device)
        self.pcl_criterion = PCLoss(self.tau, self.device)

        # intent clustering
        self.num_intent_clusters = configs['model']['num_intent_clusters']
        # initialize Kmeans
        self.cluster = KMeans(
            num_cluster=self.num_intent_clusters,
            seed=configs['train']['seed'],
            hidden_size=self.emb_size,
        )
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _cl4srec_aug(self, batch_seqs):
        def item_crop(seq, length, eta=0.2):
            num_left = math.floor(length * eta)
            crop_begin = random.randint(0, length - num_left)
            if num_left < 1:
                return seq[crop_begin:], 0
            croped_item_seq = np.zeros_like(seq)
            if crop_begin + num_left < seq.shape[0]:
                croped_item_seq[-num_left:] = seq[-(
                    crop_begin + 1 + num_left):-(crop_begin + 1)]
            else:
                croped_item_seq[-num_left:] = seq[-(crop_begin + 1):]
            return croped_item_seq.tolist(), num_left

        def item_mask(seq, length, gamma=0.7):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = seq[:]
            # token 0 has been used for semantic masking
            mask_index = [-i-1 for i in mask_index]
            masked_item_seq[mask_index] = self.mask_token
            return masked_item_seq.tolist(), length

        def item_reorder(seq, length, beta=0.2):
            num_reorder = math.floor(length * beta)
            reorder_begin = random.randint(0, length - num_reorder)
            reordered_item_seq = seq[:]
            shuffle_index = list(
                range(reorder_begin, reorder_begin + num_reorder))
            random.shuffle(shuffle_index)
            shuffle_index = [-i for i in shuffle_index]
            reordered_item_seq[-(reorder_begin + 1 + num_reorder):-
                               (reorder_begin+1)] = reordered_item_seq[shuffle_index]
            return reordered_item_seq.tolist(), length

        seqs = batch_seqs.tolist()
        lengths = batch_seqs.count_nonzero(dim=1).tolist()

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        for seq, length in zip(seqs, lengths):
            seq = np.asarray(seq.copy(), dtype=np.int64)
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq, length)

            if aug_len > 0:
                aug_seq1.append(aug_seq)
                aug_len1.append(aug_len)
            else:
                aug_seq1.append(seq.tolist())
                aug_len1.append(length)

            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq, length)

            if aug_len > 0:
                aug_seq2.append(aug_seq)
                aug_len2.append(aug_len)
            else:
                aug_seq2.append(seq.tolist())
                aug_len2.append(length)

        aug_seq1 = torch.tensor(
            aug_seq1, dtype=torch.long, device=batch_seqs.device)
        aug_seq2 = torch.tensor(
            aug_seq2, dtype=torch.long, device=batch_seqs.device)
        return aug_seq1, aug_seq2

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, batch_seqs, return_mean=False):
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x = self.emb_layer(batch_seqs)
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        if return_mean:
            output = torch.mean(x, dim=1)
        else:
            output = x[:, -1, :]
        return output  # [B H]

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_last_items, batch_negs = batch_data
        seq_output = self.forward(batch_seqs)

        # Cross Entropy
        # [batch_size, hidden_size]
        pos_item_emb = self.emb_layer.token_emb.weight[batch_last_items]
        neg_item_emb = self.emb_layer.token_emb.weight[batch_negs]
        pos_logits = torch.sum(
            pos_item_emb * seq_output, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg_item_emb * seq_output, -1)
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24)
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        ) / batch_seqs.shape[0]

        # CL4SRec
        aug_seq1, aug_seq2 = self._cl4srec_aug(batch_seqs)
        seq_output1 = self.forward(aug_seq1, return_mean=True)
        seq_output2 = self.forward(aug_seq2, return_mean=True)

        cl_loss = self.cl_weight * self.cl_criterion(
            seq_output1, seq_output2)

        # Intent CL
        seq_output_mean = self.forward(batch_seqs, return_mean=True)
        seq_output_mean = seq_output_mean.detach().cpu().numpy()
        intent_id, seq2intent = self.cluster.query(seq_output_mean)
        seq_output1_intent = self.forward(aug_seq1, return_mean=True)
        seq_output2_intent = self.forward(aug_seq2, return_mean=True)
        intent_cl_loss = self.intent_cl_weight * self.pcl_criterion(
            seq_output1_intent, seq_output2_intent, intents=seq2intent.to(self.device))

        loss_dict = {
            'rec_loss': loss.item(),
            'cl_loss': cl_loss.item(),
            'intent_cl_loss': intent_cl_loss.item()
        }
        return loss + cl_loss + intent_cl_loss, loss_dict

    def full_predict(self, batch_data):
        batch_user, batch_seqs, _ = batch_data
        logits = self.forward(batch_seqs)
        test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
        scores = torch.matmul(logits, test_item_emb.transpose(0, 1))
        return scores
