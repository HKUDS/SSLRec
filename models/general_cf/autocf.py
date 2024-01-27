import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import reg_params
from models.base_model import BaseModel

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class AutoCF(BaseModel):
	def __init__(self, data_handler):
		super(AutoCF, self).__init__(data_handler)

		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

		self.adj = data_handler.torch_adj
		self.all_one_adj = self.make_all_one_adj()

		self.gcn_layer = configs['model']['gcn_layer']
		self.gt_layer = configs['model']['gt_layer']
		self.reg_weight = configs['model']['reg_weight']
		self.ssl_reg = configs['model']['ssl_reg']
		
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.gcn_layer)])
		self.gtLayers = nn.Sequential(*[GTLayer() for i in range(self.gt_layer)])

		self.masker = RandomMaskSubgraphs()
		self.sampler = LocalGraph()

	def make_all_one_adj(self):
		idxs = self.adj._indices()
		vals = t.ones_like(self.adj._values())
		shape = self.adj.shape
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()
	
	def get_ego_embeds(self):
		return t.concat([self.user_embeds, self.item_embeds], axis=0)
	
	def sample_subgraphs(self):
		return self.sampler(self.all_one_adj, self.get_ego_embeds())
	
	def mask_subgraphs(self, seeds):
		return self.masker(self.adj, seeds)
	
	def forward(self, encoder_adj, decoder_adj=None):
		embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
		embedsLst = [embeds]
		for i, gcn in enumerate(self.gcnLayers):
			embeds = gcn(encoder_adj, embedsLst[-1])
			embedsLst.append(embeds)
		if decoder_adj is not None:
			for gt in self.gtLayers:
				embeds = gt(decoder_adj, embedsLst[-1])
				embedsLst.append(embeds)
		embeds = sum(embedsLst)
		return embeds[:self.user_num], embeds[self.user_num:]

	def contrast(self, nodes, allEmbeds, allEmbeds2=None):
		if allEmbeds2 is not None:
			pckEmbeds = allEmbeds[nodes]
			scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
		else:
			uniqNodes = t.unique(nodes)
			pckEmbeds = allEmbeds[uniqNodes]
			scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
		return scores
	
	def cal_loss(self, batch_data, encoder_adj, decoder_adj):
		user_embeds, item_embeds = self.forward(encoder_adj, decoder_adj)
		ancs, poss, _ = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		rec_loss = (-t.sum(anc_embeds * pos_embeds, dim=-1)).mean()
		reg_loss = reg_params(self) * self.reg_weight
		cl_loss = (self.contrast(ancs, user_embeds) + self.contrast(poss, item_embeds)) * self.ssl_reg + self.contrast(ancs, user_embeds, item_embeds)
		loss = rec_loss + reg_loss + cl_loss
		losses = {'rec_loss': rec_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
		return loss, losses
	
	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward(self.adj, self.adj)
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds)

class GTLayer(nn.Module):
	def __init__(self):
		super(GTLayer, self).__init__()

		self.head_num = configs['model']['head_num']
		self.embedding_size = configs['model']['embedding_size']

		self.qTrans = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
		self.kTrans = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
		self.vTrans = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
	
	def forward(self, adj, embeds):
		indices = adj._indices()
		rows, cols = indices[0, :], indices[1, :]
		rowEmbeds = embeds[rows]
		colEmbeds = embeds[cols]

		qEmbeds = (rowEmbeds @ self.qTrans).view([-1, self.head_num, self.embedding_size // self.head_num])
		kEmbeds = (colEmbeds @ self.kTrans).view([-1, self.head_num, self.embedding_size // self.head_num])
		vEmbeds = (colEmbeds @ self.vTrans).view([-1, self.head_num, self.embedding_size // self.head_num])
		
		att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
		att = t.clamp(att, -10.0, 10.0)
		expAtt = t.exp(att)
		tem = t.zeros([adj.shape[0], self.head_num]).cuda()
		attNorm = (tem.index_add_(0, rows, expAtt))[rows]
		att = expAtt / (attNorm + 1e-8) # eh
		
		resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, self.embedding_size])
		tem = t.zeros([adj.shape[0], self.embedding_size]).cuda()
		resEmbeds = tem.index_add_(0, rows, resEmbeds) # nd
		return resEmbeds

class LocalGraph(nn.Module):
	def __init__(self):
		super(LocalGraph, self).__init__()
		self.seed_num = configs['model']['seed_num']
	
	def makeNoise(self, scores):
		noise = t.rand(scores.shape).cuda()
		noise[noise == 0] = 1e-8
		noise = -t.log(-t.log(noise))
		return t.log(scores) + noise
	
	def forward(self, allOneAdj, embeds):
		# allOneAdj should be without self-loop
		# embeds should be zero-order embeds
		order = t.sparse.sum(allOneAdj, dim=-1).to_dense().view([-1, 1])
		fstEmbeds = t.spmm(allOneAdj, embeds) - embeds
		fstNum = order
		scdEmbeds = (t.spmm(allOneAdj, fstEmbeds) - fstEmbeds) - order * embeds
		scdNum = (t.spmm(allOneAdj, fstNum) - fstNum) - order
		subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)
		subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
		embeds = F.normalize(embeds, p=2)
		scores = t.sigmoid(t.sum(subgraphEmbeds * embeds, dim=-1))
		scores = self.makeNoise(scores)
		_, seeds = t.topk(scores, self.seed_num)
		return scores, seeds

class RandomMaskSubgraphs(nn.Module):
	def __init__(self):
		super(RandomMaskSubgraphs, self).__init__()
		self.flag = False
		self.mask_depth = configs['model']['mask_depth']
		self.keep_rate = configs['model']['keep_rate']
		self.user_num = configs['data']['user_num']
		self.item_num = configs['data']['item_num']
	
	def normalizeAdj(self, adj):
		degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
		newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
		rowNorm, colNorm = degree[newRows], degree[newCols]
		newVals = adj._values() * rowNorm * colNorm
		return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

	def forward(self, adj, seeds):
		rows = adj._indices()[0, :]
		cols = adj._indices()[1, :]

		maskNodes = [seeds]

		for i in range(self.mask_depth):
			curSeeds = seeds if i == 0 else nxtSeeds
			nxtSeeds = list()
			for seed in curSeeds:
				rowIdct = (rows == seed)
				colIdct = (cols == seed)
				idct = t.logical_or(rowIdct, colIdct)

				if i != self.mask_depth - 1:
					mskRows = rows[idct]
					mskCols = cols[idct]
					nxtSeeds.append(mskRows)
					nxtSeeds.append(mskCols)

				rows = rows[t.logical_not(idct)]
				cols = cols[t.logical_not(idct)]
			if len(nxtSeeds) > 0:
				nxtSeeds = t.unique(t.concat(nxtSeeds))
				maskNodes.append(nxtSeeds)
		sampNum = int((self.user_num + self.item_num) * self.keep_rate)
		sampedNodes = t.randint(self.user_num + self.item_num, size=[sampNum]).cuda()
		if self.flag == False:
			l1 = adj._values().shape[0]
			l2 = rows.shape[0]
			print('-----')
			print('LENGTH CHANGE', '%.2f' % (l2 / l1), l2, l1)
			tem = t.unique(t.concat(maskNodes))
			print('Original SAMPLED NODES', '%.2f' % (tem.shape[0] / (self.user_num + self.item_num)), tem.shape[0], (self.user_num + self.item_num))
		maskNodes.append(sampedNodes)
		maskNodes = t.unique(t.concat(maskNodes))
		if self.flag == False:
			print('AUGMENTED SAMPLED NODES', '%.2f' % (maskNodes.shape[0] / (self.user_num + self.item_num)), maskNodes.shape[0], (self.user_num + self.item_num))
			self.flag = True
			print('-----')

		
		encoder_adj = self.normalizeAdj(t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows).cuda(), adj.shape))

		temNum = maskNodes.shape[0]
		temRows = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]
		temCols = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]

		newRows = t.concat([temRows, temCols, t.arange(self.user_num+self.item_num).cuda(), rows])
		newCols = t.concat([temCols, temRows, t.arange(self.user_num+self.item_num).cuda(), cols])

		# filter duplicated
		hashVal = newRows * (self.user_num + self.item_num) + newCols
		hashVal = t.unique(hashVal)
		newCols = hashVal % (self.user_num + self.item_num)
		newRows = ((hashVal - newCols) / (self.user_num + self.item_num)).long()


		decoder_adj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.ones_like(newRows).cuda().float(), adj.shape)
		return encoder_adj, decoder_adj
