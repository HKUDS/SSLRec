import torch as t
from torch import nn
from config.configurator import configs
from models.aug_utils import KMeansClustering
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class NCL(LightGCN):
	def __init__(self, data_handler):
		super(NCL, self).__init__(data_handler)

		self.proto_weight = configs['model']['proto_weight']
		self.struct_weight = configs['model']['struct_weight']
		self.temperature = configs['model']['temperature']
		self.layer_num = configs['model']['layer_num']
		self.high_order = configs['model']['high_order']

		self.kmeans = KMeansClustering(
			cluster_num=configs['model']['cluster_num'],
			embedding_size=configs['model']['embedding_size']
		)

	def _cluster(self):
		self.user_centroids, self.user2cluster, _ = self.kmeans(self.user_embeds.detach())
		self.item_centroids, self.item2cluster, _ = self.kmeans(self.item_embeds.detach())

	def forward(self, adj):
		if not self.is_training:
			embeds_list = self.final_embeds_list
		else:
			embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
			embeds_list = [embeds]
			iteration = max(self.layer_num, self.high_order * 2)
			for i in range(iteration):
				embeds = self._propagate(adj, embeds_list[-1])
				embeds_list.append(embeds)
		self.final_embeds_list = embeds_list
		embeds = sum(embeds_list[:self.layer_num + 1])
		return embeds, embeds_list

	def _pick_embeds(self, embeds, ancs, poss, negs):
		user_embeds, item_embeds = embeds[:self.user_num], embeds[self.user_num:]
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		return anc_embeds, pos_embeds, neg_embeds

	def _cal_struct_loss(self, context_embeds, ego_embeds, ancs, poss):
		user_embeds1, item_embeds1 = context_embeds[:self.user_num], context_embeds[self.user_num:]
		user_embeds2, item_embeds2 = ego_embeds[:self.user_num], ego_embeds[self.user_num:]
		pck_user_embeds1 = user_embeds1[ancs]
		pck_user_embeds2 = user_embeds2[ancs]
		pck_item_embeds1 = item_embeds1[poss]
		pck_item_embeds2 = item_embeds2[poss]
		return (cal_infonce_loss(pck_user_embeds1, pck_user_embeds2, user_embeds2, self.temperature) + cal_infonce_loss(pck_item_embeds1, pck_item_embeds2, item_embeds2, self.temperature)) / pck_user_embeds1.shape[0]

	def _cal_proto_loss(self, ego_embeds, ancs, poss):
		user_embeds, item_embeds = ego_embeds[:self.user_num], ego_embeds[self.user_num:]
		user_clusters = self.user2cluster[ancs]
		item_clusters = self.item2cluster[poss]
		pck_user_embeds1 = user_embeds[ancs]
		pck_user_embeds2 = self.user_centroids[user_clusters]
		pck_item_embeds1 = item_embeds[poss]
		pck_item_embeds2 = self.item_centroids[item_clusters]
		return (cal_infonce_loss(pck_user_embeds1, pck_user_embeds2, self.user_centroids, self.temperature) + cal_infonce_loss(pck_item_embeds1, pck_item_embeds2, self.item_centroids, self.temperature)) / pck_user_embeds1.shape[0]

	def cal_loss(self, batch_data):
		self.is_training = True
		ancs, poss, negs, kmeans_flags = batch_data
		if t.sum(kmeans_flags) != 0 or hasattr(self, 'user2cluster') == False:
			self._cluster()
		embeds, embeds_list = self.forward(self.adj)
		ego_embeds = embeds_list[0]
		context_embeds = embeds_list[self.high_order * 2]
		anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(embeds, ancs, poss, negs)

		struct_loss = self._cal_struct_loss(context_embeds, ego_embeds, ancs, poss) * self.struct_weight
		proto_loss = self._cal_proto_loss(ego_embeds, ancs, poss) * self.proto_weight
		bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
		reg_loss = reg_params(self) * self.reg_weight
		loss = bpr_loss + struct_loss + proto_loss + reg_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'struct_loss': struct_loss, 'proto_loss': proto_loss}
		return loss, losses

	def full_predict(self, batch_data):
		embeds, _ = self.forward(self.adj)
		self.is_training = False
		user_embeds, item_embeds = embeds[:self.user_num], embeds[self.user_num:]
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds
