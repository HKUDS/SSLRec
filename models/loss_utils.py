import torch as t
import torch.nn.functional as F

def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
	pos_preds = (anc_embeds * pos_embeds).sum(-1)
	neg_preds = (anc_embeds * neg_embeds).sum(-1)
	# diff_preds = pos_preds - neg_preds
	# return - diff_preds.sigmoid().log().sum()
	return t.sum(F.softplus(neg_preds - pos_preds))

def reg_pick_embeds(embeds_list):
	reg_loss = 0
	for embeds in embeds_list:
		reg_loss += embeds.square().sum()
	return reg_loss

def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
	normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
	normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
	normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
	nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
	deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
	cl_loss = (nume_term + deno_term).sum()
	return cl_loss

def reg_params(model):
	reg_loss = 0
	for W in model.parameters():
		reg_loss += W.norm(2).square()
	return reg_loss