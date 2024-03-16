import gc
import os
import time
import copy
import torch
import random
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from numpy import random
from copy import deepcopy
import torch.optim as optim
from trainer.metrics import Metric
from config.configurator import configs
from models.bulid_model import build_model
from torch.utils.tensorboard import SummaryWriter
from .utils import DisabledSummaryWriter, log_exceptions

if 'tensorboard' in configs['train'] and configs['train']['tensorboard']:
    writer = SummaryWriter(log_dir='runs')
else:
    writer = DisabledSummaryWriter()


def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(
            ), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

    @log_exceptions
    def train(self, model):
        self.create_optimizer(model)
        train_config = configs['train']

        if not train_config['early_stop']:
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    self.evaluate(model, epoch_idx)
            self.test(model)
            self.save_model(model)
            return model

        elif train_config['early_stop']:
            now_patience = 0
            best_epoch = 0
            best_metric = -1e9
            best_state_dict = None
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    eval_result = self.evaluate(model, epoch_idx)

                    if eval_result[configs['test']['metrics'][0]][0] > best_metric:
                        now_patience = 0
                        best_epoch = epoch_idx
                        best_metric = eval_result[configs['test']['metrics'][0]][0]
                        best_state_dict = deepcopy(model.state_dict())
                        self.logger.log("Validation score increased.  Copying the best model ...")
                    else:
                        now_patience += 1
                        self.logger.log(f"Early stop counter: {now_patience} out of {configs['train']['patience']}")

                    # early stop
                    if now_patience == configs['train']['patience']:
                        break

            # re-initialize the model and load the best parameter
            self.logger.log("Best Epoch {}".format(best_epoch))
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.evaluate(model)
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.test(model)
            self.save_model(model)
            return model

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        model.eval()
        if hasattr(self.data_handler, 'valid_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)
        elif hasattr(self.data_handler, 'test_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx)
        else:
            raise NotImplemented
        return eval_result

    @log_exceptions
    def test(self, model):
        model.eval()
        if hasattr(self.data_handler, 'test_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        else:
            raise NotImplemented
        return eval_result

    def save_model(self, model):
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            data_name = configs['data']['name']
            if not configs['tune']['enable']:
                save_dir_path = './checkpoint/{}'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                timestamp = int(time.time())
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, timestamp))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, timestamp)))
            else:
                save_dir_path = './checkpoint/{}/tune'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                now_para_str = configs['tune']['now_para_str']
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, now_para_str))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str)))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log(
                "Load model parameters from {}".format(pretrain_path))
            return model
        else:
            raise KeyError("No pretrain_path in configs['train']")

"""
Special Trainer for General Collaborative Filtering methods (AutoCF, GFormer, ...)
"""
class AutoCFTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(AutoCFTrainer, self).__init__(data_handler, logger)
        self.fix_steps = configs['model']['fix_steps']

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for i, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))

            if i % self.fix_steps == 0:
                sampScores, seeds = model.sample_subgraphs()
                encoderAdj, decoderAdj = model.mask_subgraphs(seeds)

            loss, loss_dict = model.cal_loss(batch_data, encoderAdj, decoderAdj)

            if i % self.fix_steps == 0:
                localGlobalLoss = -sampScores.mean()
                loss += localGlobalLoss
                loss_dict['infomax_loss'] = localGlobalLoss

            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)


class GFormerTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(GFormerTrainer, self).__init__(data_handler, logger)
        self.handler = data_handler
        self.user = configs['data']['user_num']
        self.item = configs['data']['item_num']
        self.latdim = configs['model']['embedding_size']
        self.fixSteps = configs['model']['fix_steps']

    def train_epoch(self, model, epoch_idx):
        """ train in train mode """
        model.train()
        """ train Rec """
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        model.preSelect_anchor_set()
        # for recording loss
        loss_log_dict = {}
        # start this epoch
        for i, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            if i % self.fixSteps == 0:
                att_edge, add_adj = model.localGraph(self.handler.torch_adj, model.getEgoEmbeds(), self.handler)
                encoderAdj, decoderAdj, sub, cmp = model.masker(add_adj, att_edge)

            loss, loss_dict = model.cal_loss(batch_data, encoderAdj, decoderAdj, sub, cmp)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

"""
Special Trainer for Sequential Recommendation methods (ICLRec, MAERec, ...)
"""
class ICLRecTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(ICLRecTrainer, self).__init__(data_handler, logger)
        self.cluster_dataloader = copy.deepcopy(self.data_handler.train_dataloader)

    def train_epoch(self, model, epoch_idx):
        """ prepare clustering in eval mode """
        model.eval()
        kmeans_training_data = []
        cluster_dataloader = self.cluster_dataloader
        cluster_dataloader.dataset.sample_negs()
        for _, tem in tqdm(enumerate(cluster_dataloader), desc='Training Clustering', total=len(cluster_dataloader)):
            batch_data = list(
                map(lambda x: x.long().to(configs['device']), tem))
            # feed batch_seqs into model.forward()
            sequence_output = model(batch_data[1], return_mean=True)
            kmeans_training_data.append(sequence_output.detach().cpu().numpy())
        kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
        model.cluster.train(kmeans_training_data)
        del kmeans_training_data
        gc.collect()

        """ train in train mode """
        model.train()
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        # for recording loss
        loss_log_dict = {}
        # start this epoch
        model.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(
                map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

class MAERecTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(MAERecTrainer, self).__init__(data_handler, logger)
        self.logger = logger

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(
                [{"params": model.encoder.parameters()},
                {"params": model.decoder.parameters()},
                {"params": model.emb_layer.parameters()},
                {"params": model.transformer_layers.parameters()}],
                lr=optim_config['lr'], weight_decay=optim_config['weight_decay']
            )

    def calc_reward(self, lastLosses, eps):
        if len(lastLosses) < 3:
            return 1.0
        curDecrease = lastLosses[-2] - lastLosses[-1]
        avgDecrease = 0
        for i in range(len(lastLosses) - 2):
            avgDecrease += lastLosses[i] - lastLosses[i + 1]
        avgDecrease /= len(lastLosses) - 2
        return 1 if curDecrease > avgDecrease else eps

    def sample_pos_edges(self, masked_edges):
        return masked_edges[torch.randperm(masked_edges.shape[0])[:configs['model']['con_batch']]]

    def sample_neg_edges(self, pos, dok):
        neg = []
        for u, v in pos:
            cu_neg = []
            num_samp = configs['model']['num_reco_neg'] // 2
            for i in range(num_samp):
                while True:
                    v_neg = np.random.randint(1, configs['data']['item_num'] + 1)
                    if (u, v_neg) not in dok:
                        break
                cu_neg.append([u, v_neg])
            for i in range(num_samp):
                while True:
                    u_neg = np.random.randint(1, configs['data']['item_num'] + 1)
                    if (u_neg, v) not in dok:
                        break
                cu_neg.append([u_neg, v])
            neg.append(cu_neg)
        return torch.Tensor(neg).long()

    def train_epoch(self, model, epoch_idx):
        model.train()

        loss_his = []
        loss_log_dict = {'loss': 0, 'loss_main': 0, 'loss_reco': 0, 'loss_regu': 0, 'loss_mask': 0}
        trn_loader = self.data_handler.train_dataloader
        trn_loader.dataset.sample_negs()

        for i, batch_data in tqdm(enumerate(trn_loader), desc='Training MAERec', total=len(trn_loader)):
            if i % configs['model']['mask_steps'] == 0:
                sample_scr, candidates = model.sampler(model.ii_adj_all_one, model.encoder.get_ego_embeds())
                masked_adj, masked_edg = model.masker(model.ii_adj, candidates)

            batch_data = list(map(lambda x: x.long().to(configs['device']), batch_data))

            item_emb, item_emb_his = model.encoder(masked_adj)
            pos = self.sample_pos_edges(masked_edg)
            neg = self.sample_neg_edges(pos, model.ii_dok)

            loss, loss_main, loss_reco, loss_regu = model.cal_loss(batch_data, item_emb, item_emb_his, pos, neg)
            loss_his.append(loss_main)

            if i % configs['model']['mask_steps'] == 0:
                reward = self.calc_reward(loss_his, configs['model']['eps'])
                loss_mask = -sample_scr.mean() * reward
                loss_log_dict['loss_mask'] += loss_mask / (len(trn_loader) // configs['model']['mask_steps'])
                loss_his = loss_his[-1:]
                loss += loss_mask

            loss_log_dict['loss'] += loss.item() / len(trn_loader)
            loss_log_dict['loss_main'] += loss_main.item() / len(trn_loader)
            loss_log_dict['loss_reco'] += loss_reco.item() / len(trn_loader)
            loss_log_dict['loss_regu'] += loss_regu.item() / len(trn_loader)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        writer.add_scalar('Loss/train', loss_log_dict['loss'], epoch_idx)

        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)


"""
Special Trainer for Social Recommendation methods (DSL, ...)
"""
class DSLTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(DSLTrainer, self).__init__(data_handler, logger)

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(
                map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val
        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)


"""
Special Trainer for Knowledge Graph-enhanced Recommendation methods (KGCL, ...)
"""
class KGCLTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(KGCLTrainer, self).__init__(data_handler, logger)
        self.train_trans = configs['model']['train_trans']
        # if self.train_trans:
        #     self.triplet_dataloader = data_handler.triplet_dataloader

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
            self.kgtrans_optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def train_epoch(self, model, epoch_idx):
        """ train in train mode """
        model.train()
        """ train Rec """
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        # for recording loss
        loss_log_dict = {}
        # start this epoch
        kg_view_1, kg_view_2, ui_view_1, ui_view_2 = model.get_aug_views()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            batch_data = list(
                map(lambda x: x.long().to(configs['device']), tem))
            batch_data.extend([kg_view_1, kg_view_2, ui_view_1, ui_view_2])
            loss, loss_dict = model.cal_loss(batch_data)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        if self.train_trans:
            """ train KG trans """
            n_kg_batch = configs['data']['triplet_num'] // configs['train']['kg_batch_size']
            for iter in tqdm(range(n_kg_batch), desc='Training KG Trans', total=n_kg_batch):
                batch_data = self.data_handler.generate_kg_batch()
                batch_data = list(map(lambda x: x.long().to(configs['device']), batch_data))
                # feed batch_seqs into model.forward()
                kg_loss = model.cal_kg_loss(batch_data)

                self.kgtrans_optimizer.zero_grad(set_to_none=True)
                kg_loss.backward()
                self.kgtrans_optimizer.step()

                if 'kg_loss' not in loss_log_dict:
                    loss_log_dict['kg_loss'] = float(kg_loss) / n_kg_batch
                else:
                    loss_log_dict['kg_loss'] += float(kg_loss) / n_kg_batch

        # if self.train_trans:
        #     """ train KG trans """
        #     triplet_dataloader = self.triplet_dataloader
        #     for _, tem in tqdm(enumerate(triplet_dataloader), desc='Training KG Trans', total=len(triplet_dataloader)):
        #         batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
        #         # feed batch_seqs into model.forward()
        #         kg_loss = model.cal_kg_loss(batch_data)

        #         self.kgtrans_optimizer.zero_grad(set_to_none=True)
        #         kg_loss.backward()
        #         self.kgtrans_optimizer.step()

        #         if 'kg_loss' not in loss_log_dict:
        #             loss_log_dict['kg_loss'] = float(kg_loss) / len(triplet_dataloader)
        #         else:
        #             loss_log_dict['kg_loss'] += float(kg_loss) / len(triplet_dataloader)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)


"""
Special Trainer for Multi-behavior Recommendation methods (CML, KMCLR, MBGMN, ...)
"""
class CMLTrainer(Trainer):
    def __init__(self, data_handler, logger):
        from models.multi_behavior.cml import MetaWeightNet
        super(CMLTrainer, self).__init__(data_handler, logger)
        self.meta_weight_net = MetaWeightNet(len(self.data_handler.behaviors)).cuda()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

        self.opt = torch.optim.AdamW(model.parameters(), lr=configs['optimizer']['lr'], weight_decay=configs['optimizer']['opt_weight_decay'])
        self.meta_opt = torch.optim.AdamW(self.meta_weight_net.parameters(), lr=configs['optimizer']['meta_lr'], weight_decay=configs['optimizer']['meta_opt_weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.opt, configs['optimizer']['opt_base_lr'],
                                                           configs['optimizer']['opt_max_lr'], step_size_up=5,
                                                           step_size_down=10, mode='triangular', gamma=0.99,
                                                           scale_fn=None, scale_mode='cycle', cycle_momentum=False,
                                                           base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        self.meta_scheduler = torch.optim.lr_scheduler.CyclicLR(self.meta_opt, configs['optimizer']['meta_opt_base_lr'],
                                                                configs['optimizer']['meta_opt_max_lr'], step_size_up=2,
                                                                step_size_down=3, mode='triangular', gamma=0.98,
                                                                scale_fn=None, scale_mode='cycle', cycle_momentum=False,
                                                                base_momentum=0.9, max_momentum=0.99, last_epoch=-1)

    def train_epoch(self, model, epoch_idx):
        from models.multi_behavior.cml import CML
        train_loader = self.data_handler.train_dataloader
        train_loader.dataset.ng_sample()

        epoch_loss = 0

        # prepare
        self.behavior_loss_list = [None] * len(self.data_handler.behaviors)
        self.user_id_list = [None] * len(self.data_handler.behaviors)
        self.item_id_pos_list = [None] * len(self.data_handler.behaviors)
        self.item_id_neg_list = [None] * len(self.data_handler.behaviors)
        self.meta_start_index = 0
        self.meta_end_index = self.meta_start_index + configs['train']['meta_batch']

        # epoch
        cnt = 0
        for user, item_i, item_j in tqdm(train_loader):

            user = user.long().cuda()
            self.user_step_index = user
            self.meta_user = torch.as_tensor(self.data_handler.meta_multi_single[self.meta_start_index:self.meta_end_index]).cuda()
            if self.meta_end_index == self.data_handler.meta_multi_single.shape[0]:
                self.meta_start_index = 0
            else:
                self.meta_start_index = (self.meta_start_index + configs['train']['meta_batch']) % (self.data_handler.meta_multi_single.shape[0] - 1)
            self.meta_end_index = min(self.meta_start_index + configs['train']['meta_batch'], self.data_handler.meta_multi_single.shape[0])

            # round one
            meta_behavior_loss_list = [None] * len(self.data_handler.behaviors)
            meta_user_index_list = [None] * len(self.data_handler.behaviors)
            meta_model = CML(self.data_handler).cuda()
            meta_opt = torch.optim.AdamW(meta_model.parameters(), lr=configs['optimizer']['lr'], weight_decay=configs['optimizer']['opt_weight_decay'])
            meta_model.load_state_dict(model.state_dict())
            meta_user_embed, meta_item_embed, meta_user_embeds, meta_item_embeds = meta_model()
            for index in range(len(self.data_handler.behaviors)):
                not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]
                self.user_id_list[index] = user[not_zero_index].long().cuda()
                meta_user_index_list[index] = self.user_id_list[index]
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()
                meta_userEmbed = meta_user_embed[self.user_id_list[index]]
                meta_posEmbed = meta_item_embed[self.item_id_pos_list[index]]
                meta_negEmbed = meta_item_embed[self.item_id_neg_list[index]]
                meta_pred_i, meta_pred_j = self._innerProduct(meta_userEmbed, meta_posEmbed, meta_negEmbed)
                meta_behavior_loss_list[index] = - (meta_pred_i.view(-1) - meta_pred_j.view(-1)).sigmoid().log()
            meta_infoNCELoss_list, SSL_user_step_index = self._SSL(meta_user_embeds, meta_item_embeds, meta_user_embed, meta_item_embed, self.user_step_index)
            meta_infoNCELoss_list_weights, meta_behavior_loss_list_weights = self.meta_weight_net(
                meta_infoNCELoss_list,
                meta_behavior_loss_list,
                SSL_user_step_index,
                meta_user_index_list,
                meta_user_embeds,
                meta_user_embed
            )

            for i in range(len(self.data_handler.behaviors)):
                meta_infoNCELoss_list[i] = (meta_infoNCELoss_list[i] * meta_infoNCELoss_list_weights[i]).sum()
                meta_behavior_loss_list[i] = (meta_behavior_loss_list[i] * meta_behavior_loss_list_weights[i]).sum()
            meta_bprloss = sum(meta_behavior_loss_list) / len(meta_behavior_loss_list)
            meta_infoNCELoss = sum(meta_infoNCELoss_list) / len(meta_infoNCELoss_list)
            meta_regLoss = (torch.norm(meta_userEmbed) ** 2 + torch.norm(meta_posEmbed) ** 2 + torch.norm(meta_negEmbed) ** 2)
            meta_model_loss = (meta_bprloss + configs['train']['reg'] * meta_regLoss + configs['train']['beta'] * meta_infoNCELoss) / configs['train']['batch_size']
            meta_opt.zero_grad(set_to_none=True)
            self.meta_opt.zero_grad(set_to_none=True)
            meta_model_loss.backward()
            nn.utils.clip_grad_norm_(self.meta_weight_net.parameters(), max_norm=20, norm_type=2)
            nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=20, norm_type=2)
            meta_opt.step()
            self.meta_opt.step()

            # round two
            behavior_loss_list = [None] * len(self.data_handler.behaviors)
            user_index_list = [None] * len(self.data_handler.behaviors)
            user_embed, item_embed, user_embeds, item_embeds = meta_model()
            for index in range(len(self.data_handler.behaviors)):
                user_id, item_id_pos, item_id_neg = self._sampleTrainBatch(torch.as_tensor(self.meta_user), self.data_handler.behaviors_data[i])
                user_index_list[index] = user_id.long()
                userEmbed = user_embed[user_id.long()]
                posEmbed = item_embed[item_id_pos.long()]
                negEmbed = item_embed[item_id_neg.long()]
                pred_i, pred_j = self._innerProduct(userEmbed, posEmbed, negEmbed)
                behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()
            self.infoNCELoss_list, SSL_user_step_index = self._SSL(user_embeds, item_embeds, user_embed, item_embed, self.meta_user)
            infoNCELoss_list_weights, behavior_loss_list_weights = self.meta_weight_net(
                self.infoNCELoss_list,
                behavior_loss_list,
                SSL_user_step_index,
                user_index_list,
                user_embeds,
                user_embed
            )
            for i in range(len(self.data_handler.behaviors)):
                self.infoNCELoss_list[i] = (self.infoNCELoss_list[i] * infoNCELoss_list_weights[i]).sum()
                behavior_loss_list[i] = (behavior_loss_list[i] * behavior_loss_list_weights[i]).sum()

            bprloss = sum(behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(self.infoNCELoss_list) / len(self.infoNCELoss_list)
            round_two_regLoss = (torch.norm(userEmbed) ** 2 + torch.norm(posEmbed) ** 2 + torch.norm(negEmbed) ** 2)
            meta_loss = 0.5 * (bprloss + configs['train']['reg'] * round_two_regLoss + configs['train']['beta'] * infoNCELoss) / configs['train']['batch_size']
            self.meta_opt.zero_grad()
            meta_loss.backward()
            nn.utils.clip_grad_norm_(self.meta_weight_net.parameters(), max_norm=20, norm_type=2)
            self.meta_opt.step()

            # round three
            user_embed, item_embed, user_embeds, item_embeds = model()
            for index in range(len(self.data_handler.behaviors)):
                userEmbed = user_embed[self.user_id_list[index]]
                posEmbed = item_embed[self.item_id_pos_list[index]]
                negEmbed = item_embed[self.item_id_neg_list[index]]
                pred_i, pred_j = self._innerProduct(userEmbed, posEmbed, negEmbed)
                self.behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()
            infoNCELoss_list, SSL_user_step_index = self._SSL(
                user_embeds, item_embeds, user_embed, item_embed, self.user_step_index)
            with torch.no_grad():
                infoNCELoss_list_weights, behavior_loss_list_weights = self.meta_weight_net(
                    infoNCELoss_list,
                    self.behavior_loss_list,
                    SSL_user_step_index,
                    self.user_id_list,
                    user_embeds,
                    user_embed
                )
            for i in range(len(self.data_handler.behaviors)):
                infoNCELoss_list[i] = (infoNCELoss_list[i] * infoNCELoss_list_weights[i]).sum()
                self.behavior_loss_list[i] = (self.behavior_loss_list[i] * behavior_loss_list_weights[i]).sum()
            bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(infoNCELoss_list) / len(infoNCELoss_list)
            regLoss = (torch.norm(userEmbed) ** 2 + torch.norm(posEmbed) ** 2 + torch.norm(negEmbed) ** 2)
            loss = (bprloss + configs['train']['reg'] * regLoss + configs['train']['beta'] * infoNCELoss) / configs['train']['batch_size']
            epoch_loss = epoch_loss + loss.item()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
            cnt += 1

    def _innerProduct(self, u, i, j):
        pred_i = torch.sum(torch.mul(u, i), dim=1) * configs['model']['inner_product_mult']
        pred_j = torch.sum(torch.mul(u, j), dim=1) * configs['model']['inner_product_mult']
        return pred_i, pred_j

    def _negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def _sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds.cpu()].toarray()
        batch = len(batIds)
        user_id = []
        item_id_pos = []
        item_id_neg = []

        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(configs['train']['sampNum'], len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self._negSamp(temLabel[i], sampNum, labelMat.shape[1])

            for j in range(sampNum):
                user_id.append(batIds[i].item())
                item_id_pos.append(poslocs[j].item())
                item_id_neg.append(neglocs[j])
                cur += 1

        return torch.as_tensor(np.array(user_id)).cuda(), torch.as_tensor(
            np.array(item_id_pos)).cuda(), torch.as_tensor(np.array(item_id_neg)).cuda()

    def _SSL(self, user_embeddings, item_embeddings, target_user_embeddings, target_item_embeddings, user_step_index):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(
                embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(
                embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(
                corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        def neg_sample_pair(x1, x2, τ=0.05):
            for i in range(x1.shape[0]):
                index_set = set(np.arange(x1.shape[0]))
                index_set.remove(i)
                index_set_neg = torch.as_tensor(np.array(list(index_set))).long().cuda()
                x_pos = x1[i].repeat(x1.shape[0] - 1, 1)
                x_neg = x2[index_set]
                if i == 0:
                    x_pos_all = x_pos
                    x_neg_all = x_neg
                else:
                    x_pos_all = torch.cat((x_pos_all, x_pos), 0)
                    x_neg_all = torch.cat((x_neg_all, x_neg), 0)
            x_pos_all = torch.as_tensor(x_pos_all)  # [9900, 100]
            x_neg_all = torch.as_tensor(x_neg_all)  # [9900, 100]
            return x_pos_all, x_neg_all

        def one_neg_sample_pair_index(i, step_index, embedding1, embedding2):
            index_set = set(np.array(step_index))
            index_set.remove(i.item())
            neg2_index = torch.as_tensor(np.array(list(index_set))).long().cuda()
            neg1_index = torch.ones((2,), dtype=torch.long)
            neg1_index = neg1_index.new_full((len(index_set),), i)
            neg_score_pre = torch.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze())
            return neg_score_pre

        def multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2):
            index_set = set(np.array(step_index.cpu()))
            batch_index_set = set(np.array(batch_index.cpu()))
            neg2_index_set = index_set - batch_index_set  # beh
            neg2_index = torch.as_tensor(np.array(list(neg2_index_set))).long().cuda()  # [910]
            neg2_index = torch.unsqueeze(neg2_index, 0)  # [1, 910]
            neg2_index = neg2_index.repeat(len(batch_index), 1)  # [100, 910]
            neg2_index = torch.reshape(neg2_index, (1, -1))  # [1, 91000]
            neg2_index = torch.squeeze(neg2_index)  # [91000]
            neg1_index = batch_index.long().cuda()  # [100]
            neg1_index = torch.unsqueeze(neg1_index, 1)  # [100, 1]
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))  # [100, 910]
            neg1_index = torch.reshape(neg1_index, (1, -1))  # [1, 91000]
            neg1_index = torch.squeeze(neg1_index)  # [91000]
            neg_score_pre = torch.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze().view(len(batch_index), -1), -1)
            return neg_score_pre  # [100]

        def compute(x1, x2, neg1_index=None, neg2_index=None, τ=0.05):
            if neg1_index != None:
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]
            N = x1.shape[0]
            D = x1.shape[1]
            x1 = x1
            x2 = x2
            scores = torch.exp(torch.div(torch.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1) + 1e-8))
            return scores

        def single_infoNCE_loss_simple(embedding1, embedding2):
            pos = score(embedding1, embedding2)
            neg1 = score(embedding2, row_column_shuffle(embedding1))
            one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
            return con_loss

        # use_less
        def single_infoNCE_loss(embedding1, embedding2):
            N = embedding1.shape[0]
            D = embedding1.shape[1]
            pos_score = compute(embedding1, embedding2).squeeze()
            neg_x1, neg_x2 = neg_sample_pair(embedding1, embedding2)
            neg_score = torch.sum(compute(neg_x1, neg_x2).view(N, (N - 1)), dim=1)
            con_loss = -torch.log(1e-8 + torch.div(pos_score, neg_score))
            con_loss = torch.mean(con_loss)
            return max(0, con_loss)

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index):
            N = step_index.shape[0]
            D = embedding1.shape[1]
            pos_score = compute(embedding1[step_index], embedding2[step_index]).squeeze()  #
            neg_score = torch.zeros((N,), dtype=torch.float64).cuda()  #
            steps = int(np.ceil(N / configs['train']['SSL_batch']))
            for i in range(steps):
                st = i * configs['train']['SSL_batch']
                ed = min((i + 1) * configs['train']['SSL_batch'], N)
                batch_index = step_index[st: ed]
                neg_score_pre = multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2)
                if i == 0:
                    neg_score = neg_score_pre
                else:
                    neg_score = torch.cat((neg_score, neg_score_pre), 0)
            con_loss = - torch.log(1e-8 + torch.div(pos_score, neg_score + 1e-8))  #
            assert not torch.any(torch.isnan(con_loss))
            assert not torch.any(torch.isinf(con_loss))
            return torch.where(torch.isnan(con_loss), torch.full_like(con_loss, 0 + 1e-8), con_loss)

        user_con_loss_list = []
        SSL_len = int(user_step_index.shape[0] / 10)
        user_step_index = torch.as_tensor(np.random.choice(user_step_index.cpu(), size=SSL_len, replace=False, p=None)).cuda()
        for i in range(len(self.data_handler.behaviors)):
            user_con_loss_list.append(single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index))
        return user_con_loss_list, user_step_index


class KMCLRTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(KMCLRTrainer, self).__init__(data_handler, logger)
        self.Kg_model = self.data_handler.Kg_model
        self.contrast_model = self.data_handler.contrast_model
        self.kg_optimizer = self.data_handler.kg_optimizer
        self.bpr = self.data_handler.bpr

    def train_epoch(self, model, epoch_idx):
        model.kg_init_transR(self.Kg_model.kg_dataset, self.Kg_model, self.kg_optimizer, index=0)
        model.kg_init_TATEC(self.Kg_model.kg_dataset, self.Kg_model, self.kg_optimizer, index=1)
        contrast_views = self.contrast_model.get_ui_kg_view()
        model.BPR_train_contrast(self.Kg_model.dataset, self.Kg_model, self.bpr, self.contrast_model, contrast_views, self.optimizer, neg_k=1)
        train_loader = self.data_handler.train_dataloader
        train_loader.dataset.ng_sample()

        epoch_loss = 0
        self.behavior_loss_list = [None] * len(self.data_handler.behaviors)
        self.user_id_list = [None] * len(self.data_handler.behaviors)
        self.item_id_pos_list = [None] * len(self.data_handler.behaviors)
        self.item_id_neg_list = [None] * len(self.data_handler.behaviors)

        for user, item_i, item_j in tqdm(train_loader):
            user = user.long().cuda()
            self.user_step_index = user
            mul_behavior_loss_list = [None] * len(self.data_handler.behaviors)
            mul_user_index_list = [None] * len(self.data_handler.behaviors)
            mul_user_embed, mul_item_embed, mul_user_embeds, mul_item_embeds = model()
            for index in range(len(self.data_handler.behaviors)):
                not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]
                self.user_id_list[index] = user[not_zero_index].long().cuda()
                mul_user_index_list[index] = self.user_id_list[index]
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()
                mul_userEmbed = mul_user_embed[self.user_id_list[index]]
                mul_posEmbed = mul_item_embed[self.item_id_pos_list[index]]
                mul_negEmbed = mul_item_embed[self.item_id_neg_list[index]]
                mul_pred_i, mul_pred_j = self.innerProduct(mul_userEmbed, mul_posEmbed, mul_negEmbed)
                mul_behavior_loss_list[index] = - (mul_pred_i.view(-1) - mul_pred_j.view(-1)).sigmoid().log()
            mul_infoNCELoss_list, SSL_user_step_index = self.SSL(mul_user_embeds, self.user_step_index)
            for i in range(len(self.data_handler.behaviors)):
                mul_infoNCELoss_list[i] = (mul_infoNCELoss_list[i]).sum()
                mul_behavior_loss_list[i] = (mul_behavior_loss_list[i]).sum()
            mul_bprloss = sum(mul_behavior_loss_list) / len(mul_behavior_loss_list)
            mul_infoNCELoss = sum(mul_infoNCELoss_list) / len(mul_infoNCELoss_list)
            mul_regLoss = (torch.norm(mul_userEmbed) ** 2 + torch.norm(mul_posEmbed) ** 2 + torch.norm(mul_negEmbed) ** 2)
            mul_model_loss = (mul_bprloss + configs['optimizer']['weight_decay'] * mul_regLoss + configs['model']['beta'] * mul_infoNCELoss) / configs['train']['batch_size']
            epoch_loss = epoch_loss + mul_model_loss.item()

            self.optimizer.zero_grad(set_to_none=True)
            mul_model_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            self.optimizer.step()
            user_embed, item_embed, user_embeds, item_embeds = model()
            with torch.no_grad():
                user_embed1, item_embed1 = self.Kg_model.getAll()
            user_embed = 0.9 * user_embed + 0.1 * user_embed1
            item_embed = item_embed

            for index in range(len(self.data_handler.behaviors)):
                userEmbed = user_embed[self.user_id_list[index]]
                posEmbed = item_embed[self.item_id_pos_list[index]]
                negEmbed = item_embed[self.item_id_neg_list[index]]
                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)
                self.behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()
            infoNCELoss_list, SSL_user_step_index = self.SSL(user_embeds, self.user_step_index)
            for i in range(len(self.data_handler.behaviors)):
                infoNCELoss_list[i] = (infoNCELoss_list[i]).sum()
                self.behavior_loss_list[i] = (self.behavior_loss_list[i]).sum()
            bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(infoNCELoss_list) / len(infoNCELoss_list)
            regLoss = (torch.norm(userEmbed) ** 2 + torch.norm(posEmbed) ** 2 + torch.norm(negEmbed) ** 2)
            loss = (bprloss + configs['optimizer']['weight_decay'] * regLoss + configs['model']['beta'] * infoNCELoss) / configs['train']['batch_size']
            epoch_loss = epoch_loss + loss.item()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def innerProduct(self, u, i, j):
        pred_i = torch.sum(torch.mul(u, i), dim=1) * configs['model']['inner_product_mult']
        pred_j = torch.sum(torch.mul(u, j), dim=1) * configs['model']['inner_product_mult']
        return pred_i, pred_j

    def SSL(self, user_embeddings, user_step_index):
        def multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2):
            index_set = set(np.array(step_index.cpu()))
            batch_index_set = set(np.array(batch_index.cpu()))
            neg2_index_set = index_set - batch_index_set
            neg2_index = torch.as_tensor(np.array(list(neg2_index_set))).long().cuda()
            neg2_index = torch.unsqueeze(neg2_index, 0)
            neg2_index = neg2_index.repeat(len(batch_index), 1)
            neg2_index = torch.reshape(neg2_index, (1, -1))
            neg2_index = torch.squeeze(neg2_index)
            neg1_index = batch_index.long().cuda()
            neg1_index = torch.unsqueeze(neg1_index, 1)
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))
            neg1_index = torch.reshape(neg1_index, (1, -1))
            neg1_index = torch.squeeze(neg1_index)
            neg_score_pre = torch.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze().view(len(batch_index), -1), -1)
            return neg_score_pre

        def compute(x1, x2, neg1_index=None, neg2_index=None):
            if neg1_index != None:
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]
            N = x1.shape[0]
            D = x1.shape[1]
            x1 = x1
            x2 = x2
            scores = torch.exp(torch.div(torch.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1) + 1e-8))
            return scores

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index):
            N = step_index.shape[0]
            pos_score = compute(embedding1[step_index], embedding2[step_index]).squeeze()
            neg_score = torch.zeros((N,), dtype=torch.float64).cuda()

            steps = int(np.ceil(N / configs['train']['SSL_batch']))
            for i in range(steps):
                st = i * configs['train']['SSL_batch']
                ed = min((i + 1) * configs['train']['SSL_batch'], N)
                batch_index = step_index[st: ed]
                neg_score_pre = multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2)
                if i == 0:
                    neg_score = neg_score_pre
                else:
                    neg_score = torch.cat((neg_score, neg_score_pre), 0)

            con_loss = - torch.log(1e-8 + torch.div(pos_score, neg_score + 1e-8))
            assert not torch.any(torch.isnan(con_loss))
            assert not torch.any(torch.isinf(con_loss))
            return torch.where(torch.isnan(con_loss), torch.full_like(con_loss, 0 + 1e-8), con_loss)

        user_con_loss_list = []
        SSL_len = int(user_step_index.shape[0] / 10)
        user_step_index = torch.as_tensor(np.random.choice(user_step_index.cpu(), size=SSL_len, replace=False, p=None)).cuda()
        for i in range(len(self.data_handler.behaviors)):
            user_con_loss_list.append(single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index))
        return user_con_loss_list, user_step_index

    def negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset


class MBGMNTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(MBGMNTrainer, self).__init__(data_handler, logger)

    def train_epoch(self, model, epoch_idx):
        model.train()
        num = configs['data']['user_num']
        sfIds = np.random.permutation(num)[:configs['model']['trnNum']]
        uids, iids = [0] * len(self.data_handler.behaviors), [0] * len(self.data_handler.behaviors)
        num = len(sfIds)
        steps = int(np.ceil(num / configs['train']['batch_size']))
        for i in tqdm(range(steps), desc='Training Recommender'):
            st = i * configs['train']['batch_size']
            ed = min((i + 1) * configs['train']['batch_size'], num)
            batIds = sfIds[st: ed]
            for beh in range(len(self.data_handler.behaviors)):
                uLocs, iLocs = self._sampleTrainBatch(batIds, self.data_handler.behaviors_data[beh])
                uids[beh] = uLocs
                iids[beh] = iLocs
            loss = model.cal_loss(uids, iids)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def _sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * configs['model']['sampNum']
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(configs['model']['sampNum'], len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(configs['data']['item_num'])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self._negSamp(temLabel[i], sampNum, configs['data']['item_num'])
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs


class AdaGCLTrainer(Trainer):
    def __init__(self, data_handler, logger):
        from models.general_cf.adagcl import VGAE, DenoiseNet
        super(AdaGCLTrainer, self).__init__(data_handler, logger)
        self.generator_1 = VGAE().cuda()
        self.generator_2 = DenoiseNet().cuda()

    def create_optimizer(self, model):
        self.generator_1.set_adagcl(model)
        self.generator_2.set_adagcl(model)
        model.set_denoiseNet(self.generator_2)

        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
            self.optimizer_gen_1 = optim.Adam(self.generator_1.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
            self.optimizer_gen_2 = optim.Adam(filter(lambda p: p.requires_grad, self.generator_2.parameters()), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def generator_generate(self, model):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj = deepcopy(self.data_handler.torch_adj)
        idxs = adj._indices()

        with torch.no_grad():
            view = model.vgae_generate(self.data_handler.torch_adj, idxs, adj)

        return view

    def train_epoch(self, model, epoch_idx):
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        model.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            self.optimizer_gen_1.zero_grad()
            self.optimizer_gen_2.zero_grad()

            temperature = max(0.05, configs['model']['init_temperature'] * pow(configs['model']['temperature_decay'], epoch_idx))

            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            data1 = self.generator_generate(self.generator_1)

            loss_cl, loss_dict_cl, out1, out2 = model.cal_loss_cl(batch_data, data1)
            ep_loss += loss_cl.item()
            loss_cl.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_ib, loss_dict_ib = model.cal_loss_ib(batch_data, data1, out1, out2)
            ep_loss += loss_ib.item()
            loss_ib.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_main, loss_dict_main = model.cal_loss(batch_data)
            ep_loss += loss_main.item()
            loss_main.backward()

            loss_vgae, loss_dict_vgae = self.generator_1.cal_loss_vgae(self.data_handler.torch_adj, batch_data)
            loss_denoise, loss_dict_denoise = self.generator_2.cal_loss_denoise(batch_data, temperature)
            loss_generator = loss_vgae + loss_denoise
            ep_loss += loss_generator.item()
            loss_generator.backward()

            self.optimizer.step()
            self.optimizer_gen_1.step()
            self.optimizer_gen_2.step()

            loss_dict = {**loss_dict_cl, **loss_dict_ib, **loss_dict_main, **loss_dict_vgae, **loss_dict_denoise}
             # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val
        
        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

class DiffKGTrainer(Trainer):
    def __init__(self, data_handler, logger):
        from models.kg.diffkg import GaussianDiffusion, Denoise
        super(DiffKGTrainer, self).__init__(data_handler, logger)
        self.diffusion = GaussianDiffusion(configs['model']['noise_scale'], configs['model']['noise_min'], configs['model']['noise_max'], configs['model']['steps']).cuda()
        out_dims = eval(configs['model']['dims']) + [configs['data']['entity_num']]
        in_dims = out_dims[::-1]
        self.denoise = Denoise(in_dims, out_dims, configs['model']['d_emb_size'], norm=True).cuda()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
            self.optimizer_denoise = optim.Adam(self.denoise.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
    
    def train_epoch(self, model, epoch_idx):
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()
        diffusionLoader = self.data_handler.diffusionLoader

        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        model.train()
        for _, tem in tqdm(enumerate(diffusionLoader), desc='Training Diffusion', total=len(diffusionLoader)):
            batch_data = list(map(lambda x: x.to(configs['device']), tem))

            ui_matrix = self.data_handler.ui_matrix
            iEmbeds = model.getEntityEmbeds().detach()
            uEmbeds = model.getUserEmbeds().detach()

            self.optimizer_denoise.zero_grad()
            loss_diff, loss_dict_diff = self.diffusion.cal_loss_diff(self.denoise, batch_data, ui_matrix, uEmbeds, iEmbeds)
            loss_diff.backward()
            self.optimizer_denoise.step()

        with torch.no_grad():
            denoised_edges = []
            h_list = []
            t_list = []

            for _, tem in enumerate(diffusionLoader):
                batch_data = list(map(lambda x: x.to(configs['device']), tem))
                batch_item, batch_index = batch_data
                denoised_batch = self.diffusion.p_sample(self.denoise, batch_item, configs['model']['sampling_steps'])
                top_item, indices_ = torch.topk(denoised_batch, k=configs['model']['rebuild_k'])
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        h_list.append(batch_index[i])
                        t_list.append(indices_[i][j])

            edge_set = set()
            for index in range(len(h_list)):
                edge_set.add((int(h_list[index].cpu().numpy()), int(t_list[index].cpu().numpy())))
            for index in range(len(h_list)):
                if (int(t_list[index].cpu().numpy()), int(h_list[index].cpu().numpy())) not in edge_set:
                    h_list.append(t_list[index])
                    t_list.append(h_list[index])
            
            relation_dict = self.data_handler.relation_dict
            for index in range(len(h_list)):
                try:
                    denoised_edges.append([h_list[index], t_list[index], relation_dict[int(h_list[index].cpu().numpy())][int(t_list[index].cpu().numpy())]])
                except Exception:
                    continue
            graph_tensor = torch.tensor(denoised_edges)
            index_ = graph_tensor[:, :-1]
            type_ = graph_tensor[:, -1]
            denoisedKG = (index_.t().long().cuda(), type_.long().cuda())
            model.setDenoisedKG(denoisedKG)

        with torch.no_grad():
            index_, type_ = denoisedKG
            mask = ((torch.rand(type_.shape[0]) + configs['model']['keepRate']).floor()).type(torch.bool)
            denoisedKG = (index_[:, mask], type_[mask])
            self.generatedKG = denoisedKG
        
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))

            self.optimizer.zero_grad()

            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data, denoisedKG)
            ep_loss += loss.item()
            loss.backward()

            self.optimizer.step()

            loss_dict = {**loss_dict}
            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val
        
        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

        
