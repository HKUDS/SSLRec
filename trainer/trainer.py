import os
import time
import torch
import torch.optim as optim

from tqdm import tqdm
from config.configurator import configs

print(configs)

class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.neg_sampling()

        # for recording loss
        loss_log_dict = {}

        # start this epoch
        model.train()
        for _, tem in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().cuda()), tem)
            loss, loss_dict = model.cal_loss(batch_data)
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_dict: loss_log_dict[loss_name] = _loss_val
                else: loss_log_dict[loss_name] += _loss_val

        # log
        self.logger.log_loss(epoch_idx, loss_log_dict)

    def train(self, model):
        train_config = configs['train']
        for epoch_idx in train_config['epoch']:
            # train
            self.train_epoch(epoch_idx)
            # evaluate
            if epoch_idx % train_config['test_step'] == 0:
                self.evaluate()

        self.save_model(model)

    def evaluate(self, model):
        model.eval()

    def save_model(self, model):
        if configs['trian']['save_model']:
            model_state_dict = model.stat_dict()
            model_name = configs['model']['name']
            save_dir_path = './checkpoint/{}'.format(model_name)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            timestamp = int(time.time())
            torch.save(model_state_dict, '{}/{}-{}.pth'.format(save_dir_path, model_name, timestamp))
            self.logger.log("Save model parameters to {}".format('{}/{}-{}.pth'.format(save_dir_path, model_name, timestamp)))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrian_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log("Load model parameters from {}".format(pretrain_path))


