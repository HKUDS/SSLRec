import os
import time
import torch
import torch.optim as optim

from config.configurator import configs

print(configs)

class Trainer(object):
    def __init__(self):
        pass

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def train_epoch(self, epoch_idx):
        pass

    def train(self, model):
        train_config = configs['train']
        for epoch_idx in train_config['epoch']:
            self.train_epoch(epoch_idx)
        self.save_model(model)

    def evaluate(self):
        pass

    def save_model(self, model):
        if configs['trian']['save_model']:
            model_state_dict = model.stat_dict()
            model_name = configs['model']['name'].lower()
            save_dir_path = './checkpoint/{}'.format(model_name)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            timestamp = int(time.time())
            torch.save(model_state_dict, '{}/{}-{}.pth'.format(save_dir_path, model_name, timestamp))
            # log
        else:
            # log no save
            pass

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrian_path']
            model.load_state_dict(torch.load(pretrain_path))
            # log
