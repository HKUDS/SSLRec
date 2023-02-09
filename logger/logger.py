import os
import logging
import datetime

from config.configurator import configs

class Logger(object):
    def __init__(self, log_configs=True):
        model_name = configs['model']['name']
        log_dir_path = './log/{}'.format(model_name)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        dataset_name = configs['data']['dataset']
        log_file = logging.FileHandler('{}/{}.log'.format(log_dir_path, dataset_name), 'a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)
        if log_configs:
            self.logger.info(configs)

    def log(self, message, print_to_consle=True):
        self.logger.info(message)
        if print_to_consle:
            print(message)

    def log_loss(self, epoch_idx, loss_log_dict, print_to_consle=True):
        epoch = configs['train']['epoch']
        message = '[Epoch {:3d} / {:3d}] '.format(epoch_idx, epoch)
        for loss_name in loss_log_dict:
            message += '{}: {}'.format(loss_name, loss_log_dict[loss_name])
        self.logger.info(message)
        if print_to_consle:
            print(message)