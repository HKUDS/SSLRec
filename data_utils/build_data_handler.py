from data_utils import *
from config.configurator import configs

def build_data_handler():
    if configs['data']['type'] == 'general_cf':
        data_handler = DataHandlerGeneralCF()
    # if configs['data']['type'] == 'multi_behavior':
    #     data_handler = DataHandlerMultiBehavior()

    return data_handler