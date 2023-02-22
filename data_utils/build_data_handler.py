from data_utils import *
from config.configurator import configs

def build_data_handler():
    if configs['data']['type'] == 'general_cf':
        data_handler = DataHandlerGeneralCF()
    elif configs['data']['type'] == 'sequential':
        data_handler = DataHandlerSequential()
    elif configs['data']['type'] == 'multi_behavior':
        data_handler = DataHandlerMultiBehavior()
    elif configs['data']['type'] == 'social':
        data_handler = DataHandlerSocial()
    elif configs['data']['type'] == 'kg':
        data_handler = DataHandlerKG()
    else:
        raise NotImplementedError

    return data_handler


