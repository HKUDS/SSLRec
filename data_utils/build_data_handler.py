from data_utils import *
from config.configurator import configs

def build_data_handler():
    if configs['data']['type'] == 'general_cf':
        data_handler = DataHandlerGeneralCF()
    elif configs['data']['type'] == 'cml':
        data_handler = DataHandlerCML()
    elif configs['data']['type'] == 'social_cf':
        data_handler = DataHandlerSocialCF()
    else:
        raise NotImplementedError

    return data_handler