from data_utils import *
from config.configurator import configs

def build_data_handler():
    if configs['data']['type'] == 'general_cf':
        data_handler = DataHandlerGeneralCF()
    elif configs['data']['type'] == 'sequential':
        data_handler = DataHandlerSequential()
    elif configs['model']['name'] == 'cml':
        data_handler = DataHandlerCML()
    elif configs['data']['type'] == 'social':
        data_handler = DataHandlerSocial()
    elif configs['model']['name'] == 'mmclr':
        data_handler = DataHandlerMMCLR()
    elif configs['model']['name'] == 'hmgcr' or 'smbrec':
        data_handler = DataHandlerHMGCRSMBRec()
    else:
        raise NotImplementedError

    return data_handler


