from models import *
from config.configurator import configs

def build_model(data_handler):
    if configs['model']['name'] == 'lightgcn':
        model = LightGCN(data_handler)
    elif configs['model']['name'] == 'sgl':
        model = SGL(data_handler)
    elif configs['model']['name'] == 'lightgcl':
        model = LightGCL(data_handler)
    elif configs['model']['name'] == 'directau':
        model = DirectAU(data_handler)
    elif configs['model']['name'] == 'simgcl':
        model = SimGCL(data_handler)
    elif configs['model']['name'] == 'cml':
        model = CML(data_handler)
    elif configs['model']['name'] == 'smin':
        model = SMIN(data_handler)
    elif configs['model']['name'] == 'kcgn':
        model = KCGN(data_handler)
    elif configs['model']['name'] == 'mmclr':
        model = MMCLR(data_handler)
    elif configs['model']['name'] == 'bert4rec':
        model = BERT4Rec(data_handler)
    elif configs['model']['name'] == 'cl4srec':
        model = CL4SRec(data_handler)
    elif configs['model']['name'] == 'hccf':
        model = HCCF(data_handler)
    elif configs['model']['name'] == 'hmgcr':
        model = HMGCR(data_handler)
    elif configs['model']['name'] == 'smbrec':
        model = SMBRec(data_handler)
    else:
        raise NotImplementedError

    return model


