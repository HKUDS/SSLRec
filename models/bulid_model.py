from models import *
from config.configurator import configs

def build_model(data_handler):
    if configs['model']['name'] == 'lightgcn':
        model = LightGCN(data_handler)
    if configs['model']['name'] == 'sgl':
        model = SGL(data_handler)

    return model