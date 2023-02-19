from config.configurator import configs
import importlib

def build_model(data_handler):
    model_type = configs['data']['type']
    model_name = configs['model']['name']
    module_path = ".".join(['models', model_type, model_name])
    if importlib.util.find_spec(module_path) is None:
        raise NotImplementedError('Model {} is not implemented'.format(model_name))
    module = importlib.import_module(module_path)
    for attr in dir(module):
        if attr.lower() == model_name.lower():
            return getattr(module, attr)(data_handler)
    else:
        raise NotImplementedError('Model Class {} is not defined in {}'.format(model_name, module_path))

# def build_model(data_handler):
#     if configs['model']['name'] == 'lightgcn':
#         model = LightGCN(data_handler)
#     elif configs['model']['name'] == 'sgl':
#         model = SGL(data_handler)
#     elif configs['model']['name'] == 'lightgcl':
#         model = LightGCL(data_handler)
#     elif configs['model']['name'] == 'directau':
#         model = DirectAU(data_handler)
#     elif configs['model']['name'] == 'simgcl':
#         model = SimGCL(data_handler)
#     elif configs['model']['name'] == 'cml':
#         model = CML(data_handler)
#     elif configs['model']['name'] == 'smin':
#         model = SMIN(data_handler)
#     elif configs['model']['name'] == 'kcgn':
#         model = KCGN(data_handler)
#     elif configs['model']['name'] == 'mhcn':
#         model = MHCN(data_handler)
#     elif configs['model']['name'] == 'mmclr':
#         model = MMCLR(data_handler)
#     elif configs['model']['name'] == 'bert4rec':
#         model = BERT4Rec(data_handler)
#     elif configs['model']['name'] == 'cl4srec':
#         model = CL4SRec(data_handler)
#     elif configs['model']['name'] == 'hccf':
#         model = HCCF(data_handler)
#     elif configs['model']['name'] == 'hmgcr':
#         model = HMGCR(data_handler)
#     elif configs['model']['name'] == 'smbrec':
#         model = SMBRec(data_handler)
#     elif configs['model']['name'] == 'ncl':
#         model = NCL(data_handler)
#     elif configs['model']['name'] == 'duorec':
#         model = DuoRec(data_handler)
#     elif configs['model']['name'] == 'iclrec':
#         model = ICLRec(data_handler)
#     elif configs['model']['name'] == 'dcrec_seq':
#         model = DCRec_seq(data_handler)
#     else:
#         raise NotImplementedError

#     return model


