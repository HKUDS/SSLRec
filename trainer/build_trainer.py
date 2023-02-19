from config.configurator import configs
import importlib

def build_trainer(data_handler, logger):
    trainer_name = 'Trainer' if 'trainer' not in configs['train'] else configs['train']['trainer']
    # delete '_' in trainer name
    trainer_name = trainer_name.replace('_', '')
    trainers = importlib.import_module('trainer.trainer')
    for attr in dir(trainers):
        if attr.lower() == trainer_name.lower():
            return getattr(trainers, attr)(data_handler, logger)
    else:
        raise NotImplementedError('Trainer Class {} is not defined in {}'.format(trainer_name, 'trainer.trainer'))

# def build_trainer(data_handler, logger):
#     if 'trainer' not in configs['train']:
#         trainer = Trainer(data_handler, logger)
#     elif configs['train']['trainer'] == 'cml_trainer':
#         trainer = CMLTrainer(data_handler, logger)
#     elif configs['train']['trainer'] == 'mmclr_trainer':
#         trainer = MMCLRTrainer(data_handler, logger)
#     elif configs['train']['trainer'] == 'iclrec_trainer':
#         trainer = ICLRecTrainer(data_handler, logger)
#     else:
#         raise NotImplementedError
#     return trainer


