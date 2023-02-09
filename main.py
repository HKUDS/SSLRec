from trainer.trainer import *
from models.bulid_model import *
from trainer.logger import *
from data_utils.data_handler_general_cf import *

def main():
    # First Step: Create data_handler
    data_handler = DataHandler()
    data_handler.load_data()

    # Second Step: Create model
    model = build_model(data_handler).cuda()

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer = Trainer(data_handler, logger)

    # Fifth Step: training
    trainer.train(model)

    # Sixth Step: evaluate
    trainer.evaluate(model)

main()


