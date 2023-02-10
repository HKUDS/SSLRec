import os
import yaml
import argparse

def parse_configure():
    parser = argparse.ArgumentParser(description='SSLRec')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if args.model == None:
        raise Exception("Please provide the model name through --model.")
    model_name = args.model.lower()
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    with open('./config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        return configs

configs = parse_configure()
