import argparse
import yaml
import os
import shutil
import numpy as np
import random

import torch
from train import pretrain, adapt
from evaluation import test
from util import feat_list
from regularizer import LifeLongAgent
from preprocess import OnlinePreprocessor
from model import LSTM, IRM, Residual
from asteroid.losses.sdr import SingleSrcNegSDR


def main():
    parser = argparse.ArgumentParser(
        description='Argument Parser for SERIL.')
    parser.add_argument('--logdir', default='log',
                        help='Name of current experiment.')
    parser.add_argument('--n_jobs', default=2, type=int)
    parser.add_argument(
        '--do', choices=['train', 'test'], default='train', type=str)
    parser.add_argument(
        '--mode', choices=['seril', 'finetune'], default='seril', type=str)
    parser.add_argument(
        '--model', choices=['LSTM', 'Residual', 'IRM'], default='LSTM', type=str)
    
    # Options
    parser.add_argument(
        '--config', default='config/config.yaml', required=False)
    parser.add_argument('--seed', default=1126, type=int,
                        help='Random seed for reproducable results.', required=False)
    parser.add_argument('--gpu', default='2', type=int,
                        help='Assigning GPU id.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build log directory
    os.makedirs(args.logdir, exist_ok=True)

    # load configure
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if config['train']['loss'] == 'sisdr':
        loss_func = SingleSrcNegSDR("sisdr", zero_mean=False,
                                    reduction='mean')
    if args.do == 'train':
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert len(config['dataset']['train']['clean']) == len(
            config['dataset']['train']['noisy']) and len(config['dataset']['train']['clean']) >= 1
        
        model_path = f'{args.logdir}/pretrain/{args.model}_model_T0.pth'
        lifelong_agent_path = f'{args.logdir}/pretrain/{args.model}_synapses_T0.pth'

        if os.path.exists(model_path) and os.path.exists(lifelong_agent_path):
            print(f'[Runner] - pretrain model has already existed!')
            model = torch.load(model_path).to(device)
            lifelong_agent = torch.load(lifelong_agent_path).to(device)
            lifelong_agent.load_config(**config['train']['strategies'])

        else:
            print(f'[Runner] - run pretrain process!')
            preprocessor = OnlinePreprocessor(feat_list=feat_list).to(device)
            model = eval(f'{args.model}')(loss_func, preprocessor, **config['model']).to(device)
            lifelong_agent = LifeLongAgent(model, **config['train']['strategies'])
            pretrain(args, config, model, lifelong_agent)

        print(f'[Runner] - run adaptation process!')
        args.logdir = f'{args.logdir}/{args.mode}'
        if args.mode == 'seril':
            adapt(args, config, model, lifelong_agent)
        elif args.mode == 'finetune':
            adapt(args, config, model)

    if args.do == 'test':
        test(args, config)

if __name__ == "__main__":
    main()
