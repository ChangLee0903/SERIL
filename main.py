import argparse
import yaml
import os
import shutil
import numpy as np
import random

import torch
from train import train_all
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
        '--do', choices=['seril', 'finetune', 'test'], default='seril', type=str)
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

    # clean log files
    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)
    os.makedirs(args.logdir)

    # load configure
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if config['train']['loss'] == 'sisdr':
        loss_func = SingleSrcNegSDR("sisdr", zero_mean=False,
                                    reduction='mean')

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preprocessor = OnlinePreprocessor(feat_list=feat_list).to(device)
    model = eval(f'{args.model}(loss_func, preprocessor)').to(device)

    if args.do == 'seril':
        assert len(config['dataset']['train']['clean']) == len(
            config['dataset']['train']['noisy']) and len(config['dataset']['train']['clean']) >= 1
        lifelong_agent = LifeLongAgent(
            model, strategies=config['train']['strategies'])
        train_all(args, config, model, lifelong_agent)

    elif args.do == 'finetune':
        assert len(config['dataset']['train']['clean']) == len(
            config['dataset']['train']['noisy']) and len(config['dataset']['train']['clean']) >= 1
        train_all(args, config, model)


if __name__ == "__main__":
    main()
