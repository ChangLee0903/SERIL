from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import argparse
import yaml
import os
import shutil
import numpy as np
import random

import torch
from util import feat_list
from regularizer import LifeLongAgent
from preprocess import OnlinePreprocessor
from model import LSTM, IRM, Residual
from asteroid.losses.sdr import SingleSrcNegSDR
import math
from util import get_dataloader
import csv


space = {
    'alpha_ewc': hp.choice('alpha_ewc', list(np.arange(0.1, 1.1, 0.1))),
    'alpha_si': hp.choice('alpha_si', list(np.arange(0.1, 1.1, 0.1))),
    'beta': hp.choice('beta', list(np.arange(0.1, 1.1, 0.1))),
    'lambda': hp.choice('lambda', [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000])
}


def train(args, config, train_loader, model, lifelong_agent=None):

    IsAdapt = lifelong_agent is not None and any(
        [lifelong_agent.regs[n].weights is not None for n in lifelong_agent.regs])
    global_step = 1
    if IsAdapt:
        total_steps = int(config['train']['adapt_epochs'] * len(train_loader))
    else:
        total_steps = int(
            config['train']['pretrain_epochs'] * len(train_loader))

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=float(config['train']['learning_rate']))

    while global_step <= total_steps:
        for (lengths, niy_audio, cln_audio) in train_loader:
            try:
                lengths, niy_audio, cln_audio = lengths.to(
                    device), niy_audio.to(device), cln_audio.to(device)

                # compute loss
                loss = model(lengths, niy_audio, cln_audio)
                if IsAdapt:
                    loss += config['train']['lambda'] * \
                        lifelong_agent(model)
                loss.backward()

                # gradient clipping
                paras = list(model.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    paras, config['train']['gradient_clipping'])

                if lifelong_agent is not None and 'si' in lifelong_agent.regs:
                    lifelong_agent.regs['si'].update_Wk(model)

                # update parameters
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    print(
                        '[Runner] - Error : grad norm is nan/inf at step ' + str(global_step))
                else:
                    optimizer.step()

                optimizer.zero_grad()

            except RuntimeError as e:
                if not 'CUDA out of memory' in str(e):
                    raise
                print('[Runner] - CUDA out of memory at step: ',
                      global_step)
                optimizer.zero_grad()
                torch.cuda.empty_cache()
        global_step += 1


def run(args, config, model, lifelong_agent=None):
    for i in range(1, len(config['dataset']['train']['noisy'])):
        train_loader = get_dataloader(args.n_jobs, config['dataset']['train']['noisy'][i],
                                      config['dataset']['train']['clean'][i], config['train']['batch_size'], True)
        train(args, config, train_loader, model, lifelong_agent)
        lifelong_agent.update_weights(model, train_loader)

    score = 0
    model.eval()
    with torch.no_grad():
        for i in range(len(config['dataset']['train']['noisy'])):
            dev_loader = get_dataloader(args.n_jobs, config['dataset']['dev']['noisy'][i],
                                        config['dataset']['dev']['clean'][i], config['eval']['batch_size'])
            loss_sum = 0
            sample_num = 0
            for (lengths, niy_audio, cln_audio) in dev_loader:
                lengths, niy_audio, cln_audio = lengths.to(
                    device), niy_audio.to(device), cln_audio.to(device)

                batch_size = len(niy_audio)
                loss = model(lengths, niy_audio, cln_audio).item()
                loss_sum += loss * batch_size
                sample_num += batch_size
            loss_sum /= sample_num

            score += loss_sum
    score -= (loss_sum * 0.05)
    return score / len(config['dataset']['train']['noisy']) 


def objective(params):
    config['train']['lambda'] = params['lambda']
    config['train']['strategies']['alpha']['ewc'] = params['alpha_ewc']
    config['train']['strategies']['alpha']['si'] = params['alpha_si']
    config['train']['strategies']['beta'] = params['beta']

    if config['train']['loss'] == 'sisdr':
        loss_func = SingleSrcNegSDR("sisdr", zero_mean=False,
                                    reduction='mean')
    model_path = f'{args.logdir}/pretrain/{args.model}_model_T0.pth'
    lifelong_agent_path = f'{args.logdir}/pretrain/{args.model}_synapses_T0.pth'

    model = torch.load(model_path)
    lifelong_agent = torch.load(lifelong_agent_path)
    lifelong_agent.load_config(**config['train']['strategies'])

    loss = run(args, config, model, lifelong_agent)

    csv_file = open(save_dir, 'a')
    writer = csv.writer(csv_file)
    writer.writerow(['{:.4f}'.format(loss)] + ['{:.2f}'.format(params[n]) for n in params])
    csv_file.close()
    
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

def main():
    global device, args, config
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

    # load configure
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    global writer, csv_file, save_dir
    save_dir = f'{args.logdir}/record.csv'
    
    csv_file = open(save_dir, 'a')
    writer = csv.writer(csv_file)
    writer.writerow(['loss'] + [n for n in space])
    csv_file.close()

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=1000, trials=trials)
    print(best)


if __name__ == "__main__":
    main()
