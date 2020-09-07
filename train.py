import warnings
from asteroid.losses.sdr import SingleSrcNegSDR
import argparse
from model import LSTM, IRM, Residual
import torch
import yaml
import os
import shutil
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
from regularizer import LifeLongRegularizer
from preprocess import OnlinePreprocessor
from util import *
from evaluation import *
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
OOM_RETRY_LIMIT = 10


def evaluate(config, dataloader, model):
    torch.cuda.empty_cache()
    model.eval()

    loss_sum = 0
    oom_counter = 0
    metrics = [eval(f'{m}_eval') for m in config['eval']['metrics']]
    scores_sum = torch.zeros(len(metrics))

    for (lengths, niy_audio, cln_audio) in tqdm(dataloader, desc="Iteration"):
        with torch.no_grad():
            try:
                lengths, niy_audio, cln_audio = lengths.to(
                    device), niy_audio.to(device), cln_audio.to(device)

                enh_audio = model.infer(niy_audio)
                loss = model(lengths, niy_audio, cln_audio).item()
                loss_sum += loss

                # split batch into list of utterances and duplicate N_METRICS times
                batch_size = len(enh_audio)
                enh_audio_list = enh_audio.detach().cpu().chunk(batch_size) * len(metrics)
                cln_audio_list = cln_audio.detach().cpu().chunk(batch_size) * len(metrics)
                lengths_list = lengths.detach().cpu().tolist() * len(metrics)

                # prepare metric function for each utterance in the duplicated list
                ones = torch.ones(batch_size).long().unsqueeze(
                    0).expand(len(metrics), -1)
                metric_ids = ones * \
                    torch.arange(len(metrics)).unsqueeze(-1)
                metric_fns = [metrics[idx.item()]
                              for idx in metric_ids.reshape(-1)]

                def calculate_metric(length, predicted, target, metric_fn):
                    return metric_fn(predicted.squeeze()[:length], target.squeeze()[:length])
    
                scores = Parallel(n_jobs=config['eval']['n_jobs'])(delayed(calculate_metric)(l, p, t, f) for l, p, t, f in zip(lengths_list, enh_audio_list, cln_audio_list, metric_fns))
                scores = torch.FloatTensor(scores).view(
                    len(metrics), batch_size).mean(dim=1)
                scores_sum += scores

            except RuntimeError as e:
                if not 'CUDA out of memory' in str(e):
                    raise
                if oom_counter >= OOM_RETRY_LIMIT:
                    oom_counter = 0
                    break
                oom_counter += 1
                torch.cuda.empty_cache()

    n_sample = len(dataloader)
    loss_avg = loss_sum / n_sample
    scores_avg = scores_sum / n_sample

    model.train()
    torch.cuda.empty_cache()
    return loss_avg, scores_avg


def train(arg, config, logdir, model, optimizer, reg=None):

    # metrics_best = {'dev': torch.zeros(
    #     len(self.metrics)), 'test': torch.zeros(len(self.metrics))}
    log = SummaryWriter(logdir)

    for i in range(len(config['dataset']['train']['noisy'])):
        train_loader = get_dataloader(arg, config['dataset']['train'], i)
        dev_loader = get_dataloader(arg, config['dataset']['dev'], i)

        loss_sum = 0
        global_step = 1

        total_steps = int(config['train']['epochs'] * len(train_loader))
        pbar = tqdm(total=total_steps)
        pbar.n = global_step - 1

        while global_step <= total_steps:
            for (lengths, niy_audio, cln_audio) in train_loader:
                try:
                    lengths, niy_audio, cln_audio = lengths.to(
                        device), niy_audio.to(device), cln_audio.to(device)

                    # compute loss
                    loss = model(lengths, niy_audio, cln_audio)
                    if reg is not None:
                        loss += config['lambda'] * reg(model)
                    loss.backward()
                    loss_sum += loss.item()

                    # # gradient clipping
                    # paras = list(model.parameters())
                    # grad_norm = torch.nn.utils.clip_grad_norm_(
                    #     paras, config['train']['gradient_clipping'])
                   
                    # update parameters
                    # if math.isnan(grad_norm) or math.isinf(grad_norm):
                    #     print(
                    #         '[Runner] - Error : grad norm is nan/inf at step ' + str(self.global_step))
                    # else:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    with torch.no_grad():
                        enh_audio = model.infer(niy_audio[:1]).squeeze().cpu()[:lengths[0]]
                        niy_audio_ = niy_audio[:1].squeeze().cpu()[:lengths[0]]
                        cln_audio_ = cln_audio[:1].squeeze().cpu()[:lengths[0]]
                        print('stoi', stoi_eval(src=niy_audio_, tar=cln_audio_), stoi_eval(src=enh_audio, tar=cln_audio_), loss.item())

                    # log process
                    if global_step % int(config['train']['log_step']) == 0:
                        loss_avg = loss_sum / config['train']['log_step']
                        log.add_scalar('loss', loss_avg, global_step)
                        pbar.set_description('Loss %.5f' % (loss_avg))
                        loss_sum = 0

                    # evaluate and save the best
                    # if global_step != 0 and global_step % int(config['train']['eval_step']) == 0:
                    #     print(f'[Runner] - Evaluating on development set')
                    #     loss, metrics = evaluate(config, dev_loader, model)
                    #     print(loss, metrics)
                    #     split = 'dev'
                    #     log.add_scalar(f'{split}_si-sdr',
                    #                    loss.item(), global_step)
                    #     log.add_scalar(f'{split}_stoi',
                    #                    metrics[0].item(), global_step)
                    #     log.add_scalar(f'{split}_estoi',
                    #                    metrics[1].item(), global_step)
                    #     log.add_scalar(f'{split}_pesq',
                    #                    metrics[2].item(), global_step)
                    #     print(loss, metrics)
                    #         if (metrics > metrics_best[split]).sum() > 0:
                    #             metrics_best[split] = torch.max(
                    #                 metrics, metrics_best[split])
                    #             print('[Runner] - Saving new best model')
                    #             save_model(save_best=f'best_{split}')

                    #     if dev_loader['dev'] is not None:
                    #         evaluate('dev')

                except RuntimeError as e:
                    if not 'CUDA out of memory' in str(e):
                        raise
                    print('[Runner] - CUDA out of memory at step: ',
                          self.global_step)
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

                pbar.update(1)
                global_step += 1

    pbar.close()
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Argument Parser for SERIL.')
    parser.add_argument('--logdir', default='log',
                        help='Name of current experiment.')
    parser.add_argument('--n_jobs', default=0, type=int)
    parser.add_argument(
        '--do', choices=['pretrain', 'adaptation'], default='pretrain', type=str)
    parser.add_argument(
        '--model', choices=['LSTM', 'Residual', 'IRM'], default='Residual', type=str)

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

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    shutil.rmtree(args.logdir)
    os.makedirs(args.logdir)

    if config['train']['loss'] == 'si_sdr':
        loss_func = SingleSrcNegSDR("sisdr", zero_mean=False,
                                    reduction='mean')

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preprocessor = OnlinePreprocessor(feat_list=feat_list).to(device)
    model = eval(f'{args.model}(loss_func, preprocessor)').to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=float(config['train']['learning_rate']))

    if args.do == 'pretrain':
        assert len(config['dataset']['train']['clean']) == 1 and len(
            config['dataset']['train']['noisy']) == 1
        train(args, config, args.logdir, model, optimizer)

    elif args.do == 'adaptation':
        assert len(config['dataset']['train']['clean']) == len(
            config['dataset']['train']['noisy'])
        regularizer = LifeLongRegularizer(
            model, strategies=config['train']['strategies'])
        train(args, config, args.logdir, model, optimizer, regularizer)
