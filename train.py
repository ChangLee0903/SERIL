import warnings
from asteroid.losses.sdr import SingleSrcNegSDR
import argparse
from dataset import SpeechDataset, collate_fn
from model import LSTM
import torch
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from regularizer import LifeLongRegularizer
from preprocess import OnlinePreprocessor

warnings.filterwarnings("ignore")
channel = 0

feat_list = [
    {'feat_type': 'complx', 'channel': channel},
    {'feat_type': 'linear', 'channel': channel},
    {'feat_type': 'phase', 'channel': channel}
]


def train(arg, config, logdir, model, optimizer, reg=None):
    train_noisy_list = config['dataset']['train']['noisy']
    train_clean_list = config['dataset']['train']['clean']

    # metrics_best = {'dev': torch.zeros(
    #     len(self.metrics)), 'test': torch.zeros(len(self.metrics))}
    log = SummaryWriter(logdir)

    for i in range(len(train_noisy_list)):
        dataloader = torch.utils.data.DataLoader(SpeechDataset(
            train_noisy_list[i], train_clean_list[i]), config['dataloader']['batch_size'], collate_fn=collate_fn, shuffle=True)

        total_steps = int(config['train']['epochs'] * len(dataloader))
        pbar = tqdm(total=total_steps)

        loss_sum = 0
        global_step = 0

        while global_step <= total_steps:
            for (lengths, niy_audio, cln_audio) in dataloader:
                try:
                    niy_audio, cln_audio = niy_audio.to(
                        device), cln_audio.to(device)

                    # compute loss
                    loss = model(lengths, niy_audio, cln_audio)
                    if reg is not None:
                        loss += config['lambda'] * reg(model)
                    loss.backward()
                    print(loss.item())
                    loss_sum += loss.item()

                    # update parameters
                    optimizer.step()
                    optimizer.zero_grad()

                    # log process
                    if global_step % int(config['log_step']) == 0:
                        loss_avg = loss_sum / config['log_step']
                        log.add_scalar('loss', loss_avg, global_step)
                        pbar.set_description('Loss %.5f' % (loss_avg))
                        loss_sum = 0

                    # evaluate and save the best
                    if global_step % int(config['eval_step']) == 0:
                        print(f'[Runner] - Evaluating on development set')

                        def evaluate(split):
                            loss, metrics = self.evaluate(split=split)
                            log.add_scalar(f'{split}_si-sdr',
                                        loss.item(), global_step)
                            log.add_scalar(f'{split}_stoi',
                                        metrics[0].item(), global_step)
                            log.add_scalar(f'{split}_estoi',
                                        metrics[1].item(), global_step)
                            log.add_scalar(f'{split}_pesq',
                                        metrics[2].item(), global_step)

                            if (metrics > metrics_best[split]).sum() > 0:
                                metrics_best[split] = torch.max(
                                    metrics, metrics_best[split])
                                print('[Runner] - Saving new best model')
                                self.save_model(save_best=f'best_{split}')

                #         if dataloader['dev'] is not None:
                #             evaluate('dev')

                except RuntimeError as e:
                    if not 'CUDA out of memory' in str(e):
                        raise
                    print('[Runner] - CUDA out of memory at step: ',
                          self.global_step)
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

                # global_step += 1

    pbar.close()
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Argument Parser for SERIL.')
    parser.add_argument('--logdir', default='log',
                        help='Name of current experiment.')
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument(
        '--do', choices=['pretrain', 'adaptation'], default='pretrain', type=str)

    # Options
    parser.add_argument('--config', default='config/config.yaml',
                        help='Path to downstream experiment config.', required=False)
    parser.add_argument('--seed', default=1337, type=int,
                        help='Random seed for reproducable results.', required=False)
    parser.add_argument('--gpu', default='0', type=int,
                        help='Assigning GPU id.')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    logdir = args.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if config['train']['loss'] == 'si_sdr':
        loss_func = SingleSrcNegSDR("sisdr", zero_mean=False,
                                    reduction='mean')

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preprocessor = OnlinePreprocessor(feat_list=feat_list).to(device)
    model = LSTM(loss_func, preprocessor).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    if args.do == 'pretrain':
        assert len(config['dataset']['train']['clean']) == 1 and len(
            config['dataset']['train']['noisy']) == 1
        train(args, config, logdir, model, optimizer)

    elif args.do == 'adaptation':
        assert len(config['dataset']['train']['clean']) == len(
            config['dataset']['train']['noisy'])
        regularizer = LifeLongRegularizer(
            model, strategies=config['train']['strategies'])
        train(args, config, logdir, model, optimizer, regularizer)
