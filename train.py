import warnings
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
from util import get_dataloader
from evaluation import evaluate

warnings.filterwarnings("ignore")
OOM_RETRY_LIMIT = 10


def train_all(args, config, model, lifelong_agent=None, start=0):
    log = SummaryWriter(args.logdir)
    save_dir = f'{args.logdir}/model'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(start, len(config['dataset']['train']['noisy'])):
        # if os.path.isfile(f'{save_dir}/model_T{i-1}.pth'):
        #     model.load_state_dict(torch.load(f'{save_dir}/model_T{i-1}.pth'))
        #     if lifelong_agent is not None and os.path.isfile(f'{save_dir}/synapses_T{i-1}.pth'):
        #         lifelong_agent.load_state_dict(torch.load(f'{save_dir}/synapses_T{i-1}.pth'))

        train_loader = get_dataloader(args.n_jobs, config['dataset']['train']['noisy'][i],
                                      config['dataset']['train']['clean'][i], config['train']['batch_size'], True)
        dev_loader = get_dataloader(args.n_jobs, config['dataset']['dev']['noisy'][i],
                                    config['dataset']['dev']['clean'][i], config['eval']['batch_size'])

        train(args, config, log, train_loader,
              dev_loader, model, lifelong_agent)
        torch.save(model.state_dict(), f'{save_dir}/model_T{i}.pth')

        if lifelong_agent is not None:
            lifelong_agent.update_weights(model, train_loader)
            torch.save(lifelong_agent.state_dict(),
                       f'{save_dir}/synapses_T{i}.pth')

    log.close()


def train(args, config, log, train_loader, dev_loader, model, lifelong_agent=None):
    IsAdapt = lifelong_agent is not None and any(
        [lifelong_agent.regs[n].weights is not None for n in lifelong_agent.regs])
    # metrics_best = torch.zeros(len(config['eval']['metrics']))
    device = next(model.parameters()).device

    loss_sum = 0
    global_step = 1

    if IsAdapt:
        total_steps = int(config['train']['adapt_epochs'] * len(train_loader))
    else:
        total_steps = int(
            config['train']['pretrain_epochs'] * len(train_loader))

    pbar = tqdm(total=total_steps)
    pbar.n = global_step - 1

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=float(config['train']['learning_rate']))

    while global_step <= total_steps:
        for (lengths, niy_audio, cln_audio) in train_loader:
            try:
                lengths, niy_audio, cln_audio = lengths.to(
                    device), niy_audio.to(device), cln_audio.to(device)

                # compute loss
                loss = model(lengths, niy_audio, cln_audio)
                loss_sum += loss.item()
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

                # log process
                if global_step % int(config['train']['log_step']) == 0:
                    loss_avg = loss_sum / config['train']['log_step']
                    log.add_scalar('loss', loss_avg, global_step)
                    pbar.set_description('Loss %.5f' % (loss_avg))
                    loss_sum = 0

                # evaluate and save the best
                if (global_step != 0 and global_step % int(config['train']['eval_step']) == 0) or \
                        global_step == total_steps:
                    print(f'[Runner] - Evaluating on development set')
                    loss, scores = evaluate(args, config, dev_loader, model)
                    log.add_scalar('dev_loss', loss, global_step)
                    for score, metric_name in zip(scores, config['eval']['metrics']):
                        log.add_scalar(
                            f'dev_{metric_name}', score.item(), global_step)

                    # if not IsAdapt and (scores > metrics_best).sum() > 0:
                    #     metrics_best.data = torch.max(
                    #         scores, metrics_best).data
                    #     save_dir = f'{args.logdir}/model'
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     torch.save(model.dict.state_dict(), f'{save_dir}/model_T0.pth')

            except RuntimeError as e:
                if not 'CUDA out of memory' in str(e):
                    raise
                print('[Runner] - CUDA out of memory at step: ',
                      global_step)
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            pbar.update(1)
            global_step += 1

    pbar.close()
