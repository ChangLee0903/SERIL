import numpy as np
from pesq import pesq
from pypesq import pesq as pypesq
from pystoi import stoi
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from util import get_dataloader
import os

warnings.filterwarnings("ignore")
OOM_RETRY_LIMIT = 10

def sisdr_eval(src, tar, sr=16000, eps=1e-10):
    alpha = (src * tar).sum() / ((tar * tar).sum() + eps)
    ay = alpha * tar
    norm = ((ay - src) * (ay - src)).sum() + eps
    sisdr = 10 * ((ay * ay).sum() / norm + eps).log10()
    return sisdr.item()


def pesq_nb_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    if np.allclose(src.sum(), 0.0, atol=1e-10) or np.allclose(tar.sum(), 0.0, atol=1e-10):
        print(
            f'[Evaluation] wav values too small: src {src.sum()}, tar {tar.sum()}')
    mos_lqo = pesq(sr, tar, src/np.abs(src).max(), 'nb')
    return mos_lqo


def pesq_wb_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    if np.allclose(src.sum(), 0.0, atol=1e-10) or np.allclose(tar.sum(), 0.0, atol=1e-10):
        print(
            f'[Evaluation] wav values too small: src {src.sum()}, tar {tar.sum()}')
    mos_lqo = pesq(sr, tar, src, 'wb')
    return mos_lqo


def pypesq_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    assert not np.allclose(
        src.sum(), 0.0, atol=1e-6) and not np.allclose(tar.sum(), 0.0, atol=1e-6)
    raw_pesq = pypesq(tar, src, sr)
    return raw_pesq


def stoi_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=False)


def estoi_eval(src, tar, sr=16000):
    src, tar = src.numpy(), tar.numpy()
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=True)


def evaluate(args, config, dataloader, model):
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    model.eval()

    loss_sum = 0
    oom_counter = 0
    n_sample = 0
    metrics = [eval(f'{m}_eval') for m in config['eval']['metrics']]
    scores_sum = torch.zeros(len(metrics))

    for (lengths, niy_audio, cln_audio) in tqdm(dataloader, desc="Iteration"):
        with torch.no_grad():
            try:
                lengths, niy_audio, cln_audio = lengths.to(
                    device), niy_audio.to(device), cln_audio.to(device)

                # split batch into list of utterances and duplicate N_METRICS times
                enh_audio = model.infer(niy_audio)
                batch_size = len(enh_audio)
                enh_audio_list = enh_audio.detach().cpu().chunk(batch_size) * len(metrics)
                cln_audio_list = cln_audio.detach().cpu().chunk(batch_size) * len(metrics)
                lengths_list = lengths.detach().cpu().tolist() * len(metrics)

                # compute loss
                loss = model(lengths, niy_audio, cln_audio).item()
                loss_sum += loss * batch_size

                # prepare metric function for each utterance in the duplicated list
                ones = torch.ones(batch_size).long().unsqueeze(
                    0).expand(len(metrics), -1)
                metric_ids = ones * \
                    torch.arange(len(metrics)).unsqueeze(-1)
                metric_fns = [metrics[idx.item()]
                              for idx in metric_ids.reshape(-1)]

                def calculate_metric(length, predicted, target, metric_fn):
                    return metric_fn(predicted.squeeze()[:length], target.squeeze()[:length])

                scores = Parallel(n_jobs=args.n_jobs)(delayed(calculate_metric)(
                    l, p, t, f) for l, p, t, f in zip(lengths_list, enh_audio_list, cln_audio_list, metric_fns))
                scores = torch.FloatTensor(scores).view(
                    len(metrics), batch_size).sum(dim=1)
                scores_sum += scores
                n_sample += batch_size

            except RuntimeError as e:
                if not 'CUDA out of memory' in str(e):
                    raise
                if oom_counter >= OOM_RETRY_LIMIT:
                    oom_counter = 0
                    break
                oom_counter += 1
                torch.cuda.empty_cache()

    loss_avg = loss_sum / n_sample
    scores_avg = scores_sum / n_sample

    model.train()
    torch.cuda.empty_cache()
    return loss_avg, scores_avg


def test(args, config):
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = f'{args.logdir}/{args.mode}'
    assert os.path.exists(save_dir)
    task_num = len(config['dataset']['test']['noisy'])

    model_list = []
    
    pretrain_model = torch.load(
        f'{args.logdir}/pretrain/{args.model}_model_T0.pth')
    for i in range(1, task_num):
        model_list.append(torch.load(
            f'{save_dir}/{args.model}_model_T{i}.pth'))
    
    if len(model_list) > 0:
        results = [np.zeros((task_num, task_num))
                            for m in config['eval']['metrics']]
    
    for Ti in range(task_num):
        test_loader = get_dataloader(args.n_jobs, config['dataset']['test']['noisy'][Ti],
                                     config['dataset']['test']['clean'][Ti], config['eval']['batch_size'])

        _, score = evaluate(args, config, test_loader,
                            pretrain_model.to(device))
        for i in range(len(score)):
            results[i][0, Ti] = score[i]
            
        for Mi in range(len(model_list)):
            _, score = evaluate(args, config, test_loader,
                                model_list[Mi].to(device))
            for i in range(len(score)):
                results[i][Mi+1, Ti] = score[i]

    for i in range(len(config['eval']['metrics'])):
        metric = config['eval']['metrics'][i]
        np.savetxt(f'{args.logdir}/{metric}.csv', results[i], delimiter=",", fmt='%1.4f')