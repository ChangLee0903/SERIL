import torch
from pypesq import pesq
from pystoi import stoi
from dataset import SpeechDataset
from joblib import Parallel, delayed


def pesq_eval(src, tar, sr=16000):
    src = src.squeeze().cpu().detach().numpy()
    tar = tar.squeeze().cpu().detach().numpy()
    assert src.ndim == 1 and tar.ndim == 1
    return pesq(tar, src, sr)


def stoi_eval(src, tar, sr=16000):
    src = src.squeeze().cpu().detach().numpy()
    tar = tar.squeeze().cpu().detach().numpy()
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=False)


def estoi_eval(src, tar, sr=16000):
    src = src.squeeze().cpu().detach().numpy()
    tar = tar.squeeze().cpu().detach().numpy()
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=True)


def eval(arg, config, model, (dev_noisy_path, dev_clean_path)):
    device = next(model.parameters()).device
    dataloader = torch.utils.data.DataLoader(SpeechDataset(dev_noisy_path, dev_clean_path, config['dataloader']['eval_batch_size'], shuffle=False)

    torch.cuda.empty_cache()
    model.eval()

    oom_counter=0
    scores_sum=torch.zeros(len(config['train']['metrics']))

    for (lengths, niy_audio, cln_audio) in tqdm(dataloader, desc="Iteration"):
        with torch.no_grad():
            try:
                niy_audio, cln_audio=niy_audio.to(device), cln_audio.to(device)
                pred_audio=model.infer(niy_audio)

                # split batch into list of utterances
                batch_size=pred_audio.shape[0]
                pred_audio=np.split(pred_audio, batch_size)
                cln_audio=np.split(cln_audio, batch_size)

                # duplicate list
                pred_audio *= len(config['train']['metrics'])
                cln_audio *= len(config['train']['metrics'])
                lengths *= len(config['train']['metrics'])

                # prepare metric function for each utterance in the duplicated list
                ones=torch.ones(batch_size).long()
                metric_ids=torch.cat(
                    [ones * i for i in range(len(self.metrics))], dim=0)
                metric_fns=[self.metrics[idx.item()] for idx in metric_ids]

                def calculate_metric(length, predicted, target, metric_fn):
                    return metric_fn(predicted[:, :length].squeeze(), target[:, :length].squeeze())
                
                scores=Parallel(n_jobs=self.args.n_jobs)(delayed(calculate_metric)(
                    l, p, t, f) for l, p, t, f in zip(lengths, wav_predicted, wav_tar, metric_fns))

                scores=torch.FloatTensor(scores).view(
                    len(self.metrics), batch_size).mean(dim=1)

                scores_sum += scores

            except RuntimeError as e:
                if not 'CUDA out of memory' in str(e):
                    raise
                if oom_counter >= OOM_RETRY_LIMIT:
                    oom_counter=0
                    break
                oom_counter += 1
                print(f'[Runner] - CUDA out of memory during testing {split} set, aborting after ' + str(
                    10 - oom_counter) + ' more tries')
                torch.cuda.empty_cache()

    n_sample=len(self.dataloader[split])
    scores_avg=scores_sum / n_sample
    print(f'[Runner] - {split} result: loss {loss_avg}, metrics {scores_avg}')

    model.train()
    torch.cuda.empty_cache()

    return loss_avg, scores_avg
