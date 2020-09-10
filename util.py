import torch
import numpy as np
from dataset import SpeechDataset

channel = 0
feat_list = [
    {'feat_type': 'complx', 'channel': channel},
    {'feat_type': 'linear', 'channel': channel},
    {'feat_type': 'phase', 'channel': channel}
]


def adnoise(speech_data, noise_data, SNR):
    noise_length = noise_data.shape[0]
    speech_length = speech_data.shape[0]

    if noise_length - speech_length <= 0:
        dup_num = np.ceil(speech_length / noise_length).astype(int)
        noise_data = np.tile(noise_data, dup_num)
        noise_length = noise_data.shape[0]

    start = np.random.randint(0, noise_length - speech_length, 1)[0]
    noise_data = noise_data[start: start + speech_length]

    SNR_exp = 10.0**(SNR / 10.0)
    speech_var = np.dot(speech_data, speech_data)
    noise_var = np.dot(noise_data, noise_data)
    scaler = np.sqrt(speech_var / (SNR_exp * noise_var))

    return speech_data + scaler * noise_data


def get_dataloader(n_jobs, noisy_list, clean_list, batch_size, shuffle=False):
    def collate_fn(samples):
        niy_samples = [s[0] for s in samples]
        cln_samples = [s[1] for s in samples]
        lengths = torch.LongTensor([len(s[0]) for s in samples])

        niy_samples = torch.nn.utils.rnn.pad_sequence(
            niy_samples, batch_first=True)
        cln_samples = torch.nn.utils.rnn.pad_sequence(
            cln_samples, batch_first=True)
        return lengths, niy_samples.transpose(-1, -2).contiguous(), cln_samples.transpose(-1, -2).contiguous()

    dataloader = torch.utils.data.DataLoader(SpeechDataset(
        noisy_list, clean_list), batch_size, collate_fn=collate_fn, num_workers=n_jobs, shuffle=shuffle)

    return dataloader
