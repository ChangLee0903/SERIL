import torch
import os
import numpy as np
import random
from librosa import load


def read(data, normalize=False, sr=16000):
    data, sr = load(data, sr=sr)
    if normalize:
        data /= np.abs(data).max()
    return data, sr


def collate_fn(samples):
    niy_samples = [s[0] for s in samples]
    cln_samples = [s[1] for s in samples]
    lengths = [len(s[0]) for s in samples]
    niy_samples = torch.nn.utils.rnn.pad_sequence(
        niy_samples, batch_first=True)
    cln_samples = torch.nn.utils.rnn.pad_sequence(
        cln_samples, batch_first=True)
    return lengths, niy_samples.transpose(-1, -2).contiguous(), cln_samples.transpose(-1, -2).contiguous()


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, clean_path, noisy_path, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.noisy_list = [os.path.join(noisy_path, f)
                           for f in os.listdir(noisy_path)]
        self.clean_list = [os.path.join(clean_path, f)
                           for f in os.listdir(clean_path)]

        assert len(self.noisy_list) == len(self.clean_list)

    def __getitem__(self, index):
        niy_audio, sampling_rate = read(
            self.noisy_list[index], sr=self.sampling_rate)
        assert sampling_rate == self.sampling_rate

        cln_audio, sampling_rate = read(
            self.clean_list[index], sr=self.sampling_rate)
        assert sampling_rate == self.sampling_rate

        niy_audio, cln_audio = torch.from_numpy(
            niy_audio), torch.from_numpy(cln_audio)
        return niy_audio.unsqueeze(1).float(), cln_audio.unsqueeze(1).float()

    def __len__(self):
        return len(self.clean_list)
