import torch
import torch.nn as nn
import math
import numpy as np
import scipy
from scipy.signal.windows import hann as hanning
from torch import Tensor

EPS = np.finfo("float").eps
hann = torch.Tensor(scipy.hanning(258)[1:-1])
PAD_SIZE = 27


class Silence_Remover(nn.Module):
    def __init__(self, device, use_ref=False):
        super().__init__()
        self.N_FRAME = 256
        self.w = torch.Tensor(scipy.hanning(
            self.N_FRAME + 2)[1:-1]).to(device)
        self.EPS = np.finfo("float").eps
        self.use_ref = use_ref

    def forward(self, x, y, dyn_range=40, framelen=256, hop=128):
        x_frames = self.w * x.unfold(0, framelen, hop)
        y_frames = self.w * y.unfold(0, framelen, hop)

        if(self.use_ref):
            energies = 20 * \
                torch.log10(torch.norm(y_frames, p=2, dim=1) + self.EPS)
        else:
            energies = 20 * \
                torch.log10(torch.norm(x_frames, p=2, dim=1) + self.EPS)

        speech_part = (torch.max(energies) - dyn_range - energies) < 0
        silence_part = (torch.max(energies) - dyn_range - energies) >= 0

        if(silence_part.sum() != 0):
            silence = x_frames[silence_part]
            silence = silence.reshape(
                silence.shape[0]*2, math.floor(silence.shape[1]/2))
            silence = torch.cat([silence[0], torch.flatten(
                silence[::2][1:] + silence[1::2][:-1]), silence[-1]], dim=0)
        else:
            silence = torch.zeros(56)

        x_frames = x_frames[speech_part]
        y_frames = y_frames[speech_part]
        x_frames = x_frames.reshape(
            x_frames.shape[0]*2, math.floor(x_frames.shape[1]/2))
        x_speech = torch.cat([x_frames[0], torch.flatten(
            x_frames[::2][1:] + x_frames[1::2][:-1]), x_frames[-1]], dim=0)

        y_frames = y_frames.reshape(
            y_frames.shape[0]*2, math.floor(y_frames.shape[1]/2))
        y_speech = torch.cat([y_frames[0], torch.flatten(
            y_frames[::2][1:] + y_frames[1::2][:-1]), y_frames[-1]], dim=0)

        return x_speech, y_speech, silence


class Resampler(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def _lcm(self, a, b):
        return abs(a * b) // math.gcd(a, b)

    def _get_num_LR_output_samples(self, input_num_samp, samp_rate_in, samp_rate_out):
        samp_rate_in = int(samp_rate_in)
        samp_rate_out = int(samp_rate_out)

        tick_freq = self._lcm(samp_rate_in, samp_rate_out)
        ticks_per_input_period = tick_freq // samp_rate_in

        interval_length_in_ticks = input_num_samp * ticks_per_input_period
        if interval_length_in_ticks <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_rate_out

        last_output_samp = interval_length_in_ticks // ticks_per_output_period

        if last_output_samp * ticks_per_output_period == interval_length_in_ticks:
            last_output_samp -= 1

        num_output_samp = last_output_samp + 1
        return num_output_samp

    def _get_LR_indices_and_weights(self, orig_freq, new_freq, output_samples_in_unit, window_width,
                                    lowpass_cutoff, lowpass_filter_width):
        assert lowpass_cutoff < min(orig_freq, new_freq) / 2
        output_t = torch.arange(0, output_samples_in_unit,
                                dtype=torch.get_default_dtype()) / new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = torch.ceil(min_t * orig_freq)

        max_input_index = torch.floor(max_t * orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()

        j = torch.arange(max_weight_width).unsqueeze(0)
        input_index = min_input_index.unsqueeze(1) + j
        delta_t = (input_index / orig_freq) - output_t.unsqueeze(1)

        weights = torch.zeros_like(delta_t)
        inside_window_indices = delta_t.abs().lt(window_width)

        weights[inside_window_indices] = 0.5 * (1 + torch.cos(2 * math.pi * lowpass_cutoff /
                                                              lowpass_filter_width * delta_t[inside_window_indices]))

        t_eq_zero_indices = delta_t.eq(0.0)
        t_not_eq_zero_indices = ~t_eq_zero_indices

        weights[t_not_eq_zero_indices] *= torch.sin(
            2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]) / (math.pi * delta_t[t_not_eq_zero_indices])

        weights[t_eq_zero_indices] *= 2 * lowpass_cutoff

        weights /= orig_freq
        return min_input_index, weights

    def forward(self, waveform, orig_freq, new_freq, lowpass_filter_width=6):
        assert waveform.dim() == 2
        assert orig_freq > 0.0 and new_freq > 0.0

        min_freq = min(orig_freq, new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq

        base_freq = math.gcd(int(orig_freq), int(new_freq))
        input_samples_in_unit = int(orig_freq) // base_freq
        output_samples_in_unit = int(new_freq) // base_freq

        window_width = lowpass_filter_width / (2.0 * lowpass_cutoff)
        first_indices, weights = self._get_LR_indices_and_weights(orig_freq, new_freq, output_samples_in_unit,
                                                                  window_width, lowpass_cutoff, lowpass_filter_width)
        weights = weights.to(self.device)

        assert first_indices.dim() == 1
        conv_stride = input_samples_in_unit
        conv_transpose_stride = output_samples_in_unit
        num_channels, wave_len = waveform.size()
        window_size = weights.size(1)
        tot_output_samp = self._get_num_LR_output_samples(
            wave_len, orig_freq, new_freq)
        output = torch.zeros((num_channels, tot_output_samp))

        output = output.to(self.device)

        eye = torch.eye(num_channels).unsqueeze(2)

        eye = eye.to(self.device)

        for i in range(first_indices.size(0)):
            wave_to_conv = waveform
            first_index = int(first_indices[i].item())
            if first_index >= 0:
                wave_to_conv = wave_to_conv[..., first_index:]

            max_unit_index = (tot_output_samp - 1) // output_samples_in_unit
            end_index_of_last_window = max_unit_index * conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(
                0, end_index_of_last_window + 1 - current_wave_len)

            left_padding = max(0, -first_index)
            if left_padding != 0 or right_padding != 0:
                wave_to_conv = torch.nn.functional.pad(
                    wave_to_conv, (left_padding, right_padding))

            conv_wave = torch.nn.functional.conv1d(
                wave_to_conv.unsqueeze(0), weights[i].repeat(
                    num_channels, 1, 1),
                stride=conv_stride, groups=num_channels)

            dilated_conv_wave = torch.nn.functional.conv_transpose1d(
                conv_wave, eye, stride=conv_transpose_stride).squeeze(0)

            dialated_conv_wave_len = dilated_conv_wave.size(-1)
            left_padding = i
            right_padding = max(0, tot_output_samp -
                                (left_padding + dialated_conv_wave_len))
            dilated_conv_wave = torch.nn.functional.pad(
                dilated_conv_wave, (left_padding, right_padding))[..., :tot_output_samp]

            output += dilated_conv_wave

        return output


class Stoi(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.FS = 10000                          # Sampling frequency
        self.N_FRAME = 256                       # Window support
        self.NFFT = 512                          # FFT Size
        self.NUMBAND = 15                        # Number of 13 octave band

        # Center frequency of 1st octave band (Hz)
        self.MINFREQ = 150
        self.N = 30                              # N. frames for intermediate intelligibility
        self.BETA = -15.                         # Lower SDR bound
        self.DYN_RANGE = 40
        self.EPS = np.finfo("float").eps

        OBM, CF = self.thirdoct(self.FS, self.NFFT, self.NUMBAND, self.MINFREQ)
        self.OBM = torch.Tensor(OBM).to(device)
        self.w = torch.Tensor(scipy.hanning(
            self.N_FRAME + 2)[1:-1]).to(device)

        self.remove_silent_frames = Silence_Remover(device)
        self.resample_waveform = Resampler(device)

    def thirdoct(self, fs, nfft, num_bands, min_freq):
        f = np.linspace(0, fs, nfft + 1)
        f = f[:int(nfft/2) + 1]
        k = np.array(range(num_bands)).astype(float)
        cf = np.power(2. ** (1. / 3), k) * min_freq
        freq_low = min_freq * np.power(2., (2 * k - 1) / 6)
        freq_high = min_freq * np.power(2., (2 * k + 1) / 6)
        obm = np.zeros((num_bands, len(f)))  # a verifier

        for i in range(len(cf)):
            # Match 1/3 oct band freq with fft frequency bin
            f_bin = np.argmin(np.square(f - freq_low[i]))
            freq_low[i] = f[f_bin]
            fl_ii = f_bin
            f_bin = np.argmin(np.square(f - freq_high[i]))
            freq_high[i] = f[f_bin]
            fh_ii = f_bin
            # Assign to the octave band matrix
            obm[i, fl_ii:fh_ii] = 1
        return obm, cf

    def stft2mag(self, audio):
        spec = torch.stft(audio, n_fft=self.NFFT, hop_length=int(
            self.N_FRAME/2), win_length=self.N_FRAME, window=self.w, center=False)

        real_part = spec[:, :, 0]
        imag_part = spec[:, :, 1]
        return real_part**2 + imag_part**2

    def forward(self, enh, cln):
        x = cln
        y = enh

        x = self.resample_waveform(x, 16000, self.FS).squeeze()
        y = self.resample_waveform(y, 16000, self.FS).squeeze()

        x, y, _ = self.remove_silent_frames(
            x, y, self.DYN_RANGE, self.N_FRAME, int(self.N_FRAME/2))

        x_spec = self.stft2mag(x)
        y_spec = self.stft2mag(y)

        x_tob = torch.sqrt(torch.matmul(self.OBM, x_spec) + self.EPS)
        y_tob = torch.sqrt(torch.matmul(self.OBM, y_spec) + self.EPS)

        if(x_tob.shape[-1] >= self.N):
            x_segments = x_tob.unfold(1, self.N, 1).permute(1, 0, 2)
            y_segments = y_tob.unfold(1, self.N, 1).permute(1, 0, 2)

        else:
            x_segments = x_tob.unsqueeze(0)
            y_segments = y_tob.unsqueeze(0)

        normalization_consts = (
            torch.norm(x_segments, p=2, dim=2) /
            (torch.norm(y_segments, p=2, dim=2) + self.EPS)).unsqueeze(2)

        y_segments_normalized = y_segments * normalization_consts
        clip_value = 10 ** (-self.BETA / 20)

        y_primes = torch.min(y_segments_normalized,
                             x_segments * (1 + clip_value))

        # Subtract mean vectors
        y_primes = y_primes - torch.mean(y_primes, dim=2).unsqueeze(2)
        x_segments = x_segments - torch.mean(x_segments, dim=2).unsqueeze(2)

        y_primes = y_primes / \
            (torch.norm(y_primes, dim=2).unsqueeze(2) + self.EPS)
        x_segments = x_segments / \
            (torch.norm(x_segments, dim=2).unsqueeze(2) + self.EPS)

        # Find a matrix with entries summing to sum of correlations of vectors
        correlations_components = y_primes * x_segments

        # J, M as in [1], eq.6
        J = x_segments.shape[0]
        M = x_segments.shape[1]

        # Find the mean of all correlations
        d = torch.sum(correlations_components) / (J * M)
        return -d


class Estoi(Stoi):
    def __init__(self, device):
        super(ESTOI, self).__init__(device)
        self.device = device

    def row_col_normalize(self, x):
        x_normed = x + self.EPS * torch.randn(x.shape).to(self.device)
        x_normed -= torch.mean(x_normed, dim=-1).unsqueeze(2)

        x_inv = 1 / torch.norm(x_normed, dim=-1, p=2)
        x_normed = x_inv.unsqueeze(2)*x_normed

        x_normed += + self.EPS * torch.randn(x_normed.shape).to(self.device)
        x_normed -= torch.mean(x_normed, dim=1).unsqueeze(1)
        x_inv = 1 / torch.norm(x_normed, dim=1, p=2)
        x_normed = x_inv.unsqueeze(1) * x_normed
        return x_normed

    def forward(self, enh, cln):
        x = cln
        y = enh

        x = self.resample_waveform(x, 16000, self.FS).squeeze()
        y = self.resample_waveform(y, 16000, self.FS).squeeze()

        x, y, _ = self.remove_silent_frames(
            x, y, self.DYN_RANGE, self.N_FRAME, int(self.N_FRAME/2))

        x_spec = self.stft2mag(x)
        y_spec = self.stft2mag(y)

        x_tob = torch.sqrt(torch.matmul(self.OBM, x_spec) + self.EPS)
        y_tob = torch.sqrt(torch.matmul(self.OBM, y_spec) + self.EPS)

        if(x_tob.shape[-1] >= self.N):
            x_segments = x_tob.unfold(1, self.N, 1).permute(1, 0, 2)
            y_segments = y_tob.unfold(1, self.N, 1).permute(1, 0, 2)

        else:
            x_segments = x_tob.unsqueeze(0)
            y_segments = y_tob.unsqueeze(0)

        x_n = self.row_col_normalize(x_segments)
        y_n = self.row_col_normalize(y_segments)
        return -torch.sum(x_n * y_n / self.N) / x_n.shape[0]


class SI_SDR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, tar):
        src = src.reshape(-1, src.shape[1] * src.shape[2])
        tar = tar.reshape(-1, tar.shape[1] * tar.shape[2])

        alpha = torch.sum(src * tar, dim=1) / torch.sum(tar * tar, dim=1)
        ay = alpha.unsqueeze(1) * tar
        norm = torch.sum((ay - src) * (ay - src), dim=1) + EPS
        loss = -10 * torch.log10(torch.sum(ay * ay, dim=1) / norm)
        return loss.mean()
