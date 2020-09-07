import torch
import torch.nn as nn
from functools import partial

MAX_POSITIONS_LEN = 16000 * 50

class STFT(torch.nn.Module):
    def __init__(self, IsTrain=False, filter_length=512, hop_length=256, win_length=None):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        self.forward_basis = torch.nn.Parameter(torch.FloatTensor(
            fourier_basis[:, None, :]), requires_grad=False)
        self.inverse_basis = torch.nn.Parameter(torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]), requires_grad=False)

        assert(filter_length >= self.win_length)

        self.fft_window = torch.from_numpy(pad_center(get_window(
            'hann', self.win_length, fftbins=True), filter_length)).float()
        self.fft_window = torch.nn.Parameter(
            self.fft_window, requires_grad=IsTrain)

    def transform(self, input_data):
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        self.num_samples = num_samples

        input_data = input_data.view(num_batches, 1, num_samples)

        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis * self.fft_window,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[..., self.pad_amount:]
        inverse_transform = inverse_transform[..., :self.num_samples]
        inverse_transform = inverse_transform.squeeze(1)

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class SpecBase(nn.Module):
    def __init__(self, loss_func, preprocessor, input_size=257, hidden_size=257, num_layers=3, bidirectional=False):
        super(SpecBase, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.loss_func = loss_func
        self.preprocessor = preprocessor

        self._clamp_args = {'max': 2*self.preprocessor._win_args['win_length'], 'min': 0}
        self.clamp = partial(torch.clamp, **self._clamp_args)
        
        self.scaling_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size))
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'scaling_layer.0.weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def infer(self, src):
        pred_linears, src_phases = self.transform(src)
        return self.preprocessor.istft(linears=pred_linears, phases=src_phases, length=src.shape[-1])
    
    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = torch.arange(MAX_POSITIONS_LEN).to(device=lengths.device)
        ascending = ascending[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1)
        length_masks = (ascending < lengths.unsqueeze(-1)).long()
        return length_masks.unsqueeze(-1)

    def forward(self, lengths, src, tar):
        pred_linears, tar_linears = self.transform(src, tar)
        
        stft_lengths = lengths // self.preprocessor._win_args['hop_length'] + 1
        stft_length_masks = self._get_length_masks(stft_lengths)
                    
        pred_linears, tar_linears = pred_linears * stft_length_masks, tar_linears * stft_length_masks
        return self.loss_func(pred_linears.flatten(start_dim=1).contiguous(),
                              tar_linears.flatten(start_dim=1).contiguous())

class LSTM(SpecBase):
    def __init__(self, loss_func, preprocessor, input_size=257, hidden_size=257, num_layers=3, bidirectional=False):
        super(LSTM, self).__init__(loss_func, preprocessor, input_size=257, hidden_size=257, num_layers=3, bidirectional=False)
        
    def transform(self, src, tar=None):
        _, src_linears, src_phases = self.preprocessor(src)
        pred_linears, _ = self.lstm(src_linears)
        pred_linears = self.scaling_layer(pred_linears)
        pred_linears = self.clamp(pred_linears)

        if tar is not None:
            _, tar_linears, _ = self.preprocessor(tar)
            return pred_linears, tar_linears
        else:
            return pred_linears, src_phases

class IRM(SpecBase):
    def __init__(self, loss_func, preprocessor, input_size=257, hidden_size=257, num_layers=3, bidirectional=False):
        super(IRM, self).__init__(loss_func, preprocessor, input_size=257, hidden_size=257, num_layers=3, bidirectional=False)
        self.scaling_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size), nn.Sigmoid())
  
    def transform(self, src, tar=None):
        _, src_linears, src_phases = self.preprocessor(src)
        pred_masks, _ = self.lstm(src_linears)
        pred_masks = self.scaling_layer(pred_masks)
        pred_linears = src_linears * pred_masks

        if tar is not None:
            _, tar_linears, _ = self.preprocessor(tar)
            return pred_linears, tar_linears
        else:
            return pred_linears, src_phases

class Residual(SpecBase):
    def __init__(self, loss_func, preprocessor, input_size=257, hidden_size=257, num_layers=3, bidirectional=False):
        super(Residual, self).__init__(loss_func, preprocessor, input_size=257, hidden_size=257, num_layers=3, bidirectional=False)
        
    def transform(self, src, tar=None):
        _, src_linears, src_phases = self.preprocessor(src)
        src_linears = src_linears.sqrt()
        pred_masks, _ = self.lstm(src_linears)
        pred_masks = self.scaling_layer(pred_masks)
        pred_linears = src_linears + pred_masks
        pred_linears = self.clamp(pred_linears)

        if tar is not None:
            _, tar_linears, _ = self.preprocessor(tar)
            return pred_linears, tar_linears.sqrt()
        else:
            return pred_linears**2, src_phases