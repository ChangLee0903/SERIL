import torch
import torch.nn as nn
from functools import partial

MAX_POSITIONS_LEN = 16000 * 50

class SpecBase(nn.Module):
    def __init__(self, loss_func, preprocessor, input_size, hidden_size, num_layers, bidirectional):
        super(SpecBase, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.loss_func = loss_func
        self.preprocessor = preprocessor

        self._clamp_args = {'max': self.preprocessor._win_args['win_length'], 'min': 0}
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
    def __init__(self, loss_func, preprocessor, input_size, hidden_size, num_layers, bidirectional):
        super(LSTM, self).__init__(loss_func, preprocessor, input_size, hidden_size, num_layers, bidirectional)
        
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
    def __init__(self, loss_func, preprocessor, input_size, hidden_size, num_layers, bidirectional):
        super(IRM, self).__init__(loss_func, preprocessor, input_size, hidden_size, num_layers, bidirectional)
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
    def __init__(self, loss_func, preprocessor, input_size, hidden_size, num_layers, bidirectional):
        super(Residual, self).__init__(loss_func, preprocessor, input_size, hidden_size, num_layers, bidirectional)
        
    def transform(self, src, tar=None):
        _, src_linears, src_phases = self.preprocessor(src)
        pred_masks, _ = self.lstm(src_linears)
        pred_masks = self.scaling_layer(pred_masks)
        pred_linears = src_linears * pred_masks
        pred_linears = self.clamp(pred_linears)

        if tar is not None:
            _, tar_linears, _ = self.preprocessor(tar)
            return pred_linears, tar_linears
        else:
            return pred_linears, src_phases