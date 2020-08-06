import torch
import torch.nn as nn
import math

class LSTM(nn.Module):
    def __init__(self, loss_func, preprocessor, input_size=257, hidden_size=257, num_layers=3, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.loss_func = loss_func
        self.preprocessor = preprocessor
        self.scaling_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size), nn.ReLU())
        self.init_weights()
        self.bidirectional = bidirectional

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'scaling_layer.0.weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def transform(self, src, tar=None):
        _, src_linears, src_phases = self.preprocessor(src)
        pred_linears, _ = self.lstm(src_linears)
        pred_linears = self.scaling_layer(pred_linears)
        if tar is not None:
            _, tar_linears, _ = self.preprocessor(tar)
            return pred_linears, tar_linears
        else:
            return pred_linears, src_phases

    def infer(self, src):
        pred_linears, src_phases = self.transform(src)
        return self.preprocessor.istft(linears=pred_linears, phases=src_phases, length=src.shape[-1])

    def forward(self, legal_lengths, src, tar):
        pred_linears, tar_linears = self.transform(src, tar)
        for i in range(pred_linears.shape[0]):
            end = math.ceil(legal_lengths[i] /
                      self.preprocessor._win_args['hop_length'])
            pred_linears[:, end:] = 0
            tar_linears[:, end:] = 0
        return self.loss_func(pred_linears.flatten(start_dim=1).contiguous(),
                              tar_linears.flatten(start_dim=1).contiguous())
