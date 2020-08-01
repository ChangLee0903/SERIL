import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=201, hidden_size=201, num_layers=3, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size), nn.ReLU())
        self.init_weights()
        self.bidirectional = bidirectional

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'output_layer.0.weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, features, linear_tar, phase_inp):
        features, _ = self.lstm(features.transpose(1, 2))
        features = self.output_layer(features).transpose(1, 2)

    #     return torch.clamp(x_mag, min=0, max=256), y_mag
