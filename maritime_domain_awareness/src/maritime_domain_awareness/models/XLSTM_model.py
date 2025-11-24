# xLSTM: Extended Long Short-Term Memory
# https://arxiv.org/abs/2405.04517

import torch
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

class XLSTMModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2, dropout: float = 0.1, batch_first: bool = True):
        super(XLSTMModel, self).__init__()
        
        config = xLSTMLargeConfig(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first
        )
        
        self.xlstm = xLSTMLarge(config)
        self.batch_first = batch_first

    def forward(self, x):
        return self.xlstm(x)

