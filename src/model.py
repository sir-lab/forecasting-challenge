import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class MultiChannelModel(nn.Module):
    def __init__(self, input_size, context_length, horizon):
        super(MultiChannelModel, self).__init__()
        self.context_length = context_length
        self.horizon = horizon
        self.input_size = input_size
        self.layer = nn.Linear(context_length, horizon)

    def forward(self, x):
        return self.layer(x)
