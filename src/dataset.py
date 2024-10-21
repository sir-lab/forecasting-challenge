import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_length, horizon):
        self.context_length = context_length
        self.horizon = horizon
        self.samples, self.targets = self.create_samples(data)

    def create_samples(self, data):
        # Generate instances with sliding window
        instances = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=(self.context_length + self.horizon), axis=0
        )
        samples = instances[:, :, :self.context_length]  # Historical context
        targets = instances[:, :, self.context_length:]  # Horizon target
        return samples, targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context = self.samples[idx, :, :]
        target = self.targets[idx, :, :]
        return torch.tensor(context, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
