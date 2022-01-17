from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data
import numpy as np
import torch
import os


class BasicData(data.Dataset):
    def __init__(self, x_c_np, x_p_np, x_t_np, y, extra, scaler: MinMaxScaler):
        self.x_closeness = scaler.transform(x_c_np.reshape(-1, x_c_np.shape[-1])).reshape(x_c_np.shape)
        self.x_period = scaler.transform(x_p_np.reshape(-1, x_p_np.shape[-1])).reshape(x_p_np.shape)
        self.x_trend = scaler.transform(x_t_np.reshape(-1, x_t_np.shape[-1])).reshape(x_t_np.shape)
        self.y = scaler.transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
        self.extra = torch.Tensor(extra)

        self.to_tensor()

    def to_tensor(self):
        self.y = torch.Tensor(self.y)
        self.x_closeness = torch.Tensor(self.x_closeness)
        self.x_period = torch.Tensor(self.x_period)
        self.x_trend = torch.Tensor(self.x_trend)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x_closeness[item], self.x_period[item], self.x_trend[item], self.y[item], self.extra[item]




