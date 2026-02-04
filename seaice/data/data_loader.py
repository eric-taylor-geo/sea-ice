import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def create_data_loader(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
):

    ice_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train / 10))
    ice_test = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test / 10))
    train_loader = DataLoader(ice_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(ice_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, test_loader
