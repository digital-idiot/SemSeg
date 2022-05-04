import torch
import numpy as np


def image_to_tensor(arr: np.ndarray):
    arr = torch.from_numpy(arr).contiguous()
    if isinstance(arr, torch.ByteTensor):
        return arr.to(dtype=torch.get_default_dtype()).div(255)
    else:
        return arr


def label_to_tensor(arr: np.ndarray):
    return torch.from_numpy(arr).contiguous().to(dtype=torch.int64)
