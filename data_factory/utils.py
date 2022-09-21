import torch
import numpy as np
from typing import Sequence


def image_to_tensor(
        arr: np.ndarray,
        mu: Sequence[float] = None,
        sigma: Sequence[float] = None
):
    c, _, _ = arr.shape
    x = torch.from_numpy(arr).contiguous()
    if mu is None:
        mu = torch.tensor(data=([0] * c), dtype=x.dtype)
    if sigma is None:
        if isinstance(x, torch.ByteTensor):
            sigma = torch.tensor(data=([255] * c), dtype=x.dtype)
        else:
            sigma = torch.tensor(data=([1] * c), dtype=x.dtype)
    assert len(mu) == len(sigma) == x.shape[0], "Invalid 'mu', 'sigma'"
    mu = torch.tensor(
        data=np.expand_dims(a=np.array(mu), axis=(1, 2)),
        dtype=torch.get_default_dtype()
    )
    sigma = torch.tensor(
        data=np.expand_dims(a=np.array(sigma), axis=(1, 2)),
        dtype=torch.get_default_dtype()
    )
    assert torch.all(sigma > 0), (
        "'sigma' cannot have zero or negative values"
    )
    x = (x - mu) / sigma
    return x


def label_to_tensor(arr: np.ndarray):
    return torch.from_numpy(arr).contiguous().to(dtype=torch.int64)
