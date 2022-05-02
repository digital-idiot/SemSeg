import torch
from torch.nn.functional import one_hot


def one_hot_encode_label(
        label_tensor: torch.Tensor,
        n_class: int
):
    x = label_tensor.clone().detach().to(
        dtype=torch.int64, device=label_tensor.device
    )
    # noinspection PyArgumentList
    assert (x.ndim == 4) and (
        label_tensor.size(1) == 1
    ), "Only single channel images can be encoded!"
    x = x.squeeze(1)
    x = one_hot(x, n_class)
    x = x.moveaxis(source=-1, destination=1)
    return x.contiguous()
