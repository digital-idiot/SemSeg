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


def get_shape(tensor: torch.Tensor):
    shape = tuple(tensor.shape)
    if torch.onnx.is_in_onnx_export():
        shape = tuple(s.item() for s in shape)
    return shape


def make_divisible(value, divisor: int, min_value: int = None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here: https://kutt.it/bk7IRJ
    :param value:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, ((int(value + divisor / 2) // divisor) * divisor))
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value
