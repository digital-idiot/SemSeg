import torch


def class_weights(
        x: torch.Tensor,
        class_ids: torch.Tensor
):
    """

    Args:
        x:
        class_ids:
        Reference -> sklearn.utils.class_weight.compute_class_weight

    Returns:

    """
    x = x.view(-1).to(dtype=torch.long)
    class_ids = class_ids.view(-1).to(dtype=torch.long)
    n_classes = class_ids.numel()
    # noinspection PyUnresolvedReferences
    frequencies = (x[:, None] == class_ids[None, :]).sum(dim=0, keepdims=False)
    mask = frequencies == 0
    weights = frequencies.sum() / (n_classes * frequencies)
    weights[mask] = 0
    return weights
