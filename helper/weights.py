import torch


def class_weights(
        x: torch.Tensor,
        class_ids: torch.Tensor
):
    x = x.view(-1).to(dtype=torch.long)
    class_ids = class_ids.view(-1).to(dtype=torch.long)
    n_classes = class_ids.numel()
    # noinspection PyUnresolvedReferences
    frequencies = (x[:, None] == class_ids[None, :]).sum(dim=1)
    mask = frequencies == 0
    frequencies = frequencies.to(dtype=torch.get_default_dtype())
    weights = n_classes / torch.log1p(frequencies)
    weights[mask] = 0
    weights = weights / weights.sum()
    return weights
