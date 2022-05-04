import torch
from torch.nn.functional import one_hot


# noinspection SpellCheckingInspection
def dice_loss(
        preds: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 0.0,
        eps: float = 1e-7
):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss, so we
    return the negated dice loss.
    Args:
        targets: a tensor of shape [B, H, W].
        preds: a tensor of shape [B, C, H, W]. Corresponds to
            the softmax(logits) for multiclass sigmoid(logits) for binary.
        smooth: Laplace smooting parameter
        eps: added to the denominator for numerical stability.
            Not in use when smooth is specified
    Returns:
        dl: the Sørensen–Dice loss.
    """
    # noinspection PyArgumentList
    num_classes = preds.size(1)
    true_onehot = one_hot(
        targets,
        num_classes=num_classes
    ).type_as(preds)
    true_onehot = true_onehot.permute(0, 3, 1, 2)

    if num_classes == 1:
        pos_prob = preds
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        probas = preds

    dims = (0,) + tuple(range(2, probas.ndimension()))
    intersection = torch.sum(probas * true_onehot, dims)
    cardinality = torch.sum(probas + true_onehot, dims)
    m, n = (smooth, smooth) if smooth else (0, eps)
    dl = (1.0 - (((2.0 * intersection) + m) / (cardinality + n))).mean()
    return dl
