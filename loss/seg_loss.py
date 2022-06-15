import torch
from torch import Tensor
from typing import Union
from typing import Sequence
from torch.nn import Module
from torch.nn.functional import one_hot
from helper.weights import class_weights
from torch.nn.functional import cross_entropy

# noinspection SpellCheckingInspection
__all__ = ['DiceLoss', 'OhemCrossEntropyLoss']


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


# noinspection SpellCheckingInspection
class DiceLoss(Module):
    def __init__(
            self,
            delta: float = 0.5,
            weighted: bool = False,
            class_ids=Union[int, Sequence[int]]
    ):
        """
        delta: Controls weight given to FP and FN.
        This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        if isinstance(class_ids, int):
            self.class_ids = torch.arange(class_ids, dtype=torch.long)
        else:
            assert isinstance(class_ids, Sequence) and all(
                [isinstance(i, int) for i in class_ids]
            ), "'class_ids' are not a integer or a sequence of integer!"
            self.class_ids = torch.tensor(class_ids, dtype=torch.long)
        self.weighted = weighted
        self.num_classes = self.class_ids.numel()

    def forward(
            self, preds: Tensor, labels: Tensor
    ) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        # noinspection PyArgumentList

        weights = class_weights(
            x=labels, class_ids=self.class_ids
        ) if self.weighted else 1
        labels = one_hot(labels, self.num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels * preds, dim=(2, 3))
        fn = torch.sum(labels * (1 - preds), dim=(2, 3))
        fp = torch.sum((1 - labels) * preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (
                tp + self.delta * fn + (1 - self.delta) * fp + 1e-6
        )
        return (weights * (1 - torch.mean(dice_score, dim=0))).mean()


# noinspection SpellCheckingInspection
class OhemCrossEntropyLoss(Module):
    def __init__(
            self,
            ignore_label: int = 0,
            class_ids=Union[int, Sequence[int]],
            threshold: float = 0.7
    ) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.thresh = -torch.log(torch.tensor(threshold, dtype=torch.float))
        if class_ids is not None:
            if isinstance(class_ids, int):
                class_ids = tuple(range(class_ids))
            else:
                assert isinstance(class_ids, Sequence) and all(
                    [isinstance(i, int) for i in class_ids]
                ), "'class_ids' are not a integer or a sequence of integer!"
        self.class_ids = class_ids
        self.ignore_index = ignore_label

    # noinspection SpellCheckingInspection
    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        n_min = targets[targets != self.ignore_label].numel() // 16
        if self.class_ids:
            weights = class_weights(
                x=targets,
                class_ids=torch.tensor(
                    data=self.class_ids,
                    dtype=torch.long,
                    device=targets.device
                )
            )
        else:
            weights = None

        loss = cross_entropy(
            input=preds,
            target=targets,
            weight=weights,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=0.0
        ).view(-1)

        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)
