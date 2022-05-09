import torch
import pandas as pd
from typing import Union
from typing import Sequence
from torchmetrics import Recall
from torchmetrics import F1Score
from torchmetrics import Accuracy
from torchmetrics import Precision
from torchmetrics import CohenKappa
from torchmetrics import JaccardIndex
from torchmetrics import ConfusionMatrix
from torchmetrics import MetricCollection
from torchmetrics import MatthewsCorrCoef
from pytorch_lightning.utilities import rank_zero_only


class SegmentationMetrics(MetricCollection):
    # noinspection SpellCheckingInspection
    supported_norms = {'none', 'true', 'pred', 'all'}

    def __init__(
            self,
            num_classes: int,
            multiclass: bool = True,
            ignore_index: int = None,
            prefix: str = None,
            postfix: str = None,
            normalization: str = 'true',
            delimiter: str = '_'
    ):
        if not prefix:
            prefix = str()
        if not postfix:
            postfix = str()

        assert normalization in self.supported_norms, (
            f"Invalid 'normalization': {normalization}!\n" +
            f"Supported: {self.supported_norms}"
        )
        self.normalize = normalization
        self.num_classes = num_classes
        if ignore_index:
            assert 0 <= ignore_index < self.num_classes, (
                "'ignore_index' is out of range!"
            )
        self.ignore_index = ignore_index
        self.pre = prefix
        self.post = postfix
        # noinspection SpellCheckingInspection
        metrics_dict = {
            'Precision': Precision(
                num_classes=num_classes,
                threshold=0.5,
                mdmc_average='samplewise',
                ignore_index=ignore_index,
                average='none',
                top_k=None,
                multiclass=multiclass
            ),
            'Recall': Recall(
                num_classes=num_classes,
                threshold=0.5,
                mdmc_average='samplewise',
                ignore_index=ignore_index,
                average='none',
                top_k=None,
                multiclass=multiclass
            ),
            'IoU': JaccardIndex(
                num_classes=num_classes,
                ignore_index=ignore_index,
                absent_score=0.0,
                threshold=0.5,
                multilabel=False,
                reduction='none'
            ),
            'F1': F1Score(
                num_classes=num_classes,
                threshold=0.5,
                average='none',
                mdmc_average='samplewise',
                ignore_index=ignore_index,
                top_k=None,
                multiclass=multiclass
            ),
            'Accuracy': Accuracy(
                threshold=0.5,
                num_classes=num_classes,
                average='none',
                mdmc_average='samplewise',
                ignore_index=ignore_index,
                top_k=None,
                multiclass=multiclass,
                subset_accuracy=False
            ),
            "Confusion_Matrix": ConfusionMatrix(
                num_classes=num_classes,
                threshold=0.5,
                multilabel=False,
                normalize='none'
            ),
            "MCC": MatthewsCorrCoef(
                num_classes=num_classes,
                threshold=0.5
            ),
            "Kappa": CohenKappa(
                num_classes=num_classes,
                weights=None,
                threshold=0.5
            )
        }
        super(SegmentationMetrics, self).__init__(
            metrics=metrics_dict,
            prefix=f"{prefix}{delimiter}",
            postfix=f"{delimiter}{postfix}",
            compute_groups=False  # True when bug is fixed
        )
        self.vector_keys = (
            "Precision",
            "Recall",
            "IoU",
            "F1",
            "Accuracy"
        )
        self.scalar_keys = (
            "MCC",
            "Kappa"
        )
        self.cm_key = "Confusion_Matrix"

    @rank_zero_only
    def wrap_keys(self, keys=Union[str, Sequence[str]]):
        if isinstance(keys, str):
            return f"{self.prefix}{keys}{self.postfix}"
        else:
            assert isinstance(keys, Sequence), (
                "'keys' argument is not a string or Sequence"
            )
            return [
                f"{self.prefix}{k}{self.postfix}"
                for k in keys
            ]

    # noinspection PyTypeChecker,PyUnresolvedReferences
    @rank_zero_only
    def loggable_dict(self):
        class_scores = self.compute()
        score_dict = dict()
        score_dict[self.wrap_keys('Micro_Accuracy')] = torch.nanmean(
            input=1.0 * torch.diagonal(
                input=class_scores[self.wrap_keys(self.cm_key)],
                offset=0,
                dim1=0,
                dim2=1
            ),  # multiply with 1.0 to avoid explicit type cast
            keepdim=False
        )
        for k in self.wrap_keys(self.vector_keys):
            score_dict[k] = (1.0 * class_scores[k]).nanmean(
                dim=None, keepdim=False
            )
        for k in self.wrap_keys(self.scalar_keys):
            score_dict[k] = class_scores[k]

        return score_dict

    # noinspection SpellCheckingInspection,PyTypeChecker
    @rank_zero_only
    def tabular_report(self):
        score_dict = self.compute()
        data = torch.stack(
            tensors=[score_dict[self.wrap_keys(k)] for k in self.vector_keys],
            dim=-1
        ).tolist()
        indexes = [f"C_{i}" for i in range(self.num_classes)]
        if self.ignore_index:
            data = data[:self.ignore_index] + data[(self.ignore_index + 1):]
            true_indexes = (
                indexes[:self.ignore_index] +
                indexes[(self.ignore_index + 1):]
            )
        else:
            true_indexes = indexes

        class_df = pd.DataFrame(
            data=data,
            index=true_indexes,
            columns=self.vector_keys
        )
        class_df.index.name = r"Class \ Metric"

        scalar_df = pd.DataFrame(
            data=[
                [
                    score_dict[k].item()
                    for k in self.wrap_keys(self.scalar_keys)
                ]
            ],
            index=['Score'],
            columns=self.scalar_keys
        )
        scalar_df.index.name = r"Metric"

        confmat = score_dict[self.wrap_keys(self.cm_key)]

        if self.normalize == "true":
            confmat = confmat / confmat.sum(axis=1, keepdim=True)
        elif self.normalize == "pred":
            confmat = confmat / confmat.sum(axis=0, keepdim=True)
        elif self.normalize == "all":
            confmat = confmat / confmat.sum()
        else:
            confmat = confmat

        confmat_df = pd.DataFrame(
            data=confmat.tolist(),
            index=indexes,
            columns=indexes
        )
        scalar_df.index.name = r"True \ Predicted"

        return {
            'prefix': self.pre,
            'postfix': self.post,
            'class_report': class_df,
            'quality_report': scalar_df,
            'confusion_matrix': confmat_df
        }
