import torch
import pandas as pd
from typing import Any
from typing import Dict
from typing import Union
from typing import Sequence
from abc import abstractmethod
from torchmetrics import Metric
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.functional import (
    precision,
    recall,
    accuracy,
    jaccard_index,
    f1_score,
    matthews_corrcoef,
    cohen_kappa,
    confusion_matrix
)


class BaseMetric(Metric):
    @abstractmethod
    def __init__(self, **kwargs: Any):
        super(BaseMetric, self).__init__(**kwargs)
        pass

    # noinspection SpellCheckingInspection
    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def compute(self) -> Any:
        pass

    @abstractmethod
    def tabular_report(self) -> Any:
        pass

    @abstractmethod
    def loggable_dict(self) -> Dict:
        pass


class MeanScalarMetric(BaseMetric):
    def __init__(
            self,
            name: str,
            identifier: str = None,
            **kwargs
    ):
        super(MeanScalarMetric, self).__init__(**kwargs)
        self.identifier = identifier
        assert isinstance(name, str), "'name' is not a valid string!"
        self.name = name
        self.add_state(
            name="_metric",
            default=torch.tensor(data=0.0, dtype=torch.get_default_dtype()),
            dist_reduce_fx="sum"
        )
        self.add_state(
            name="_count",
            default=torch.tensor(data=0.0, dtype=torch.get_default_dtype()),
            dist_reduce_fx="sum"
        )

    def update_identifier(self, identifier: str):
        self.identifier = identifier

    # noinspection PyAttributeOutsideInit
    def update(
            self, score: torch.Tensor,
            count_factor: Union[int, float] = 1
    ) -> None:
        self._metric += score * count_factor
        self._count += torch.tensor(
            data=count_factor,
            dtype=torch.get_default_dtype()
        )

    def compute(self) -> Any:
        return self._metric / self._count

    @rank_zero_only
    def loggable_dict(self) -> Dict:
        key = (
            str(self.identifier) + '-' + self.name
        ) if self.identifier else self.name
        return {key: self.compute().item()}

    @rank_zero_only
    def tabular_report(self) -> Any:
        return pd.DataFrame(
            data=self.compute().item(),
            columns=[self.name],
            index=['Score']
        )


class ClassPrecision(BaseMetric):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            num_classes: int,
            ignore_index: int = None,
            multiclass: bool = True,
            mdmc_average: str = 'global',
            **kwargs: Any
    ):
        super(ClassPrecision, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.multiclass = multiclass
        self.mdmc_average = mdmc_average
        self.add_state(
            name="_count",
            default=torch.tensor(data=0.0, dtype=torch.get_default_dtype()),
            dist_reduce_fx="sum"
        )
        self.add_state(
            name='_precision',
            default=torch.zeros(
                self.num_classes,
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._count += 1.0
        self._precision += precision(
            preds=preds,
            target=target,
            average='none',
            mdmc_average=self.mdmc_average,
            num_classes=self.num_classes,
            multiclass=self.multiclass,
            ignore_index=self.ignore_index
        )

    def compute(self) -> Any:
        return self._precision / self._count

    @rank_zero_only
    def tabular_report(self) -> Any:
        class_metric = self.compute().tolist()
        indexes = [f'C_{i}' for i in range(len(class_metric))]
        return pd.DataFrame(
            data=class_metric,
            columns=['Precision'],
            index=indexes
        )

    @rank_zero_only
    def loggable_dict(self) -> Dict:
        return {
            'Precision': self.compute().nanmean(dim=None, keepdim=False).item()
        }


class ClassRecall(BaseMetric):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            num_classes: int,
            ignore_index: int = None,
            multiclass: bool = True,
            mdmc_average: str = 'global',
            **kwargs: Any
    ):
        super(ClassRecall, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.multiclass = multiclass
        self.mdmc_average = mdmc_average
        self.add_state(
            name="_count",
            default=torch.tensor(data=0.0, dtype=torch.get_default_dtype()),
            dist_reduce_fx="sum"
        )
        self.add_state(
            name='_precision',
            default=torch.zeros(
                self.num_classes,
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._count += 1.0
        self._recall += recall(
            preds=preds,
            target=target,
            average='none',
            mdmc_average=self.mdmc_average,
            num_classes=self.num_classes,
            multiclass=self.multiclass,
            ignore_index=self.ignore_index
        )

    def compute(self) -> Any:
        return self._recall / self._count

    @rank_zero_only
    def tabular_report(self) -> Any:
        class_metric = self.compute().tolist()
        indexes = [f'C_{i}' for i in range(len(class_metric))]
        return pd.DataFrame(
            data=class_metric,
            columns=['Recall'],
            index=indexes
        )

    @rank_zero_only
    def loggable_dict(self) -> Dict:
        return {
            'Recall': self.compute().nanmean(dim=None, keepdim=False).item()
        }


class ClassIoU(BaseMetric):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            num_classes: int,
            ignore_index: int = None,
            threshold: float = 0.5,
            absent_score: float = 0.0,
            **kwargs: Any
    ):
        super(ClassIoU, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.absent_score = absent_score
        self.add_state(
            name="_count",
            default=torch.tensor(data=0.0, dtype=torch.get_default_dtype()),
            dist_reduce_fx="sum"
        )
        self.add_state(
            name='_iou',
            default=torch.zeros(
                self.num_classes,
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._count += 1.0
        self._iou += jaccard_index(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            threshold=self.threshold,
            absent_score=self.absent_score,
            reduction='none'
        )

    def compute(self) -> Any:
        return self._iou / self._count

    @rank_zero_only
    def tabular_report(self) -> Any:
        class_metric = self.compute().tolist()
        indexes = [f'C_{i}' for i in range(len(class_metric))]
        return pd.DataFrame(
            data=class_metric,
            columns=['IoU'],
            index=indexes
        )

    @rank_zero_only
    def loggable_dict(self) -> Dict:
        return {
            'IoU': self.compute().nanmean(dim=None, keepdim=False).item()
        }


class ClassF1(BaseMetric):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            num_classes: int,
            ignore_index: int = None,
            threshold: float = 0.5,
            mdmc_average: str = 'global',
            top_k: int = None,
            multiclass: bool = True,
            **kwargs: Any
    ):
        super(ClassF1, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.mdmc_average = mdmc_average
        self.top_k = top_k
        self.multiclass = multiclass
        self.add_state(
            name="_count",
            default=torch.tensor(data=0.0, dtype=torch.get_default_dtype()),
            dist_reduce_fx="sum"
        )
        self.add_state(
            name='_iou',
            default=torch.zeros(
                self.num_classes,
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._count += 1.0
        self._f1 += f1_score(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            threshold=self.threshold,
            mdmc_average=self.mdmc_average,
            multiclass=self.multiclass,
            top_k=self.top_k,
            average='none'
        )

    def compute(self) -> Any:
        return self._f1 / self._count

    @rank_zero_only
    def tabular_report(self) -> Any:
        class_metric = self.compute().tolist()
        indexes = [f'C_{i}' for i in range(len(class_metric))]
        return pd.DataFrame(
            data=class_metric,
            columns=['F1'],
            index=indexes
        )

    @rank_zero_only
    def loggable_dict(self) -> Dict:
        return {
            'F1': self.compute().nanmean(dim=None, keepdim=False).item()
        }


class ClassAccuracy(BaseMetric):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            num_classes: int,
            subset_accuracy: bool = False,
            threshold: float = 0.5,
            mdmc_average: str = 'global',
            top_k: int = None,
            multiclass: bool = True,
            ignore_index: int = None,
            **kwargs: Any
    ):
        super(ClassAccuracy, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.subset_accuracy = subset_accuracy
        self.threshold = threshold
        self.mdmc_average = mdmc_average
        self.top_k = top_k
        self.multiclass = multiclass
        self.ignore_index = ignore_index
        self.add_state(
            name="_count",
            default=torch.tensor(data=0.0, dtype=torch.get_default_dtype()),
            dist_reduce_fx="sum"
        )
        self.add_state(
            name='_iou',
            default=torch.zeros(
                self.num_classes,
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._count += 1.0
        self._accuracy += accuracy(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
            threshold=self.threshold,
            mdmc_average=self.mdmc_average,
            multiclass=self.multiclass,
            top_k=self.top_k,
            subset_accuracy=self.subset_accuracy,
            average='none',
            ignore_index=self.ignore_index,
        )

    def compute(self) -> Any:
        return self._accuracy / self._count

    @rank_zero_only
    def tabular_report(self) -> Any:
        class_metric = self.compute().tolist()
        indexes = [f'C_{i}' for i in range(len(class_metric))]
        return pd.DataFrame(
            data=class_metric,
            columns=['Macro_Accuracy'],
            index=indexes
        )

    @rank_zero_only
    def loggable_dict(self) -> Dict:
        return {
            'Macro_Accuracy': self.compute().nanmean(
                dim=None, keepdim=False
            ).item()
        }


class KappaScore(BaseMetric):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            num_classes: int,
            weights: str = None,
            threshold: float = 0.5,
            **kwargs: Any
    ):
        super(KappaScore, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.weights = weights
        self.threshold = threshold

        self.add_state(
            name="_count",
            default=torch.tensor(
                data=0.0,
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )
        self.add_state(
            name='_kappa',
            default=torch.tensor(
                data=0.0,
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._count += 1.0
        self._kappa += cohen_kappa(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
            threshold=self.threshold,
            weights=self.weights
        )

    def compute(self) -> Any:
        return self._kappa / self._count

    @rank_zero_only
    def tabular_report(self) -> Any:
        return pd.DataFrame(
            data=[self.compute().item()],
            columns=['Kappa'],
            index=['Score']
        )

    @rank_zero_only
    def loggable_dict(self) -> Dict:
        return {
            'Kappa': self.compute().item()
        }


class MCCScore(BaseMetric):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            num_classes: int,
            threshold: float = 0.5,
            **kwargs: Any
    ):
        super(MCCScore, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state(
            name="_count",
            default=torch.tensor(
                data=0.0,
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )
        self.add_state(
            name='_mcc',
            default=torch.tensor(
                data=0.0,
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._count += 1.0
        self._mcc += matthews_corrcoef(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
            threshold=self.threshold
        )

    def compute(self) -> Any:
        return self._mcc / self._count

    @rank_zero_only
    def tabular_report(self) -> Any:
        return pd.DataFrame(
            data=[self.compute().item()],
            columns=['MCC'],
            index=['Score']
        )

    @rank_zero_only
    def loggable_dict(self) -> Dict:
        return {
            'MCC': self.compute().item()
        }


class ConfusionMatrix(BaseMetric):
    def __init__(
            self,
            num_classes: int,
            threshold: float = 0.5,
            multilabel: bool = False,
            normalize: str = 'true',
            **kwargs
    ):
        self.num_classes = num_classes
        self.threshold = threshold
        self.multilabel = multilabel
        self.normalize = normalize
        super(ConfusionMatrix, self).__init__(**kwargs)
        self.add_state(
            name="_count",
            default=torch.tensor(
                data=0.0,
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )
        self.add_state(
            name='_confusion_matrix',
            default=torch.zeros(
                size=(self.num_classes, self.num_classes),
                dtype=torch.get_default_dtype()
            ),
            dist_reduce_fx="sum"
        )

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._count += 1.0
        norm = self.normalize if self.normalize in {
            'all', 'none', None
        } else 'all'

        self._confusion_matrix += confusion_matrix(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
            threshold=self.threshold,
            multilabel=self.multilabel,
            normalize=norm
        )

    def get_cm(self):
        return self._confusion_matrix / self._count

    def compute(self) -> Any:
        cm = self.get_cm()
        # noinspection SpellCheckingInspection
        if self.normalize == 'true':
            cm = cm / cm.sum(dim=1, keepdim=True)
        elif self.normalize == 'pred':
            cm = cm / cm.sum(dim=0, keepdim=True)
        else:
            cm = cm
        return cm

    @rank_zero_only
    def tabular_report(self) -> Any:
        titles = [f'C_{i}' for i in range(self.num_classes)]
        return pd.DataFrame(
            data=self.compute().tolist(),
            index=titles,
            columns=titles
        )

    @rank_zero_only
    def loggable_dict(self) -> Dict:
        cm = self.get_cm()
        acc = (
            torch.nansum(
                input=torch.diagonal(input=cm, offset=0)
            ) / torch.nansum(input=cm)
        )
        return {'Micro_Accuracy': acc.item()}


class MetricCollection(BaseMetric):
    def __init__(
            self,
            metrics_dict: Dict[str, BaseMetric] = None,
            **kwargs
    ):
        # noinspection PyTypeChecker
        super(MetricCollection, self).__init__(**kwargs)
        if metrics_dict is None:
            metrics_dict = dict()
        assert isinstance(metrics_dict, Dict) and all(
            isinstance(k, str) and isinstance(v, Metric)
            for k, v in metrics_dict.items()
        ), "'metrics_dict' is invalid"
        self.metrics_dict = metrics_dict

    def add_metrics(self, **kwargs) -> None:
        assert all(
            [
                isinstance(k, str) and isinstance(v, Metric)
                for k, v in kwargs.items()
            ]
        ), "Invalid argument encountered!"
        for k, v in kwargs.items():
            self.metrics_dict[k] = v

    def remove_metrics(self, keys: Union[str, Sequence[str]]):
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            self.metrics_dict.pop(k, default=None)

    def reset(self) -> None:
        for m in self.metrics_dict.values():
            m.reset()

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        for m in self.metrics_dict.values():
            m.update(preds=preds, target=target)

    def compute(self) -> Any:
        computed_metric = dict()
        for k, metric in self.metrics_dict.items():
            computed_metric[k] = metric.compute()

    @abstractmethod
    def tabular_report(self) -> Any:
        pass

    @abstractmethod
    def loggable_dict(self) -> Dict:
        pass


class MixedMetricCollection(MetricCollection):
    def __init__(
            self, metrics_dict: Dict[str, Union[MetricCollection]]
    ):
        assert isinstance(metrics_dict, Dict) and all(
            isinstance(k, str) and isinstance(v, (MetricCollection, BaseMetric))
            for k, v in metrics_dict.items()
        ), "'metrics_dict' is invalid"
        super(MixedMetricCollection, self).__init__(metrics_dict=metrics_dict)

    def loggable_dict(self):
        log_dict = dict()
        for metric in self.metrics_dict.values():
            log_dict.update(metric.loggable_dict())
        return log_dict

    def tabular_report(self):
        report_dict = dict()
        for k, metric in self.metrics_dict.items():
            report_dict[k] = metric.tabular_report()
        return report_dict

    def reset(self) -> None:
        for m in self.metrics_dict.values():
            m.reset()


class SegmentationClassMetrics(MetricCollection):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            num_classes: int,
            ignore_index: int,
            **kwargs
    ):
        self.num_classes = num_classes

        metrics = {
            'Precision': ClassPrecision(
                num_classes=num_classes,
                ignore_index=ignore_index,
                multiclass=True,
                mdmc_average='global',
                **kwargs
            ),
            'Recall': ClassRecall(
                num_classes=self.num_classes,
                ignore_index=ignore_index,
                multiclass=True,
                mdmc_average='global',
                **kwargs
            ),
            'IoU': ClassIoU(
                num_classes=self.num_classes,
                ignore_index=ignore_index,
                threshold=0.5,
                absent_score=0.0,
                **kwargs
            ),
            'F1': ClassF1(
                num_classes=self.num_classes,
                ignore_index=ignore_index,
                multiclass=True,
                threshold=0.5,
                mdmc_average='global',
                top_k=None,
                **kwargs
            ),
            'Accuracy': ClassAccuracy(
                num_classes=self.num_classes,
                ignore_index=ignore_index,
                subset_accuracy=False,
                threshold=0.5,
                mdmc_average='global',
                top_k=None,
                multiclass=True,
                **kwargs
            ),
        }
        super(
            SegmentationClassMetrics, self).__init__(metrics_dict=metrics)

    def tabular_report(self):
        df_list = [
            m.tabular_report()
            for m in self.metrics_dict.values()
        ]
        return pd.concat(df_list, axis=1)

    def loggable_dict(self):
        log_dict = dict()
        for metric in self.metrics_dict.values():
            log_dict.update(metric.loggable_dict())
        return log_dict


class SegmentationQualityMetrics(MetricCollection):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            num_classes: int,
            **kwargs
    ):
        self.num_classes = num_classes

        metrics = {
            'Kappa': KappaScore(
                num_classes=self.num_classes,
                weights=None,
                threshold=0.5,
                **kwargs
            ),
            'MCC': MCCScore(
                num_classes=self.num_classes,
                threshold=0.5,
                **kwargs
            )
        }
        super(
            SegmentationQualityMetrics, self).__init__(metrics_dict=metrics)

    def tabular_report(self):
        df_list = [
            m.tabular_report()
            for m in self.metrics_dict.values()
        ]
        return pd.concat(df_list, axis=1)

    def loggable_dict(self):
        log_dict = dict()
        for metric in self.metrics_dict.values():
            log_dict.update(metric.loggable_dict())
        return log_dict


class SegmentationMetrics(MixedMetricCollection):
    def __init__(
            self,
            num_classes: int,
            ignore_index: int = None,
            normalize_cm: str = 'true',
            identifier: str = None,
            **kwargs
    ):
        self.identifier = identifier
        metrics_dict = {
            "class_metrics": SegmentationClassMetrics(
                num_classes=num_classes,
                ignore_index=ignore_index,
                **kwargs
            ),
            "scalar_metrics": SegmentationQualityMetrics(
                num_classes=num_classes,
                **kwargs
            ),
            "confusion_matrix": ConfusionMatrix(
                num_classes=num_classes,
                threshold=0.5,
                multilabel=False,
                normalize=normalize_cm,
                **kwargs
            )
        }
        super(SegmentationMetrics, self).__init__(metrics_dict=metrics_dict)

    def update_identifier(self, identifier: str):
        self.identifier = identifier

    def loggable_dict(self):
        log_dict = dict()
        prefix = (str(self.identifier) + '-') if self.identifier else ''
        for metric in self.metrics_dict.values():
            log_dict.update(
                {
                    (prefix + k): m
                    for k, m in metric.loggable_dict().items()
                }
            )
        return log_dict
