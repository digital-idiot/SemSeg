import torch
from typing import Any
from io import StringIO
from typing import Union
from typing import Callable
from typing import Optional
from typing import Sequence
from rich.table import Table
from functools import partial
from torchmetrics import Metric
from rich.console import Console
from matplotlib import cm as mpl_cm
from helper.utils import format_float
from pytorch_lightning import Trainer
from torch.optim.optimizer import Optimizer
from pytorch_lightning import LightningModule
from helper.utils import plot_confusion_matrix
from pytorch_lightning.callbacks import Callback
from data_factory.dataset import WritableDataset
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import BasePredictionWriter
from torchmetrics.functional import (
    precision,
    recall,
    accuracy,
    jaccard_index,
    f1_score,
    matthews_corrcoef,
    cohen_kappa
)


class WrappedOptimizer(object):
    def __init__(self, optimizer: Optimizer, **opt_params):
        self.opt = optimizer
        self.params = opt_params

    def __call__(self, model_params):
        # noinspection PyCallingNonCallable
        return self.opt(params=model_params, **self.params)


class WrappedLoss(object):
    def __init__(self, loss_fn: Callable, **extra_params):
        # noinspection SpellCheckingInspection
        """
        Wrap a loss function
        Args:
            loss_fn: target function,
                must take preds and targets as keyword arguments
            **extra_params: additional keyword arguments for loss_fn
        """
        self.loss_fn = loss_fn
        self.extra_params = extra_params

    # noinspection SpellCheckingInspection
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        return self.loss_fn(preds=preds, targets=targets, **self.extra_params)


class WrappedScheduler(object):
    def __init__(self, scheduler: Callable, **extra_params):
        """
        Wrap a scheduler
        Args:
            scheduler: target scheduler class
            **extra_params: additional keyword arguments for the scheduler
        """
        self.scheduler = scheduler
        self.extra_params = extra_params

    def __call__(self, optimizer):
        return self.scheduler(optimizer=optimizer, **self.extra_params)


class SemSegMultiMetrics(Metric):
    __names = ['Precision', 'Recall', 'Accuracy', 'IoU', 'F1']

    def __init__(
            self,
            num_classes: int,
            ignore_index: int,
            **kwargs
    ):
        super(SemSegMultiMetrics, self).__init__(
            **kwargs
        )
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state(
            name="_count",
            default=torch.tensor(0.0),
            dist_reduce_fx="mean"
        )
        self.add_state(
            name="_accuracy",
            default=torch.tensor(0.0),
            dist_reduce_fx="mean"
        )
        self.add_state(
            name='_precision',
            default=torch.tensor(0.0),
            dist_reduce_fx="mean"
        )
        self.add_state(
            name='_recall',
            default=torch.tensor(0.0),
            dist_reduce_fx="mean"
        )
        self.add_state(
            name='_iou',
            default=torch.tensor(0.0),
            dist_reduce_fx="mean"
        )
        self.add_state(
            name='_f1',
            default=torch.tensor(0.0),
            dist_reduce_fx="mean"
        )

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.count += 1.0
        self._accuracy += accuracy(
            preds=preds,
            target=target,
            average='none',
            mdmc_average='global',
            num_classes=self.num_classes,
            multiclass=True,
            ignore_index=self.ignore_index
        )
        self._precision += precision(
            preds=preds,
            target=target,
            average='none',
            mdmc_average='global',
            num_classes=self.num_classes,
            multiclass=True,
            ignore_index=self.ignore_index
        )
        self._recall += recall(
            preds=preds,
            target=target,
            average='none',
            mdmc_average='global',
            num_classes=self.num_classes,
            multiclass=True,
            ignore_index=self.ignore_index
        )
        self._iou += jaccard_index(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
            reduction='none',
            ignore_index=self.ignore_index
        )
        self._f1 += f1_score(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
            average='none',
            mdmc_average='global',
            multiclass=True,
            ignore_index=self.ignore_index
        )

    def compute(self):
        return torch.stack(
            (
                self._precision,
                self._recall,
                self._accuracy,
                self._iou,
                self._f1
            ),
            dim=-1
        ) / self.count

    @rank_zero_only
    def make_report(self):
        rich_table = Table(
            title="Report",
            style='cyan',
            show_lines=True
        )
        ff2 = partial(format_float, n=2)
        rich_table.add_column(r'Class \ Metrics', style='bold')
        for name in self.__names:
            rich_table.add_column(name)
        nested_list = self.compute().tolist()
        for i in range(len(nested_list)):
            rich_table.add_row(
                f"C_{i}", *list(map(ff2, nested_list[i]))
            )
        out = Console(file=StringIO())
        out.print(rich_table)
        # noinspection PyUnresolvedReferences
        return out.file.getvalue()

    @rank_zero_only
    def loggable_dict(self):
        dat = self.compute().nanmean(dim=0).tolist()
        assert len(dat) == len(self.__names)
        return {
            k: v
            for k, v in zip(self.__names, dat)
        }


class SemSegScalarMetrics(Metric):
    def __init__(
            self,
            num_classes: int,
            dist_sync_on_step=False
    ):
        # noinspection PyTypeChecker
        super(SemSegScalarMetrics, self).__init__(
            dist_sync_on_step=dist_sync_on_step
        )
        self.num_classes = num_classes
        self.add_state(
            name="_count",
            default=torch.tensor(0.0),
            dist_reduce_fx="mean"
        )
        self.add_state(
            name="_kappa",
            default=torch.tensor(0.0),
            dist_reduce_fx="mean"
        )
        self.add_state(
            name='_mcc',
            default=torch.tensor(0.0),
            dist_reduce_fx="mean"
        )

    # noinspection SpellCheckingInspection
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.count += 1.0
        self._kappa += cohen_kappa(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
        )
        self._mcc += matthews_corrcoef(
            preds=preds,
            target=target,
            num_classes=self.num_classes,
        )

    def compute(self):
        return torch.stack(
            (
                self._kappa,
                self._mcc,
            ),
            dim=-1
        ) / self.count

    @rank_zero_only
    def make_report(self):
        k, mcc = self.compute().tolist()
        return "\n".join(
            [
                f"Cohen Kappa: {k:.2f}",
                f"Matthews Correlation Coefficient: {mcc:.2f}"
            ]
        )

    @rank_zero_only
    def loggable_dict(self):
        dat = self.compute().nanmean(dim=0).tolist()
        assert len(dat) == 2
        return {'Kappa': dat[0], 'MCC': dat[1]}


class ShowMetric(Callback):
    def __init__(self):
        super(ShowMetric, self).__init__()
        self.report = None

    def on_train_epoch_end(self, trainer, pl_module):
        assert not(pl_module is None)
        scalar_metrics = trainer.callback_metrics['scalar_metrics_training']
        multi_metrics = trainer.callback_metrics['class_metrics_training']
        scalar_report = scalar_metrics.make_report()
        tabular_report = multi_metrics.make_report()
        self.report = ' \n '.join(
            [
                scalar_report, tabular_report
            ]
        )
        rank_zero_info(self.report)

    def on_validation_epoch_end(self, trainer, pl_module):
        assert not(pl_module is None)
        scalar_metrics = trainer.callback_metrics['scalar_metrics_validation']
        multi_metrics = trainer.callback_metrics['class_metrics_validation']
        scalar_report = scalar_metrics.make_report()
        tabular_report = multi_metrics.make_report()
        self.report = ' \n '.join(
            [
                scalar_report, tabular_report
            ]
        )
        rank_zero_info(self.report)

    def on_test_epoch_end(self, trainer, pl_module):
        assert not(pl_module is None)
        scalar_metrics = trainer.callback_metrics['scalar_metrics_test']
        multi_metrics = trainer.callback_metrics['class_metrics_test']
        scalar_report = scalar_metrics.make_report()
        tabular_report = multi_metrics.make_report()
        self.report = ' \n '.join(
            [
                scalar_report, tabular_report
            ]
        )
        rank_zero_info(self.report)


class LogConfusionMatrix(Callback):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            annot=True,
            fig_size=10,
            dpi=300,
            font_size=22,
            fmt='.2f',
            cmap=mpl_cm.plasma,
            cbar=True,
    ):
        super(LogConfusionMatrix, self).__init__()
        self.options = {
            'annot': annot,
            'fig_size': fig_size,
            'dpi': dpi,
            'font_size': font_size,
            'fmt': fmt,
            'cmap': cmap,
            'cbar': cbar
        }

    @rank_zero_only
    def on_train_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
    ) -> None:
        cm = trainer.callback_metrics['cm_training']
        self.options['normed'] = True if (
                cm.normalize in {None, 'none'}
        ) else False
        cm = cm.compute().clone().detach().cpu().numpy()
        self.options['key'] = 'Training'
        fig = plot_confusion_matrix(cm=cm, **self.options)
        # noinspection PyUnresolvedReferences
        trainer.logger.experiment.add_figure(
            tag='confusion_matrix_training',
            figure=fig,
            close=True
        )

    @rank_zero_only
    def on_validation_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule
    ) -> None:
        cm = trainer.callback_metrics['cm_validation']
        self.options['normed'] = True if (
                cm.normalize in {None, 'none'}
        ) else False
        cm = cm.compute().clone().detach().cpu().numpy()
        self.options['key'] = 'Validation'
        fig = plot_confusion_matrix(cm=cm, **self.options)
        # noinspection PyUnresolvedReferences
        trainer.logger.experiment.add_figure(
            tag='confusion_matrix_validation',
            figure=fig,
            close=True
        )

    @rank_zero_only
    def on_test_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule
    ) -> None:
        cm = trainer.callback_metrics['cm_test']
        self.options['normed'] = True if (
                cm.normalize in {None, 'none'}
        ) else False
        cm = cm.compute().clone().detach().cpu().numpy()
        self.options['key'] = 'Testing'
        fig = plot_confusion_matrix(cm=cm, **self.options)
        # noinspection PyUnresolvedReferences
        trainer.logger.experiment.add_figure(
            tag='confusion_matrix_test',
            figure=fig,
            close=True
        )


class PredictionWriter(BasePredictionWriter):
    def __init__(
            self,
            writable_datasets: Union[
                WritableDataset, Sequence[WritableDataset]
            ],
            overwrite: bool = False
    ):
        super(PredictionWriter, self).__init__(write_interval='batch')
        if isinstance(writable_datasets, WritableDataset):
            writable_datasets = [writable_datasets]
        assert all(
            [
                isinstance(wds, WritableDataset)
                for wds in writable_datasets
            ]
        ), "one or more datasets are not instances of WritableDataset!"
        self.writable_datasets = writable_datasets
        self.overwrite = bool(overwrite)

    @rank_zero_only
    def write_on_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            prediction: Any,
            batch_indices: Optional[Sequence[int]],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int
    ) -> None:
        # noinspection PyUnresolvedReferences
        self.writable_datasets[dataloader_idx].write_batch(
            data_batch=prediction,
            indexes=batch_indices,
            overwrite=self.overwrite
        )
        return None

    def on_predict_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int
    ) -> None:
        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        if self.interval.on_batch:
            self.write_on_batch_end(
                trainer=trainer,
                pl_module=pl_module,
                predictio=outputs,
                batch_indices=batch_indices,
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx
            )
        return None

    @rank_zero_only
    def write_on_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            predictions: Sequence[Any],
            batch_indices: Optional[Sequence[Any]]
    ) -> None:
        raise NotImplementedError(
            "Writing on epoch end is not supported!"
        )

    def on_predict_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: Sequence[Any]
    ) -> None:
        raise NotImplementedError(
            "Writing on epoch end is not supported!"
        )
