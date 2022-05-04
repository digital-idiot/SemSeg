from typing import Any
from typing import Union
from typing import Optional
from typing import Sequence
from matplotlib import cm as mpl_cm
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from helper.utils import plot_confusion_matrix
from data_factory.dataset import WritableDataset
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import BasePredictionWriter


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


class ShowMetric(Callback):
    def __init__(self):
        super(ShowMetric, self).__init__()
        self.report = None

    def on_train_epoch_end(self, trainer, pl_module):
        scalar_metrics = trainer.callback_metrics['scalar_metrics_training']
        multi_metrics = trainer.callback_metrics['class_metrics_training']
        scalar_report = scalar_metrics.tabular_report()
        tabular_report = multi_metrics.tabular_report()
        self.report = ' \n '.join(
            [
                scalar_report, tabular_report
            ]
        )
        rank_zero_info(self.report)

    def on_validation_epoch_end(self, trainer, pl_module):
        scalar_metrics = trainer.callback_metrics['scalar_metrics_validation']
        multi_metrics = trainer.callback_metrics['class_metrics_validation']
        scalar_report = scalar_metrics.tabular_report()
        tabular_report = multi_metrics.tabular_report()
        self.report = ' \n '.join(
            [
                scalar_report, tabular_report
            ]
        )
        rank_zero_info(self.report)

    def on_test_epoch_end(self, trainer, pl_module):
        scalar_metrics = trainer.callback_metrics['scalar_metrics_test']
        multi_metrics = trainer.callback_metrics['class_metrics_test']
        scalar_report = scalar_metrics.tabular_report()
        tabular_report = multi_metrics.tabular_report()
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
