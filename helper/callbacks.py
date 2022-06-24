import torch
import pandas as pd
from typing import Any
from typing import Union
from typing import Optional
from typing import Sequence
from matplotlib import cm as mpl_cm
from pytorch_lightning import Trainer
from helper.utils import format_report
from pytorch_lightning import LightningModule
from helper.utils import plot_confusion_matrix
from data_factory.dataset import WritableDataset
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loggers.base import LightningLoggerBase


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
        data_batch = prediction.detach().clone().softmax(dim=1).argmax(
            dim=1, keepdim=True
        ).cpu().numpy()
        # noinspection PyUnresolvedReferences
        self.writable_datasets[dataloader_idx].write_batch(
            data_batch=data_batch,
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
                prediction=outputs,
                batch_indices=batch_indices,
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx
            )
        return None


class ShowMetric(Callback):
    def __init__(self):
        super(ShowMetric, self).__init__()
        self.reports = dict()

    @rank_zero_only
    def prepare_report(
            self,
            tabular_dict: dict,
            loss: Union[float, torch.Tensor],
            key: str
    ):
        loss_df = pd.DataFrame(
            data=[loss.item()], index=['Score'], columns=['Mean Loss']
        )
        tabular_dict['quality_report'] = pd.concat(
            objs=[tabular_dict['quality_report'], loss_df], axis=1
        )
        tabular_dict['quality_report'].index.name = r"Metric"
        self.reports[key] = format_report(tabular_dict)

    def on_validation_epoch_end(
            self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        key = 'Validation'
        self.prepare_report(
            tabular_dict=pl_module.validation_metrics.tabular_report(),
            loss=pl_module.validation_loss.compute(),
            key=key
        )
        rank_zero_info(self.reports[key])

    def on_test_epoch_end(
            self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        key = 'Test'
        self.prepare_report(
            tabular_dict=pl_module.test_metrics.tabular_report(),
            loss=pl_module.test_loss.compute(),
            key=key
        )
        rank_zero_info(self.reports[key])


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
    def log_figure(
            self, cm_df: pd.DataFrame, key: str, logger: LightningLoggerBase
    ):
        fig = plot_confusion_matrix(cm_df=cm_df, key=key, **self.options)
        # noinspection PyUnresolvedReferences
        logger.experiment.add_figure(
            tag=f'{key}-Confusion_Matrix',
            figure=fig,
            close=True
        )

    def on_train_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
    ) -> None:
        metric_collection = pl_module.training_metrics.tabular_report()
        confusion_matrix = metric_collection['confusion_matrix']
        self.log_figure(
            cm_df=confusion_matrix,
            key='Training',
            logger=trainer.logger
        )

    def on_validation_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule
    ) -> None:
        metric_collection = pl_module.validation_metrics.tabular_report()
        confusion_matrix = metric_collection['confusion_matrix']
        self.log_figure(
            cm_df=confusion_matrix,
            key='Validation',
            logger=trainer.logger
        )

    def on_test_epoch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule
    ) -> None:
        metric_collection = pl_module.test_metrics.tabular_report()
        confusion_matrix = metric_collection['confusion_matrix']
        self.log_figure(
            cm_df=confusion_matrix,
            key='Training',
            logger=trainer.logger
        )
