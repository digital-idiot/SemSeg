import torch
from typing import Any
from typing import Union
from typing import Tuple
from typing import Optional
from torch import nn as tnn
from torchmetrics import MeanMetric
from helper.assist import WrappedLoss
from helper.assist import WrappedOptimizer
from helper.assist import WrappedScheduler
from helper.metrics import SegmentationMetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.core.lightning import LightningModule


class LightningSemSeg(LightningModule):
    def __init__(
            self,
            model: tnn.Module,
            optimizer: WrappedOptimizer,
            criterion: Union[WrappedLoss, tnn.Module],
            scheduler: WrappedScheduler = None,
            ignore_index: int = None,
            normalize_cm: str = 'true'
    ) -> None:
        """
        Wrapper to make model lightning compatible
        Args:
            model: Model
            optimizer: Optimizer, use partial function for extra args
            criterion: Loss function
            scheduler: scheduler, optional, use partial function for extra args
        """
        super(LightningSemSeg, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ignore_index = ignore_index
        self.normalization = normalize_cm

        self.training_metrics = SegmentationMetrics(
            num_classes=self.model.out_channels,
            multiclass=True,
            ignore_index=self.ignore_index,
            normalization=self.normalization,
            prefix='Training',
            postfix=None,
            delimiter='-'
        )
        self.validation_metrics = SegmentationMetrics(
            num_classes=self.model.out_channels,
            multiclass=True,
            ignore_index=self.ignore_index,
            normalization=self.normalization,
            prefix='Validation',
            postfix=None,
            delimiter='-'
        )
        self.test_metrics = SegmentationMetrics(
            num_classes=self.model.out_channels,
            multiclass=True,
            ignore_index=self.ignore_index,
            normalization=self.normalization,
            prefix='Test',
            postfix=None,
            delimiter='-'
        )

        self.training_loss = MeanMetric(nan_strategy='warn')
        self.validation_loss = MeanMetric(nan_strategy='warn')
        self.test_loss = MeanMetric(nan_strategy='warn')

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def loss_function(self, prediction: Any, target: Any) -> Any:
        return self.criterion(prediction, target)

    def configure_optimizers(self) -> Any:
        # noinspection PyCallingNonCallable
        out = dict()
        optimizer = self.optimizer(self.model.parameters())
        out['optimizer'] = optimizer
        if self.scheduler:
            scheduler = self.scheduler(optimizer=optimizer)
            out['lr_scheduler'] = scheduler
        return out

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
            optimizer_idx: int = 0
    ) -> STEP_OUTPUT:
        image, label = batch
        # noinspection PyArgumentList
        current_batch_size = image.size(0)
        prediction = self.forward(image)
        current_loss = self.loss_function(prediction, label)
        self.training_metrics.update(preds=prediction, target=label)
        self.training_loss.update(
            value=current_loss.clone().detach().squeeze(),
            weight=(1.0 / current_batch_size)
        )
        self.log(
            name="Training-Mean_Loss",
            value=self.training_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        self.log_dict(
            dictionary=self.training_metrics.loggable_dict(),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        return {
            'loss': current_loss
        }

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Optional[STEP_OUTPUT]:
        image, label = batch
        # noinspection PyArgumentList
        current_batch_size = image.size(0)
        prediction = self.forward(image)
        current_loss = self.loss_function(prediction, label)
        self.validation_metrics.update(preds=prediction, target=label)
        self.validation_loss.update(
            value=current_loss.clone().detach().squeeze(),
            weight=(1.0 / current_batch_size)
        )
        self.log(
            name="Validation-Mean_Loss",
            value=self.validation_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        self.log_dict(
            dictionary=self.validation_metrics.loggable_dict(),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        return {
            'val_loss': current_loss
        }

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Optional[STEP_OUTPUT]:
        image, label = batch
        # noinspection PyArgumentList
        current_batch_size = image.size(0)
        prediction = self.forward(image)
        current_loss = self.loss_function(prediction, label)
        self.test_metrics.update(preds=prediction, target=label)
        self.test_loss.update(
            value=current_loss.clone().detach().squeeze(),
            weight=(1.0 / current_batch_size)
        )
        self.log(
            name="Test-Mean_Loss",
            value=self.test_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        self.log_dict(
            dictionary=self.test_metrics.loggable_dict(),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        return {
            'test_loss': current_loss
        }

    def on_test_epoch_end(self) -> None:
        self.test_metrics.reset()
        self.test_loss.reset()

    def predict_step(
            self,
            batch: torch.Tensor,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Any:
        return self.forward(batch)
