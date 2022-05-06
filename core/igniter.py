import torch
from typing import Tuple
from torch import nn as tnn
from types import SimpleNamespace
from helper.assist import WrappedLoss
from typing import Any, Optional, Dict
from helper.assist import WrappedOptimizer
from helper.assist import WrappedScheduler
from helper.metrics import MeanScalarMetric
from helper.metrics import SegmentationMetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.core.lightning import LightningModule
from helper.metrics import BaseMetric


class LightningSemSeg(LightningModule):
    def __init__(
            self,
            model: tnn.Module,
            optimizer: WrappedOptimizer,
            criterion: WrappedLoss,
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
        self.normalize_cm = normalize_cm
        self.metric_collection = SimpleNamespace(
            training={
                'metrics': SegmentationMetrics(
                    num_classes=self.model.out_channels,
                    ignore_index=self.ignore_index,
                    normalize_cm=self.normalize_cm,
                    identifier='Training'
                ),
                'loss': MeanScalarMetric(
                    name='Loss',
                    identifier='Training'
                )
            },
            validation={
                'metrics': SegmentationMetrics(
                    num_classes=self.model.out_channels,
                    ignore_index=self.ignore_index,
                    normalize_cm=self.normalize_cm,
                    identifier='Validation'
                ),
                'loss': MeanScalarMetric(
                    name='Loss',
                    identifier='Validation'
                )
            },
            test={
                'metrics': SegmentationMetrics(
                    num_classes=self.model.out_channels,
                    ignore_index=self.ignore_index,
                    normalize_cm=self.normalize_cm,
                    identifier='Test'
                ),
                'loss': MeanScalarMetric(
                    name='Loss',
                    identifier='Test'
                )
            }
        )

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

    def step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            metrics_dict: Dict[str, BaseMetric]
    ):
        image, label = batch
        # noinspection PyArgumentList
        current_batch_size = image.size(0)
        prediction = self.forward(image)
        current_loss = self.loss_function(prediction, label)
        loss_score = current_loss.clone().detach().squeeze()
        metrics_dict['metrics'].update(preds=prediction, target=label)
        metrics_dict['loss'].update(
            score=current_loss.clone().detach().squeeze(),
            count_factor=current_batch_size
        )
        self.log_dict(
            dictionary=loss_score,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        self.log_dict(
            dictionary=self.metrics_dict['loss'].loggable_dict(),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        self.log(
            name='Current Loss',
            value=loss_score,
            on_step=True,
            on_epoch=False,
            logger=False,
            prog_bar=True,
            sync_dist=False
        )
        return current_loss

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        return {
            'loss': self.step(
                batch=batch,
                metrics_dict=self.metric_collection.training
            )
        }

    def on_train_epoch_end(self) -> None:
        self.metric_collection.training['metrics'].reset()
        self.metric_collection.training['loss'].reset()

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Optional[STEP_OUTPUT]:
        return {
            'loss': self.step(
                batch=batch,
                metrics_dict=self.metric_collection.validation
            )
        }

    def on_validation_epoch_end(self) -> None:
        self.metric_collection.validation['metrics'].reset()
        self.metric_collection.validation['loss'].reset()

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Optional[STEP_OUTPUT]:
        return {
            'loss': self.step(
                batch=batch,
                metrics_dict=self.metric_collection.test
            )
        }

    def on_test_epoch_end(self) -> None:
        self.metric_collection.training['metrics'].reset()
        self.metric_collection.training['loss'].reset()

    def predict_step(
            self,
            batch: torch.Tensor,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Any:
        return self.forward(batch)
