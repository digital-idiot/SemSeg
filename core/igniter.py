from torch import nn as tnn
from typing import Any, Optional
from helper.utils import format_dict
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

        self.training_metrics = SegmentationMetrics(
            num_classes=self.model.out_channels,
            ignore_index=self.ignore_index,
            normalize_cm=self.normalize_cm
        )
        self.validation_metrics = SegmentationMetrics(
            num_classes=self.model.out_channels,
            ignore_index=self.ignore_index,
            normalize_cm=self.normalize_cm
        )
        self.test_metrics = SegmentationMetrics(
            num_classes=self.model.out_channels,
            ignore_index=ignore_index,
            normalize_cm=normalize_cm
        )

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def loss(self, prediction: Any, target: Any) -> Any:
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

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img, lbl = batch
        prd = self.forward(img)
        lss = self.loss(prd, lbl)

        self.log_dict(
            dictionary=self.training_metrics.loggable_dict(),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        self.class_metrics_training.update(preds=prd, target=lbl)
        self.scalar_metrics_training.update(preds=prd, target=lbl)
        self.cm_training.update(preds=prd, target=lbl)
        self.log(
            name='training_loss',
            value=lss,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
            prog_bar=True
        )
        cm_training = format_dict(
            dictionary=self.class_metrics_training.loggable_dict(),
            key_prefix='training',
            key_suffix='',
            delimiter='_'
        )
        self.log_dict(
            dictionary=cm_training,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        sm_training = format_dict(
            dictionary=self.class_metrics_training.loggable_dict(),
            key_prefix='validation',
            key_suffix='',
            delimiter='_'
        )
        self.log_dict(
            dictionary=sm_training,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=False,
            prog_bar=False
        )
        return {'loss': lss}

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img, lbl = batch
        prd = self.forward(img)
        lss = self.loss(prd, lbl)
        self.class_metrics_validation.update(preds=prd, target=lbl)
        self.scalar_metrics_validation.update(preds=prd, target=lbl)
        self.cm_validation.update(preds=prd, target=lbl)
        self.log(
            name='validation_loss',
            value=lss,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
            prog_bar=True
        )
        cm_validation = format_dict(
            dictionary=self.class_metrics_validation.loggable_dict(),
            key_prefix='validation',
            key_suffix='',
            delimiter='_'
        )
        self.log_dict(
            dictionary=cm_validation,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=False,
            prog_bar=False
        )
        sm_validation = format_dict(
            dictionary=self.class_metrics_validation.loggable_dict(),
            key_prefix='validation',
            key_suffix='',
            delimiter='_'
        )
        self.log_dict(
            dictionary=sm_validation,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=False
        )
        # noinspection PyUnresolvedReferences
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     lss = lss.unsqueeze(0)
        return {'loss': lss}

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img, lbl = batch
        prd = self.forward(img)
        lss = self.loss(prd, lbl)
        self.class_metrics_test.update(preds=prd, target=lbl)
        self.scalar_metrics_test.update(preds=prd, target=lbl)
        self.cm_test.update(preds=prd, target=lbl)
        self.log(
            name='test_loss',
            value=lss,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
            prog_bar=True
        )
        cm_test = format_dict(
            dictionary=self.class_metrics_test.loggable_dict(),
            key_prefix='test',
            key_suffix='',
            delimiter='_'
        )
        self.log_dict(
            dictionary=cm_test,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=False,
            prog_bar=False
        )
        sm_test = format_dict(
            dictionary=self.class_metrics_test.loggable_dict(),
            key_prefix='test',
            key_suffix='',
            delimiter='_'
        )
        self.log_dict(
            dictionary=sm_test,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=False,
            prog_bar=False
        )
        # noinspection PyUnresolvedReferences
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     lss = lss.unsqueeze(0)
        return {'loss': lss}

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> Any:
        return self.forward(batch)
