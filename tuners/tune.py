from typing import Any
from typing import Dict
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.core.lightning import LightningModule


def tune_batch(
        model: LightningModule,
        tuning_params: Dict[str, Any] = None,
        trainer_args: Dict[str, Any] = None
):
    if trainer_args is None:
        trainer_args = dict()
    if tuning_params is None:
        tuning_params = dict()
    dummy_trainer = Trainer(
        **trainer_args
    )
    tuner = Tuner(dummy_trainer)
    batch_size = tuner.scale_batch_size(model=model, **tuning_params)
    del tuner, dummy_trainer
    return batch_size


def tune_lr(
        model: LightningModule,
        tuning_params: Dict[str: Any] = None,
        trainer_args: Dict[str, Any] = None
):
    if trainer_args is None:
        trainer_args = dict()
    if tuning_params is None:
        tuning_params = dict()
    dummy_trainer = Trainer(
        **trainer_args
    )
    tuner = Tuner(dummy_trainer)
    lr_finder = tuner.lr_find(model=model, **tuning_params)
    lr = lr_finder.suggestion()
    del tuner, dummy_trainer
    return lr
