import warnings
from pathlib import Path
from tuners.tune import tune_lr
from tuners.tune import tune_batch
from torch_optimizer import AdaBound
from pytorch_lightning import Trainer
from helper.assist import WrappedLoss
from core.models import TopFormerModel
from helper.callbacks import ShowMetric
from core.igniter import LightningSemSeg
from helper.assist import WrappedOptimizer
from helper.assist import WrappedScheduler
from loss.seg_loss import OhemCrossEntropyLoss
from helper.callbacks import LogConfusionMatrix
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import OneCycleLR
from data_factory.transforms import RandomAffine
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import vflip
from data_factory.transforms import RandomSharpness
from data_factory.dataset import DatasetConfigurator
from torchvision.transforms import InterpolationMode
from data_factory.data_module import IgniteDataModule
from data_factory.transforms import RandomPerspective
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from data_factory.dataset import ReadableImagePairDataset
from data_factory.transforms import SegmentationTransform
from torchvision.transforms.functional import autocontrast
from data_factory.utils import image_to_tensor, label_to_tensor
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


warnings.filterwarnings('error', category=UserWarning)


if __name__ == '__main__':
    # TODO: Read all parameters from a conf file
    checkpoint_dir = Path("checkpoints")
    image_shape = (768, 1024)
    max_epochs = 500
    model = TopFormerModel(
        num_classes=10,
        config_alias='B',
        input_channels=3,
        injection_type='dot_sum',
        fusion='sum',
        head_cfg=frozenset({'alias': 'refiner'}.items())
    )
    optimizer = WrappedOptimizer(
        optimizer=AdaBound,
        lr=1e-3,
        final_lr=0.1,
        amsbound=False
    )

    loss_function = WrappedLoss(
        loss_fn=OhemCrossEntropyLoss(
            ignore_index=0,
            threshold=0.7,
            min_kept=0.1
        )
    )

    augmentor = SegmentationTransform(mode='any')
    augmentor.add_transform(
        transform=hflip,
        target='both',
        probability=0.5
    )
    augmentor.add_transform(
        transform=vflip,
        target='both',
        probability=0.5
    )
    augmentor.add_transform(
        transform=autocontrast,
        target='image',
        probability=0.5
    )
    augmentor.add_transform(
        transform=RandomSharpness(
            sharpness_range=(0.8, 1.2)
        ),
        target='image',
        probability=0.5
    )
    augmentor.add_transform(
        transform=RandomAffine(
            degrees=(-180, 180),
            translate=(0.15, 0.15),
            scale=(0.8, 1.2),
            shear=None,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
            center=None
        ),
        target='sync_pair',
        probability=0.5
    )
    augmentor.add_transform(
        transform=RandomPerspective(
            image_shape=image_shape,
            distortion_scale=0.2,
            interpolation=InterpolationMode.NEAREST,
            fill=0
        ),
        target='sync_pair',
        probability=0.5
    )

    train_dataset = DatasetConfigurator(
        conf_path="Data/FloodNetData/Train/Train.json"
    ).generate_paired_dataset(
        image_channels=(1, 2, 3),
        label_channels=1,
        target_shape=image_shape,
        pad_aspect=0,
        image_resampling=0,
        label_resampling=0,
        transform=augmentor,
        image_converter=image_to_tensor,
        label_converter=label_to_tensor
    )

    val_dataset = DatasetConfigurator(
        conf_path="Data/FloodNetData/Val/Val.json"
    ).generate_paired_dataset(
        image_channels=(1, 2, 3),
        label_channels=1,
        target_shape=image_shape,
        pad_aspect=0,
        image_resampling=0,
        label_resampling=0,
        transform=None,
        image_converter=image_to_tensor,
        label_converter=label_to_tensor
    )

    scheduler = WrappedScheduler(
        scheduler=OneCycleLR,
        max_lr=0.01,
        steps_per_epoch=len(train_dataset),
        epochs=max_epochs
    )

    net = LightningSemSeg(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=loss_function,
        ignore_index=0,
        normalize_cm='true'
    )

    test_dataset = DatasetConfigurator(
        conf_path="Data/FloodNetData/Test/Test.json"
    ).generate_paired_dataset(
        image_channels=(1, 2, 3),
        label_channels=1,
        target_shape=image_shape,
        pad_aspect=0,
        image_resampling=0,
        label_resampling=0,
        transform=None,
        image_converter=image_to_tensor,
        label_converter=label_to_tensor
    )

    data_module = IgniteDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        predict_dataset=None,
        num_workers=8,
        batch_size=1,
        shuffle=True,
        collate_fn=ReadableImagePairDataset.collate
    )

    # noinspection SpellCheckingInspection
    data_module.batch_size = tune_batch(
        model=net,
        tuning_params={
            "mode": "power",
            "datamodule": data_module,
            "max_trials": 2
        },
        trainer_args={
            "callbacks": [
                StochasticWeightAveraging(swa_lrs=1e-2),
                RichProgressBar(),
                ShowMetric(),
                LogConfusionMatrix(),
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename='FloodNet-{epoch}-{val_loss:.3f}',
                    monitor='val_loss',
                    save_top_k=2,
                    save_last=True,
                    save_on_train_epoch_end=False
                ),
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=10,
                    strict=True,
                    check_finite=True,
                    min_delta=1e-3,
                    check_on_train_epoch_end=False,
                )
            ],
            "accumulate_grad_batches": 1,
            "check_val_every_n_epoch": 10,
            "num_sanity_val_steps": 0,
            "detect_anomaly": False,
            "log_every_n_steps": 1,
            "enable_progress_bar": True,
            "precision": 16,
            "sync_batchnorm": False,
            "enable_model_summary": False,
            "max_epochs": max_epochs,
            "accelerator": "gpu",
            "devices": -1
            # "strategy": DDPStrategy(find_unused_parameters=False),
        }
    )

    # noinspection SpellCheckingInspection
    net.hparams.lr = tune_lr(
        model=net,
        tuning_params={
            "mode": "exponential",
            "datamodule": data_module,
            "min_lr": 1e-08,
            "max_lr": 1.0
        },
        trainer_args={
            "callbacks": [
                StochasticWeightAveraging(swa_lrs=1e-2),
                EarlyStopping(
                    monitor="Validation-mIoU",
                    mode="max",
                    patience=100,
                    strict=True,
                    check_finite=True,
                    min_delta=1e-3,
                    check_on_train_epoch_end=False,
                )
            ],
            "accumulate_grad_batches": 1,
            "check_val_every_n_epoch": 10,
            "num_sanity_val_steps": 0,
            "detect_anomaly": False,
            "log_every_n_steps": 1,
            "enable_progress_bar": True,
            "precision": 32,
            "sync_batchnorm": False,
            "enable_model_summary": False,
            "max_epochs": max_epochs,
            "accelerator": "gpu",
            "devices": -1,
            # "strategy": DDPStrategy(find_unused_parameters=False),
        }
    )

    trainer = Trainer(
        logger=TensorBoardLogger(save_dir="logs", name='FloodNet'),
        callbacks=[
            StochasticWeightAveraging(swa_epoch_start=0.1, swa_lrs=1e-2),
            RichProgressBar(),
            ShowMetric(),
            LogConfusionMatrix(),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename=(
                    "FloodNet-" +
                    "{epoch}-" +
                    "{Validation-Min_Loss:.3f}"
                ),
                monitor='Validation-Min_Loss',
                save_top_k=5,
                save_last=True,
                save_on_train_epoch_end=False
            ),
            EarlyStopping(
                monitor="Validation-Min_Loss",
                mode="min",
                patience=50,
                strict=True,
                check_finite=True,
                min_delta=1e-3,
                check_on_train_epoch_end=False,
            )
        ],
        accumulate_grad_batches=2,
        check_val_every_n_epoch=20,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        log_every_n_steps=2,
        enable_progress_bar=True,
        precision=16,
        strategy=DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=False,
        enable_model_summary=False,
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=-1
    )

    # Training
    trainer.fit(model=net, datamodule=data_module)
