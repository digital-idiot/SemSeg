import torch
import warnings
import argparse
from pathlib import Path
from pytorch_lightning import Trainer
from helper.assist import WrappedLoss
from core.models import TopFormerModel
from helper.callbacks import ShowMetric
from core.igniter import LightningSemSeg
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
    parser = argparse.ArgumentParser(description='Retrain')
    parser.add_argument(
        '-c', '--checkpoint',
        metavar='Previous Checkpoint Path',
        action='store',
        type=str,
        required=True,
        dest='checkpoint_path',
        help='Specify Previous Checkpoint Path'
    )
    args = parser.parse_args()
    last_checkpoint = Path(args.checkpoint_path)
    checkpoint_dir = Path("checkpoints")
    image_shape = (768, 1024)
    max_epochs = 30
    model = TopFormerModel(
        num_classes=10,
        config_alias='B',
        input_channels=3,
        injection_type='dot_sum',
        fusion='sum',
        head_cfg=frozenset({'alias': 'refiner'}.items())
    )

    loss_function = WrappedLoss(
        loss_fn=OhemCrossEntropyLoss(
            ignore_index=0,
            threshold=0.7,
            min_kept=0.5
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

    # predict_writer = predict_dataset.writable_clone(
    #     dst_dir='Predictions',
    #     overlay_dir='Overlays',
    #     color_table={
    #         0: (0, 0, 0, 0),
    #         1: (31, 119, 180, 255),
    #         2: (174, 199, 232, 255),
    #         3: (255, 127, 14, 255),
    #         4: (255, 187, 120, 255),
    #         5: (44, 160, 44, 255),
    #         6: (152, 223, 138, 255),
    #         7: (214, 39, 40, 255),
    #         8: (255, 152, 150, 255),
    #         9: (148, 103, 189, 255),
    #         10: (197, 176, 213, 255)
    #     },
    #     boundary_color=(1, 1, 1, 255),
    #     overlay_transparency=150,
    #     dtype=np.uint8,
    #     count=1,
    #     driver='PNG',
    #     height=image_shape[0],
    #     width=image_shape[1],
    #     nodata=0
    # )
    data_module = IgniteDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        predict_dataset=None,
        num_workers=8,
        batch_size=8,
        shuffle=True,
        collate_fn=ReadableImagePairDataset.collate
    )

    assert last_checkpoint.is_file(), "Previous checkpoint does not exists!"
    max_lr = 0.5 * torch.load(str(last_checkpoint))['hyper_parameters']['lr']

    scheduler = WrappedScheduler(
        scheduler=OneCycleLR,
        max_lr=max_lr,
        steps_per_epoch=len(train_dataset),
        epochs=max_epochs
    )
    net = LightningSemSeg.load_from_checkpoint(
        last_checkpoint, model=model, scheduler=scheduler
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
                    "{val_loss:.3f}"
                ),
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
        accumulate_grad_batches=2,
        check_val_every_n_epoch=10,
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
