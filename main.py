import warnings
from core.dnet import DNet
from loss.seg_loss import dice_loss
from torch_optimizer import AdaBound
from pytorch_lightning import Trainer
from helper.assist import WrappedLoss
from helper.callbacks import ShowMetric
from core.igniter import LightningSemSeg
from helper.assist import WrappedOptimizer
from helper.callbacks import PredictionWriter
from helper.callbacks import LogConfusionMatrix
from data_factory.dataset import DatasetConfigurator
from data_factory.data_module import IgniteDataModule
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from data_factory.dataset import ReadableImagePairDataset
from data_factory.utils import image_to_tensor, label_to_tensor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

warnings.filterwarnings('error', category=UserWarning)


if __name__ == '__main__':
    # TODO: Read all parameters from a conf file
    model = DNet(
        image_channels=3,
        num_classes=8,
        embed_dim=256
    )
    optimizer = WrappedOptimizer(
        optimizer=AdaBound,
        lr=1e-3,
        final_lr=0.1,
        amsbound=True
    )
    loss_function = WrappedLoss(loss_fn=dice_loss, smooth=1.0)
    net = LightningSemSeg(
        model=model,
        optimizer=optimizer,
        criterion=loss_function
    )
    train_dataset = DatasetConfigurator(
        conf_path="Data/FloodNetData/Urban/Train/Train_DataConf.json"
    ).generate_paired_dataset(
        image_channels=(1, 2, 3),
        label_channels=1,
        transform=None,
        image_maker=image_to_tensor,
        label_maker=label_to_tensor
    )

    train_dataset, val_dataset = train_dataset.split(
        ratios=(8, 2),
        random=True
    )

    test_dataset = DatasetConfigurator(
        conf_path="Data/FloodNetData/Urban/Validation/Validation_DataConf.json"
    ).generate_paired_dataset(
        image_channels=(1, 2, 3),
        label_channels=1,
        transform=None,
        image_maker=image_to_tensor,
        label_maker=label_to_tensor
    )

    predict_dataset = DatasetConfigurator(
        conf_path="Data/FloodNetData/Urban/Test/Test_DataConf.json"
    ).generate_image_dataset(
        transform=None,
        channels=(1, 2, 3),
        tensor_maker=image_to_tensor
    )

    predict_writer = predict_dataset.writable_clone(dst_dir='Predictions')
    data_module = IgniteDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        test_dataset=test_dataset,
        predict_dataset=predict_dataset,
        num_workers=16,
        batch_size=25,
        shuffle=True,
        collate_fn=ReadableImagePairDataset.collate

    )
    trainer = Trainer(
        logger=TensorBoardLogger(save_dir="logs", name='FloodNet'),
        callbacks=[
            RichProgressBar(),
            ShowMetric(),
            LogConfusionMatrix(),
            PredictionWriter(writable_datasets=[predict_writer]),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename='FloodNet-{epoch}-{validation_loss:.3f}',
                monitor='Validation-Mean_Loss',
                save_top_k=2,
                save_last=True,
                save_on_train_epoch_end=False
            ),
            EarlyStopping(
                monitor="Validation-Mean_Loss",
                mode="min",
                patience=5,
                strict=True,
                check_finite=True,
                min_delta=1e-3,
                check_on_train_epoch_end=False,
            )
        ],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        log_every_n_steps=10,
        enable_progress_bar=True,
        precision=16,
        strategy=DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=False,
        enable_model_summary=False,
        max_epochs=100,
        accelerator="gpu",
        devices=-1
    )
    trainer.fit(model=net, datamodule=data_module)
