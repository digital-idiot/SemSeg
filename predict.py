import warnings
import argparse
import numpy as np
from pathlib import Path
from pytorch_lightning import Trainer
from core.models import TopFormerModel
from core.igniter import LightningSemSeg
from helper.callbacks import PredictionWriter
from data_factory.utils import image_to_tensor
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import OneCycleLR
from data_factory.dataset import DatasetConfigurator
from data_factory.data_module import IgniteDataModule
from data_factory.dataset import ReadableImageDataset
from pytorch_lightning.callbacks import RichProgressBar


warnings.filterwarnings('error', category=UserWarning)

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

model = TopFormerModel(
    num_classes=10,
    config_alias='B',
    input_channels=3,
    injection_type='dot_sum',
    fusion='sum'
)

predict_dataset = DatasetConfigurator(
    conf_path="Data/FloodNetData/Test/Test.json"
).generate_image_dataset(
    transform=None,
    target_shape=image_shape,
    pad_aspect='symmetric',
    resampling=0,
    channels=(1, 2, 3),
    tensor_maker=image_to_tensor
)

predict_writer = predict_dataset.writable_clone(
    dst_dir='Predictions',
    overlay_dir='Overlays',
    color_table={
        0: (0, 0, 0, 0),
        1: (31, 119, 180, 255),
        2: (174, 199, 232, 255),
        3: (255, 127, 14, 255),
        4: (255, 187, 120, 255),
        5: (44, 160, 44, 255),
        6: (152, 223, 138, 255),
        7: (214, 39, 40, 255),
        8: (255, 152, 150, 255),
        9: (148, 103, 189, 255),
        10: (197, 176, 213, 255)
    },
    boundary_color=(1, 1, 1, 255),
    overlay_transparency=150,
    dtype=np.uint8,
    count=1,
    driver='GTiff',
    height=image_shape[0],
    width=image_shape[1],
    nodata=0
)

data_module = IgniteDataModule(
    predict_dataset=predict_dataset,
    num_workers=8,
    batch_size=1,
    shuffle=False,
    collate_fn=ReadableImageDataset.collate
)

assert last_checkpoint.is_file(), "Previous checkpoint does not exists!"

net = LightningSemSeg.load_from_checkpoint(
    last_checkpoint, model=model
)

trainer = Trainer(
    callbacks=[
        RichProgressBar(),
        PredictionWriter(
            writable_datasets=[predict_writer],
            overwrite=True
        )
    ],
    num_sanity_val_steps=0,
    detect_anomaly=False,
    enable_progress_bar=True,
    precision=16,
    sync_batchnorm=False,
    enable_model_summary=False,
    max_epochs=-1,
    accelerator="gpu",
    devices=1
)

trainer.predict(model=net, datamodule=data_module)
