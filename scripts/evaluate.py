import json
import numpy as np
import rasterio as rio
from typing import Union
from pathlib import Path
from torchmetrics import Dice
from torchmetrics import Accuracy
from torchmetrics import JaccardIndex
from torchmetrics import ConfusionMatrix


# TODO

class ImageRepo(object):
    def __init__(
            self, config: Union[str, Path],
            gt_dir: Union[str, Path],
            pr_dir: Union[str, Path],
            num_classes: int
    ):
        config = Path(config)
        gt_dir = Path(gt_dir)
        pr_dir = Path(pr_dir)
        assert config.is_file(), "Config file does not exist!"
        assert gt_dir.is_dir(), "Ground truth directory does not exists!"
        assert pr_dir.is_dir(), "Prediction directory does not exists!"
        with open(config, 'r') as src:
            conf = json.load(fp=src)
        micro_acc = Accuracy(num_classes=num_classes, average='micro')
        # noinspection SpellCheckingInspection
        micro_acc = Accuracy(num_classes=num_classes, average='macro')
        sample_acc = Accuracy(num_classes=num_classes, average='sample')
        class_acc = Accuracy(num_classes=num_classes, average='none')

        for k in conf['keys']:
            name = (
                    f"{conf['label_prefix']}{k}{conf['label_prefix']}." +
                    f"{conf['label_extension']}"
            )
            gt_path = gt_dir / name
            pr_path = pr_dir / name
            with rio.open(gt_path, 'r') as gt, rio.open(pr_path, 'r') as pr:
                gt_img = gt.read(indexes=1, masked=True)
                pr_img = pr.read(indexes=1, masked=True)
                mask = np.logical_and(gt_img.mask, pr_img.mask)
                gt_pixels = gt_img[~mask]
                pr_pixels = pr_img[~mask]


