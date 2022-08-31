import json
import warnings
import numpy as np
import pandas as pd
import rasterio as rio
from typing import Dict
from typing import Union
from pathlib import Path
from rich.progress import track
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from rasterio.errors import NotGeoreferencedWarning


class Evaluator(object):
    def __init__(
            self, config: Union[str, Path],
            gt_dir: Union[str, Path],
            pr_dir: Union[str, Path],
            class_labels: Dict[int, str]
    ):
        config = Path(config)
        gt_dir = Path(gt_dir)
        pr_dir = Path(pr_dir)
        assert config.is_file(), "Config file does not exist!"
        assert gt_dir.is_dir(), "Ground truth directory does not exists!"
        assert pr_dir.is_dir(), "Prediction directory does not exists!"
        label_names = list(class_labels.values())
        label_indices = list(class_labels.keys())
        with open(config, 'r') as src:
            conf = json.load(fp=src)
        image_pairs = dict()
        for k in conf['keys']:
            name = (
                    f"{conf['label_prefix']}{k}{conf['label_prefix']}." +
                    f"{conf['label_extension']}"
            )
            gt_path = gt_dir / name
            pr_path = pr_dir / name
            image_pairs[k] = {'gt': gt_path, 'pr': pr_path}
        self.image_pairs = image_pairs
        self.label_names = label_names
        self.label_indices = label_indices

    def make_report(self, report_path: Union[str, Path]):
        label_names = self.label_names
        label_indices = self.label_indices
        df_f1 = pd.DataFrame(
            columns=label_names
        )
        df_iou = pd.DataFrame(
            columns=label_names
        )
        df_acc = pd.DataFrame(
            columns=label_names
        )
        df_k = pd.DataFrame(columns=['Kappa'])
        df_cm = 0
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=NotGeoreferencedWarning
            )
            for k, path_dict in track(tuple(self.image_pairs.items())):
                gt_path = path_dict['gt']
                pr_path = path_dict['pr']
                with rio.open(gt_path, 'r') as gt, rio.open(pr_path, 'r') as pr:
                    pr_shape = np.array([pr.height, pr.width])
                    gt_shape = np.array([gt.height, gt.width])
                    min_idx = np.argmin(gt_shape)
                    max_idx = np.argmax(gt_shape)
                    gt_shape[min_idx] = round(
                        (
                            (
                                gt_shape[min_idx] *
                                pr_shape[max_idx]
                            ) / gt_shape[max_idx]
                        )
                    )
                    gt_shape[max_idx] = pr_shape[max_idx]
                    gt_arr = gt.read(
                        indexes=1,
                        masked=True,
                        out_shape=gt_shape.tolist(),
                        resampling=0
                    )
                    diff = pr_shape - gt_shape
                    if np.any(a=(diff > 0)):
                        pad_a = diff // 2
                        pad_b = diff - pad_a
                        pad_widths = [
                            (i, j)
                            for i, j in zip(pad_a.tolist(), pad_b.tolist())
                        ]
                        if gt_arr.ndim > len(pad_widths):
                            pad_widths = (
                                [(0, 0)] * (gt_arr.ndim - len(pad_widths))
                            ) + pad_widths
                        nodata = gt_arr.fill_value
                        gt_mask = np.pad(
                            array=gt_arr.mask,
                            pad_width=pad_widths,
                            mode='constant',
                            constant_values=True
                        )
                        gt_arr = np.pad(
                            array=gt_arr.filled(),
                            pad_width=pad_widths,
                            mode='constant',
                            constant_values=nodata
                        )
                        gt_arr = np.ma.masked_array(
                            data=gt_arr, mask=gt_mask, fill_value=nodata
                        )
                    pr_arr = pr.read(indexes=1, masked=True)
                valid_mask = np.logical_not(
                    np.logical_and(gt_arr.mask, pr_arr.mask)
                )
                gt_arr = gt_arr[valid_mask]
                pr_arr = pr_arr[valid_mask]
                df_f1.loc[k] = f1_score(
                    y_true=gt_arr,
                    y_pred=pr_arr,
                    labels=label_indices,
                    average=None
                )
                df_iou.loc[k] = jaccard_score(
                    y_true=gt_arr,
                    y_pred=pr_arr,
                    labels=label_indices,
                    average=None
                )
                df_cm = df_cm + confusion_matrix(
                    y_true=gt_arr,
                    y_pred=pr_arr,
                    labels=label_indices,
                    normalize='true'
                )
                # noinspection PyUnresolvedReferences
                df_acc.loc[k] = df_cm.diagonal().copy()
                df_k.loc[k] = cohen_kappa_score(
                    y1=gt_arr,
                    y2=pr_arr,
                    labels=label_indices
                )
        df_cm = pd.DataFrame(
            data=df_cm,
            index=label_names,
            columns=label_names
        )
        with pd.ExcelWriter(
                path=report_path,
                engine='odf',
                mode='w',
                if_sheet_exists='overlay'
        ) as writer:
            df_acc.to_excel(
                excel_writer=writer,
                sheet_name='Accuracy',
                header=True,
                index=True,
                index_label='Sample'
            )
            df_f1.to_excel(
                excel_writer=writer,
                sheet_name='F1',
                header=True,
                index=True,
                index_label='Sample'
            )
            df_iou.to_excel(
                excel_writer=writer,
                sheet_name='IoU',
                header=True,
                index=True,
                index_label='Sample'
            )
            df_k.to_excel(
                excel_writer=writer,
                sheet_name='Kappa',
                header=True,
                index=True,
                index_label='Sample'
            )
            df_cm.to_excel(
                excel_writer=writer,
                sheet_name='F1',
                header=True,
                index=True,
                index_label='Sample'
            )
