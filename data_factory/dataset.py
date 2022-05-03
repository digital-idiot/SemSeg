import json
import torch
import logging
import warnings
import numpy as np
import rasterio as rio
from pathlib import Path
from affine import Affine
from click import FileError
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
from data_factory.transforms import TransformPair
from rasterio.errors import NotGeoreferencedWarning
from typing import Tuple, List, Union, Callable, Dict, Sequence, Any


def to_tensor(arr: np.ndarray):
    arr = torch.from_numpy(arr).contiguous()
    if isinstance(arr, torch.ByteTensor):
        return arr.to(dtype=torch.get_default_dtype()).div(255)
    else:
        return arr


class ReadableImageDataset(Dataset):
    def __init__(
            self,
            path_list: Union[Tuple[Union[str, Path]], List[Union[str, Path]]],
            transform: Callable = None,
            channels: Union[int, Tuple[int], List[int]] = None
    ):
        super(ReadableImageDataset, self).__init__()
        self.path_list = [
            p if isinstance(p, Path) else Path(p) for p in path_list
        ]
        self.transform = transform
        self.existence_list = [path.is_file() for path in path_list]
        self.channels = channels

    def readable(self, idx):
        assert 0 <= idx < len(self), f'Index ({idx}) is out of bound!'
        return self.existence_list[idx]

    def all_readable(self):
        return all(self.existence_list)

    def missing_files(self):
        if self.all_readable():
            return None
        else:
            return [
                self.path_list[i]
                for i, flag in enumerate(self.existence_list) if not flag
            ]

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        if self.existence_list[index]:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=NotGeoreferencedWarning
                )
                with rio.open(self.path_list[index], 'r') as src:
                    arr = to_tensor(src.read(indexes=self.channels))
                    if self.transform:
                        arr = self.transform(arr)
                    return arr
        else:
            raise FileNotFoundError(
                f"File don't exist: {self.path_list[index]}"
            )

    def get_meta(self, idx):
        assert 0 <= idx < len(self), f'Index ({idx}) is out of bound!'
        if self.existence_list[idx]:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=NotGeoreferencedWarning
                )
                with rio.open(self.path_list[idx], 'r') as src:
                    meta = src.meta.copy()
            return meta
        else:
            return None

    def get_metas(self):
        return [self.get_meta(idx=i) for i in range(len(self))]

    def get_filename(self, idx):
        assert 0 <= idx < len(self), f'Index ({idx}) is out of bound!'
        return self.path_list[idx].name

    def get_filenames(self):
        return [self.get_filename(i) for i in range(len(self))]

    def get_shape(self, idx):
        meta = self.get_meta(idx)
        if meta is None:
            return None
        return meta['count'], meta['height'], meta['width']

    def get_shapes(self):
        return [self.get_shape(i) for i in range(len(self))]

    def stackable(self):
        return len(set(self.get_shapes())) == 1

    def get_paths(self):
        return self.path_list

    def get_transform(self):
        return self.transform

    def writable_clone(
            self,
            dst_dir: Union[Union[str, Path], Union[Sequence[Union[str, Path]]]]
    ):
        src_filenames = self.get_filenames()
        if isinstance(dst_dir, str):
            dir_list = [Path(dst_dir)] * len(src_filenames)
        elif isinstance(dst_dir, Path):
            dir_list = [dst_dir] * len(src_filenames)
        else:
            assert (
                isinstance(dst_dir, Sequence) and
                (len(dst_dir) == len(src_filenames))
            ), (
                f'{len(dst_dir)} directories can not be broadcast ' +
                f'with {len(src_filenames)} files!'
            )
            dir_list = list()
            for d in dst_dir:
                if isinstance(d, str):
                    d = Path(d)
                assert isinstance(d, Path), (
                    f"Not a 'str' ot 'pathlib.Path' instance: {d}"
                )
                dir_list.append(d)
        file_paths = list()
        for p_dir, name in zip(dir_list, src_filenames):
            if not(p_dir.is_dir()):
                assert not(p_dir.is_file()), (
                    f'File exists instead of directory: {p_dir}'
                )
                p_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            file_paths.append(p_dir)

        meta_list = self.get_metas()
        return WriteableImageDataset(path_list=file_paths, meta_list=meta_list)

    def split(self, ratios: Sequence[int], random: bool = True):
        if not isinstance(ratios, np.ndarray):
            ratios = np.array(ratios)
        indexes = np.arange(len(self))
        if random:
            np.random.shuffle(indexes)
        split_sizes = np.around(
            len(self) * (ratios / ratios.sum()), decimals=0
        ).astype(int)
        diff = len(self) - split_sizes.sum()
        sorted_part_indexes = np.argsort(split_sizes)
        if diff > 0:
            split_sizes[sorted_part_indexes[:diff]] += 1
        else:
            split_sizes[sorted_part_indexes[diff:]] -= 1
        if np.any(split_sizes == 0):
            raise RuntimeWarning(
                "All partitions could not be accommodated!"
            )
        split_stops = np.cumsum(split_sizes).tolist()
        split_starts = 0, * split_stops[:-1]
        return [
            ReadableImageDataset(
                path_list=self.path_list[indexes[i:j]],
                transform=self.transform,
                channels=self.channels
            )
            for i, j in zip(split_starts, split_stops)
        ]

    @classmethod
    def collate(cls, samples):
        return torch.stack(samples, dim=0)


class WritableDataset(object, metaclass=ABCMeta):

    @abstractmethod
    def write(self, idx: int, data: Any, overwrite: bool = False):
        raise NotImplementedError("'write' method has not been implemented!")

    @abstractmethod
    def write_batch(
            self,
            data_batch: Any,
            indexes: Sequence[int],
            overwrite: bool = False
    ):
        raise NotImplementedError(
            "'write_batch' method has not been implemented!"
        )


class WriteableImageDataset(WritableDataset):
    def __init__(
            self,
            path_list: Union[Tuple[Union[str, Path]], List[Union[str, Path]]],
            meta_list: Union[Dict, List[Dict]] = None
    ):
        base_meta = {
            'driver': 'GTiff',
            'crs': None,
            'nodata': None,
            'transform': Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }
        if meta_list is None:
            meta_list = [base_meta] * len(path_list)
        else:
            for meta in meta_list:
                for k in base_meta.keys():
                    if not(k in meta.keys()):
                        meta[k] = base_meta[k]
        assert (
                len(meta_list) == len(path_list)
        ), (
            f"Number of files is {len(path_list)}, " +
            f"while number of metadata objects is {len(meta_list)}"
        )
        self.path_list = [
            p if isinstance(p, Path) else Path(p) for p in path_list
        ]
        self.meta_list = meta_list

    def write(self, idx: int, data: np.ndarray, overwrite: bool = False):
        dst_path = self.path_list[idx]
        dst_meta = self.meta_list[idx]
        assert data.ndim == 3, (
            f'expected a 3D (C, H, W) array, ' +
            f'received an {data.ndim} dimensional array instead!'
        )
        if not('count' in dst_meta.keys()):
            dst_meta['count'] = data.shape[0]
        if not('height' in dst_meta.keys()):
            dst_meta['height'] = data.shape[1]
        if not('width' in dst_meta.keys()):
            dst_meta['width'] = data.shape[2]
        # noinspection SpellCheckingInspection
        if not('dtype' in dst_meta.keys()):
            # noinspection SpellCheckingInspection
            dst_meta['dtype'] = data.dtype
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=NotGeoreferencedWarning
            )
            if not overwrite and dst_path.is_file():
                raise FileExistsError(f"File already exists: {str(dst_path)}")
            try:
                with rio.open(dst_path, 'w', **dst_meta) as dst:
                    dst.write(data)
                return True
            except rio.errors.RasterioError as rio_error:
                logging.error(f'RasterioError: {rio_error}')
                return False
            except rio.errors.CRSError as crs_error:
                logging.error(f'CRSError: {crs_error}')
                return False
            except ValueError as val_error:
                logging.error(f'ValueError: {val_error}')
                return False
            except FileError as file_error:
                logging.error(f'FileError: {file_error}')
                return False
            except OSError as os_error:
                logging.error(f'OSError: {os_error}')
                return False

    def write_batch(
            self,
            data_batch: np.ndarray,
            indexes: Sequence[int],
            overwrite: bool = False
    ) -> list:
        assert data_batch.ndim == 4, (
                f'expected a 4D (N, C, H, W) array, ' +
                f'received an {data_batch.ndim} dimensional array instead!'
        )
        assert data_batch.shape[0] == len(indexes), (
            f"Batch size is {data_batch.shape[0]}, " +
            f"while there are {len(indexes)} indexes"
        )
        status_list = list()
        for i in range(len(indexes)):
            status = self.write(
                idx=indexes[i],
                data=data_batch[i, :, :, :],
                overwrite=overwrite
            )
            status_list.append(status)
        return status_list


class DatasetConfigurator(object):
    def __init__(self, conf_path: Union[str, Path]):
        if isinstance(conf_path, str):
            conf_path = Path(conf_path)
        assert isinstance(conf_path, Path), (
            "'conf_path' is not a valid string or Path like object"
        )
        with open(conf_path, 'r') as src:
            data_map = json.load(src)

        keys = data_map['keys']

        if data_map["image_directory"]:
            if data_map['relative_path']:
                img_dir = conf_path.parent / data_map["image_directory"]
            else:
                img_dir = Path(data_map["image_directory"])
            img_list = [
                (img_dir / '.'.join([stem, data_map['file_extension']]))
                for stem in keys
            ]
        else:
            img_list = None

        if data_map["label_directory"]:
            if data_map['relative_path']:
                lbl_dir = conf_path.parent / data_map["label_directory"]
            else:
                lbl_dir = Path(data_map["label_directory"])
            lbl_list = [
                (lbl_dir / '.'.join([stem, data_map['file_extension']]))
                for stem in keys
            ]
        else:
            lbl_list = None

        self.image_list = img_list
        self.label_list = lbl_list

    def generate_paired_dataset(
            self,
            image_channels: Union[int, List[int], Tuple[int]] = None,
            label_channels: Union[int, List[int], Tuple[int]] = 1,
            transform: Callable = None,
    ):
        if (self.image_list is not None) and (self.label_list is not None):
            return ReadableImagePairDataset(
                image_list=self.image_list,
                label_list=self.label_list,
                image_channels=image_channels,
                label_channels=label_channels,
                transform=transform,
            )
        else:
            raise AssertionError(
                "Image set and/or label set is not available!"
            )

    def generate_image_dataset(
            self,
            transform: Callable = None,
            channels: Union[int, Tuple[int], List[int]] = None,
    ):
        if self.image_list is not None:
            return ReadableImageDataset(
                path_list=self.image_list,
                transform=transform,
                channels=channels
            )
        else:
            raise AssertionError(
                "Image set is not available!"
            )

    def generate_label_dataset(
            self,
            transform: Callable = None,
            channels: Union[int, Tuple[int], List[int]] = None,
    ):
        if self.label_list is not None:
            return ReadableImageDataset(
                path_list=self.label_list,
                transform=transform,
                channels=channels
            )
        else:
            raise AssertionError(
                "Label set is not available!"
            )


class ReadableImagePairDataset(Dataset):
    def __init__(
            self,
            image_list: Union[Tuple[str], List[Path]],
            label_list: Union[Tuple[str], List[Path]],
            image_channels: Union[int, List[int], Tuple[int]] = None,
            label_channels: Union[int, List[int], Tuple[int]] = 1,
            transform: TransformPair = None,
    ):
        """

        Args:
            image_list:
            label_list:
            image_channels:
            label_channels:
            transform:
        """
        assert len(image_list) == len(label_list), (
            f"Length mismatch, {len(image_list)} images provided " +
            f"with {len(label_list)} labels"
        )
        self.size = len(image_list)
        self.image_channels = image_channels
        self.label_channels = label_channels
        self.img_ds = ReadableImageDataset(
            path_list=image_list,
            transform=None,
            channels=image_channels
        )
        self.lbl_ds = ReadableImageDataset(
            path_list=label_list,
            transform=None,
            channels=label_channels
        )
        if transform:
            assert isinstance(transform, TransformPair), (
                "'transform' is not a valid TransformPair object"
            )
        self.transform = transform

    def all_readable(self):
        return (
            self.img_ds.all_readable() and self.lbl_ds.all_readable()
        )

    def all_stackable(self):
        return self.img_ds.stackable() and self.lbl_ds.stackable()

    def all_shapes(self):
        return self.img_ds.get_shapes(), self.lbl_ds.get_shapes()

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        img = self.img_ds[index]
        lbl = self.lbl_ds[index]
        if self.transform:
            image_transform, label_transform = self.transform()
            img = image_transform(img)
            lbl = label_transform(lbl)
        return img, lbl

    def image_dataset(self):
        return self.img_ds

    def label_dataset(self):
        return self.lbl_ds

    def split(self, ratios: Sequence[int], random: bool = True):
        if not isinstance(ratios, np.ndarray):
            ratios = np.array(ratios)
        indexes = np.arange(len(self))
        if random:
            np.random.shuffle(indexes)
        split_sizes = np.around(
            len(self) * (ratios / ratios.sum()), decimals=0
        ).astype(int)
        diff = len(self) - split_sizes.sum()
        sorted_part_indexes = np.argsort(split_sizes)
        if diff > 0:
            split_sizes[sorted_part_indexes[:diff]] += 1
        else:
            split_sizes[sorted_part_indexes[diff:]] -= 1
        if np.any(split_sizes == 0):
            raise RuntimeWarning(
                "All partitions could not be accommodated!"
            )
        split_stops = np.cumsum(split_sizes).tolist()
        split_starts = 0, * split_stops[:-1]
        return [
            ReadableImagePairDataset(
                image_list=self.img_ds.get_paths()[indexes[i:j]],
                label_list=self.lbl_ds.get_paths()[indexes[i:j]],
                transform=self.transform,
                image_channels=self.image_channels,
                label_channels=self.label_channels
            )
            for i, j in zip(split_starts, split_stops)
        ]

    @classmethod
    def collate(cls, samples: Sequence):
        images, labels = list(zip(*samples))
        image_batch = ReadableImageDataset.collate(samples=images)
        label_batch = ReadableImageDataset.collate(samples=labels)
        return image_batch, label_batch
