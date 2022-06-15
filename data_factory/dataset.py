import json
import torch
import logging
import warnings
import numpy as np
import rasterio as rio
from abc import ABCMeta
from pathlib import Path
from affine import Affine
from click import FileError
from abc import abstractmethod
from operator import itemgetter
from torch.utils.data import Dataset
from data_factory.utils import image_to_tensor
from data_factory.utils import label_to_tensor
from data_factory.transforms import TransformPair
from rasterio.errors import NotGeoreferencedWarning
from typing import (
    Union,
    Callable,
    Dict,
    Sequence,
    Any
)


class ReadableImageDataset(Dataset):
    def __init__(
            self,
            path_list: Sequence[Union[str, Path]],
            target_shape: Sequence[int] = None,
            pad_aspect: str = None,
            resampling: int = 0,
            transform: Callable = None,
            channels: Union[int, Sequence[int]] = None,
            converter: Callable = label_to_tensor
    ):
        super(ReadableImageDataset, self).__init__()
        self.path_list = [
            p if isinstance(p, Path) else Path(p) for p in path_list
        ]
        self.transform = transform
        self.existence_list = [path.is_file() for path in path_list]
        self.channels = channels
        if converter:
            assert isinstance(converter, Callable), (
                "'tensor_maker' is not Callable"
            )
        assert isinstance(
            converter, Callable
        ), "Specified 'converter' isn't callable!"
        self.tensor_maker = converter
        if isinstance(target_shape, int):
            target_shape = (target_shape, target_shape)
        assert isinstance(
            target_shape, Sequence
        ) and (len(target_shape) == 2) and all(
            [isinstance(x, int) and x > 0 for x in target_shape]
        ), f"Invalid 'target_shape': {target_shape}"
        self.target_shape = tuple(target_shape)

        if pad_aspect:
            assert isinstance(pad_aspect, str) and pad_aspect in {
                'symmetric', 'reflect', 'edge', 'nodata'

            }, (
                f"Invalid padding: {pad_aspect}"
            )
        self.pad_aspect = pad_aspect
        assert isinstance(resampling, int) and (
                resampling in set(range(15))
        ), f"Unknown resampling: {resampling}"
        self.resampling = resampling

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
                    if self.target_shape:
                        if self.pad_aspect:
                            img_shape = [src.height, src.width]
                            min_idx = np.argmin(img_shape)
                            max_idx = np.argmax(img_shape)
                            img_shape[min_idx] = round(
                                (
                                    (
                                        img_shape[min_idx] *
                                        self.target_shape[max_idx]
                                    ) / img_shape[max_idx]
                                )
                            )
                            img_shape[max_idx] = self.target_shape[max_idx]
                            out_shape = tuple(img_shape)
                        else:
                            out_shape = self.target_shape
                    arr = src.read(
                        indexes=self.channels,
                        out_shape=out_shape,
                        resampling=self.resampling
                    )
                    diff = np.array(self.target_shape) - np.array(out_shape)
                    if np.any(a=(diff > 0)):
                        pad_a = diff // 2
                        pad_b = diff - pad_a
                        pad_widths = [
                            (i, j)
                            for i, j in zip(pad_a.tolist(), pad_b.tolist())
                        ]
                        if arr.ndim > len(pad_widths):
                            pad_widths = (
                                 [(0, 0), ] * (arr.ndim - len(pad_widths))
                            ) + pad_widths
                        conf = dict()
                        if self.pad_aspect == 'nodata':
                            conf['mode'] = 'constant'
                            conf['constant_values'] = src.nodata
                        elif isinstance(self.pad_aspect, (int, float, complex)):
                            conf['mode'] = 'constant'
                            conf['constant_values'] = self.pad_aspect
                        else:
                            conf['mode'] = self.pad_aspect
                        arr = np.pad(
                            array=arr,
                            pad_width=pad_widths,
                            **conf
                        )
                    if self.tensor_maker:
                        arr = self.tensor_maker(arr)
                    if self.transform:
                        arr = self.transform(arr)
                    return torch.unsqueeze(input=arr, dim=0)
        else:
            raise FileNotFoundError(
                f"File doesn't exist: {self.path_list[index]}"
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
            dst_dir: Union[
                Union[str, Path], Union[Sequence[Union[str, Path]]]
            ]
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
                p_dir.mkdir(
                    mode=0o755, parents=True, exist_ok=True
                )
            file_paths.append(p_dir)

        meta_list = self.get_metas()
        return WriteableImageDataset(
            path_list=file_paths, meta_list=meta_list
        )

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
        parts = list()
        for i, j in zip(split_starts, split_stops):
            get_subsample = itemgetter(*indexes[i:j])
            # noinspection PyTypeChecker
            parts.append(
                ReadableImageDataset(
                    path_list=get_subsample(self.get_paths()),
                    transform=self.transform,
                    channels=self.channels
                )
            )
        return parts

    @classmethod
    def collate(cls, samples):
        return torch.cat(tensors=samples, dim=0)


class WritableDataset(object, metaclass=ABCMeta):

    @abstractmethod
    def write(self, idx: int, data: Any, overwrite: bool = False):
        raise NotImplementedError(
            "'write' method has not been implemented!"
        )

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
            path_list: Sequence[Union[str, Path]],
            meta_list: Union[Dict, Sequence[Dict]] = None
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

    def write(
            self,
            idx: int,
            data: np.ndarray,
            overwrite: bool = False
    ):
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
                raise FileExistsError(
                    f"File already exists: {str(dst_path)}"
                )
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


class ReadableImagePairDataset(Dataset):
    def __init__(
            self,
            image_list: Sequence[Union[str, Path]],
            label_list: Sequence[Union[str, Path]],
            target_shape: Sequence[int] = None,
            pad_aspect: str = None,
            image_resampling: int = 0,
            label_resampling: int = 0,
            image_channels: Union[int, Sequence[int]] = None,
            label_channels: Union[int, Sequence[int]] = 1,
            image_maker: Callable = image_to_tensor,
            label_maker: Callable = label_to_tensor,
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
            target_shape=target_shape,
            pad_aspect=pad_aspect,
            resampling=image_resampling,
            transform=None,
            channels=image_channels,
            converter=image_maker
        )
        self.lbl_ds = ReadableImageDataset(
            path_list=label_list,
            target_shape=target_shape,
            pad_aspect=pad_aspect,
            resampling=label_resampling,
            transform=None,
            channels=label_channels,
            converter=label_maker
        )
        if transform:
            assert isinstance(transform, TransformPair), (
                "'transform' is not a valid TransformPair object"
            )
        self.transform = transform

        if isinstance(target_shape, int):
            target_shape = (target_shape, target_shape)
        assert isinstance(
            target_shape, Sequence
        ) and (len(target_shape) == 2) and all(
            [isinstance(x, int) and x > 0 for x in target_shape]
        ), f"Invalid 'target_shape': {target_shape}"
        self.target_shape = target_shape

        if pad_aspect:
            assert isinstance(pad_aspect, str) and pad_aspect in {
                'symmetric', 'reflect', 'edge', 'nodata'

            }, (
                f"Invalid padding: {pad_aspect}"
            )
        self.pad_aspect = pad_aspect

        assert isinstance(image_resampling, int) and (
                image_resampling in set(range(15))
        ), f"Unknown resampling: {image_resampling}"
        self.image_resampling = image_resampling

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
            image_transform, label_transform = next(self.transform())
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
        parts = list()
        for i, j in zip(split_starts, split_stops):
            get_subsample = itemgetter(*indexes[i:j])
            # noinspection PyTypeChecker
            parts.append(
                ReadableImagePairDataset(
                    image_list=get_subsample(self.img_ds.get_paths()),
                    label_list=get_subsample(self.lbl_ds.get_paths()),
                    target_shape=self.target_shape,
                    pad_aspect=self.pad_aspect,
                    image_resampling=self.image_resampling,
                    transform=self.transform,
                    image_channels=self.image_channels,
                    label_channels=self.label_channels
                )
            )
        return parts

    @classmethod
    def collate(cls, samples: Sequence):
        images, labels = list(zip(*samples))
        image_batch = ReadableImageDataset.collate(samples=images)
        label_batch = ReadableImageDataset.collate(samples=labels)
        return image_batch, label_batch


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
                (img_dir / '.'.join([stem, data_map['image_ext']]))
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
                (lbl_dir / '.'.join([stem, data_map['label_ext']]))
                for stem in keys
            ]
        else:
            lbl_list = None

        self.image_list = img_list
        self.label_list = lbl_list

    def generate_paired_dataset(
            self,
            image_channels: Union[int, Sequence[int]] = None,
            label_channels: Union[int, Sequence[int]] = 1,
            target_shape: Sequence[int] = None,
            pad_aspect: str = None,
            image_resampling: int = 0,
            label_resampling: int = 0,
            image_converter: Callable = image_to_tensor,
            label_converter: Callable = label_to_tensor,
            transform: TransformPair = None,
    ):
        if (self.image_list is not None) and (self.label_list is not None):
            return ReadableImagePairDataset(
                image_list=self.image_list,
                label_list=self.label_list,
                target_shape=target_shape,
                pad_aspect=pad_aspect,
                image_resampling=image_resampling,
                label_resampling=label_resampling,
                image_channels=image_channels,
                label_channels=label_channels,
                image_maker=image_converter,
                label_maker=label_converter,
                transform=transform,
            )
        else:
            raise AssertionError(
                "Image set and/or label set is not available!"
            )

    def generate_image_dataset(
            self,
            transform: Callable = None,
            target_shape: Sequence[int] = None,
            pad_aspect: str = None,
            resampling: int = 0,
            channels: Union[int, Sequence[int]] = None,
            tensor_maker: Callable = image_to_tensor
    ):
        if self.image_list is not None:
            return ReadableImageDataset(
                path_list=self.image_list,
                target_shape=target_shape,
                pad_aspect=pad_aspect,
                resampling=resampling,
                transform=transform,
                channels=channels,
                converter=tensor_maker
            )
        else:
            raise AssertionError(
                "Image set is not available!"
            )

    def generate_label_dataset(
            self,
            transform: Callable = None,
            target_shape: Sequence[int] = None,
            pad_aspect: str = None,
            resampling: int = 0,
            channels: Union[int, Sequence[int]] = None,
            tensor_maker: Callable = label_to_tensor
    ):
        if self.label_list is not None:
            return ReadableImageDataset(
                path_list=self.label_list,
                target_shape=target_shape,
                pad_aspect=pad_aspect,
                resampling=resampling,
                transform=transform,
                channels=channels,
                converter=tensor_maker
            )
        else:
            raise AssertionError(
                "Label set is not available!"
            )
