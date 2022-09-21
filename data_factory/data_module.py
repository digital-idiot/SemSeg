import random
from collections import deque
from torch.utils.data import Subset
from pytorch_lightning import LightningDataModule
from typing import Optional, Union, Sequence, Mapping, Callable
from torch.utils.data import Dataset, IterableDataset, DataLoader


def dataloader(
        ds: Dataset,
        batch_size: int = 1,
        num_workers: int = 1,
        shuffle_flag: bool = False,
        collate: Callable = None
) -> DataLoader:
    shuffle_flag &= not isinstance(ds, IterableDataset)
    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate
    )


class IgniteDataModule(LightningDataModule):
    # TODO: Implement all the methods along with some fancy stuff
    def __init__(
            self,
            train_dataset: Optional[
                Union[Dataset, Sequence[Dataset], Mapping[str, Dataset]]
            ] = None,
            val_dataset: Optional[
                Union[Dataset, Sequence[Dataset]]
            ] = None,
            test_dataset: Optional[
                Union[Dataset, Sequence[Dataset]]
            ] = None,
            predict_dataset: Optional[
                Union[Dataset, Sequence[Dataset]]
            ] = None,
            batch_size: int = 1,
            num_workers: int = 0,
            shuffle: bool = False,
            collate_fn: Callable = None
    ):
        super(IgniteDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        if isinstance(self.train_dataset, Mapping):
            return {
                key: dataloader(
                    ds=ds,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle_flag=self.shuffle,
                    collate=self.collate_fn
                )
                for key, ds in self.train_dataset.items()
            }
        if isinstance(self.train_dataset, Sequence):
            return [
                dataloader(
                    ds=ds,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle_flag=self.shuffle,
                    collate=self.collate_fn
                )
                for ds in self.train_dataset
            ]
        return dataloader(
            ds=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_flag=self.shuffle,
            collate=self.collate_fn
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        if isinstance(self.val_dataset, Sequence):
            return [
                dataloader(
                    ds=ds,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle_flag=False,
                    collate=self.collate_fn
                )
                for ds in self.val_dataset
            ]
        return dataloader(
            ds=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_flag=False,
            collate=self.collate_fn
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        if isinstance(self.test_dataset, Sequence):
            return [
                dataloader(
                    ds=ds,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle_flag=False,
                    collate=self.collate_fn
                )
                for ds in self.test_dataset
            ]
        return dataloader(
            ds=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_flag=False,
            collate=self.collate_fn
        )

    def predict_dataloader(self):
        if self.predict_dataset is None:
            return None
        if isinstance(self.predict_dataset, Sequence):
            return [
                dataloader(
                    ds=ds,
                    batch_size=1,
                    num_workers=self.num_workers,
                    shuffle_flag=False,
                    collate=self.collate_fn
                )
                for ds in self.predict_dataset
            ]
        return dataloader(
            ds=self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_flag=False,
            collate=self.collate_fn
        )


class FoldedIgniteDataModule(LightningDataModule):
    def __init__(
            self,
            dev_dataset: Dataset,
            k_folds: int,
            test_dataset: Dataset = None,
            predict_dataset: Dataset = None,
            batch_size: int = 1,
            num_workers: int = 0,
            shuffle: bool = False,
            collate_fn: Callable = None
    ):
        super(FoldedIgniteDataModule, self).__init__()
        if isinstance(k_folds, int)  and k_folds > 1:
            self._k_folds = k_folds
        elif 0 < k_folds <= 1:
            print('normal_split')
        else:
            raise ValueError(
                f'Illegal value for fold: {k_folds}'
            )
        self._dev_dataset = dev_dataset
        self._test_dataset = test_dataset
        self._predict_dataset = predict_dataset
        self._shuffle = bool(shuffle)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        size = len(self._dev_dataset)
        indexes = list(range(size))
        if self._shuffle:
            random.shuffle(indexes)
        if isinstance(self._k_folds, int) and self._k_folds > 1:
            fold_size = size // self._k_folds
            folds = deque()
            for start in range(0, size, fold_size):
                stop = min((start + fold_size), size)
                folds.append(
                    {
                        'train': (indexes[0:start] + indexes[stop:]),
                        'val': indexes[start:stop]
                    }
                )
        elif 0 < self._k_folds <= 1:
            train_size = int(size * self._k_folds)
            folds = deque()
            folds.append(
                {
                    'train': indexes[:train_size],
                    'val': indexes[train_size:]
                }
            )
        else:
            raise ValueError(
                f'Illegal value for fold: {self._k_folds}'
            )
        self._folds = folds
        self._k_folds = len(folds)
        self._i = 0

    def rotate(self):
        self._i = (self._i + 1) % self._k_folds

    def get_current(self, key: str):
        return self._folds[self._i][key]

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        return None

    def train_dataloader(self):
        return dataloader(
            ds=Subset(
                dataset=self._dev_dataset,
                indices=self.get_current(key='train')
            ),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle_flag=self._shuffle,
            collate=self._collate_fn
        )

    def val_dataloader(self):
        return dataloader(
            ds=Subset(
                dataset=self._dev_dataset,
                indices=self.get_current(key='val')
            ),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle_flag=False,
            collate=self._collate_fn
        )

    def test_dataloader(self):
        return dataloader(
            ds=self._test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle_flag=False,
            collate=self._collate_fn
        )

    def predict_dataloader(self):
        return dataloader(
            ds=self._predict_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle_flag=False,
            collate=self._collate_fn
        )
