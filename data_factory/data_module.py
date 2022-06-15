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
            shuffle_flag=self.shuffle,
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
                    shuffle_flag=self.shuffle,
                    collate=self.collate_fn
                )
                for ds in self.test_dataset
            ]
        return dataloader(
            ds=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_flag=self.shuffle,
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
                    shuffle_flag=self.shuffle,
                    collate=self.collate_fn
                )
                for ds in self.predict_dataset
            ]
        return dataloader(
            ds=self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_flag=self.shuffle,
            collate=self.collate_fn
        )
