from pytorch_lightning import LightningDataModule
from typing import Optional, Union, Sequence, Mapping, Callable
from torch.utils.data import Dataset, IterableDataset, DataLoader


class IgniteDataModule(LightningDataModule):
    # TODO: Implement all the methods along with some fancy stuff
    def __init__(
            self
    ):
        super(IgniteDataModule, self).__init__()

    @classmethod
    def from_datasets(
            cls,
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
        def dataloader(
                ds: Dataset,
                shuffle_flag: bool = False,
                collate: Callable = None
        ) -> DataLoader:
            shuffle_flag &= not isinstance(ds, IterableDataset)
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate
            )

        def train_dataloader():
            if isinstance(train_dataset, Mapping):
                return {
                    key: dataloader(
                        ds=ds,
                        shuffle_flag=shuffle,
                        collate=collate_fn
                    )
                    for key, ds in train_dataset.items()
                }
            if isinstance(train_dataset, Sequence):
                return [
                    dataloader(
                        ds=ds,
                        shuffle_flag=shuffle,
                        collate=collate_fn
                    )
                    for ds in train_dataset
                ]
            return dataloader(
                ds=train_dataset,
                shuffle_flag=shuffle,
                collate=collate_fn
            )

        def val_dataloader():
            if isinstance(val_dataset, Sequence):
                return [
                    dataloader(
                        ds=ds,
                        shuffle_flag=False,
                        collate=collate_fn
                    )
                    for ds in val_dataset
                ]
            return dataloader(
                ds=val_dataset,
                shuffle_flag=shuffle,
                collate=collate_fn
            )

        def test_dataloader():
            if isinstance(test_dataset, Sequence):
                return [
                    dataloader(
                        ds=ds,
                        shuffle_flag=False,
                        collate=collate_fn
                    )
                    for ds in test_dataset
                ]
            return dataloader(
                ds=test_dataset,
                shuffle_flag=shuffle,
                collate=collate_fn
            )

        def predict_dataloader():
            if isinstance(predict_dataset, Sequence):
                return [
                    dataloader(
                        ds=ds,
                        shuffle_flag=False,
                        collate=collate_fn
                    )
                    for ds in predict_dataset
                ]
            return dataloader(
                ds=predict_dataset,
                shuffle_flag=shuffle,
                collate=collate_fn
            )

        datamodule = cls()
        if train_dataset is not None:
            datamodule.train_dataloader = train_dataloader
        if val_dataset is not None:
            datamodule.val_dataloader = val_dataloader
        if test_dataset is not None:
            datamodule.test_dataloader = test_dataloader
        if predict_dataset is not None:
            datamodule.predict_dataloader = predict_dataloader
        return datamodule
