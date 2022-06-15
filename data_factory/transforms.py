import torch
import torch.nn as tnn
from abc import ABCMeta
from random import uniform
from typing import Callable
from typing import Sequence
from abc import abstractmethod
from random import random as toss
from random import choice as choose
from torchvision.transforms.functional import adjust_sharpness


class TransformPair(metaclass=ABCMeta):
    def __init__(self):
        self.__transform_map = list()

    @abstractmethod
    def add_transform(
            self,
            transform: Callable,
            random_apply=False,
            target: str = 'both'
    ):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class SegmentationTransform(TransformPair):
    __valid_modes = {'all', 'any'}

    def __init__(self, mode: str = 'all'):
        super(SegmentationTransform, self).__init__()
        mode = mode.lower()
        if mode in self.__valid_modes:
            self._mode = mode
        else:
            raise ValueError(
                f"Unsupported mode: {mode}!\n" +
                f"Valid modes are: {self.__valid_modes}"
            )

    def get_mode(self):
        return self._mode

    def set_mode(self, mode: str):
        if mode in self.__valid_modes:
            self._mode = mode
        else:
            raise ValueError(
                f"Unsupported mode: {mode}!\n" +
                f"Valid modes are: {self.__valid_modes}"
            )

    def add_transform(
            self,
            transform: Callable,
            probability: float = 1,
            target: str = 'both'
    ):
        target = target.lower()
        valid_targets = {
            'image', 'label', 'both'
        }
        assert target in valid_targets, (
            f"Invalid target ({target})!\n" +
            f"Valid targets are: {valid_targets}"
        )
        assert isinstance(transform, Callable), (
            "specified transform is not callable"
        )
        assert 0 <= probability <= 1, (
            "Invalid probability!"
        )
        self.__transform_map.append(
            (transform, target, probability)
        )

    def __len__(self):
        return len(self.__transform_map)

    def __call__(self):

        def dummy_transform(x):
            return x

        if self.get_mode() == 'all':
            image_transforms = list()
            label_transforms = list()

            def sequential_image_transform(x):
                for transform_fn in image_transforms:
                    x = transform_fn(x)
                return x

            def sequential_label_transform(x):
                for transform_fn in label_transforms:
                    x = transform_fn(x)
                return x

            for transform, target, probability in self.__transform_map:
                image_transform = dummy_transform
                label_transform = dummy_transform
                if toss() <= probability:
                    if target in {'image', 'both'}:
                        image_transform = transform
                    if target in {'label', 'both'}:
                        label_transform = transform
                image_transforms.append(image_transform)
                label_transforms.append(label_transform)

            yield sequential_image_transform, sequential_label_transform

        elif self.get_mode() == 'any':
            image_transform = dummy_transform
            label_transform = dummy_transform
            transform, target, _ = choose(self.__transform_map)
            if target in {'image', 'both'}:
                image_transform = transform
            if target in {'label', 'both'}:
                label_transform = transform
            yield image_transform, label_transform
        else:
            yield dummy_transform, dummy_transform


class RandomSharpness(tnn.Module):
    def __init__(self, sharpness_range: Sequence[float]):
        super(RandomSharpness, self).__init__()
        assert isinstance(
            sharpness_range, Sequence
        ) and (len(sharpness_range) == 2) and all(
            [isinstance(n, (int, float)) for n in sharpness_range]
        ), f"Invalid 'sharpness_range': {sharpness_range}"
        self.sharpness_range = sorted(sharpness_range, reverse=False)

    def forward(self, x: torch.Tensor):
        return adjust_sharpness(
            img=x,
            sharpness_factor=uniform(*self.sharpness_range)
        )
