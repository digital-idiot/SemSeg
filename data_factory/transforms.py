import torch
import random
import torch.nn as tnn
from abc import ABCMeta
from typing import Union
from random import uniform
from typing import Callable
from typing import Sequence
from functools import partial
from abc import abstractmethod
from random import random as toss
from random import choice as choose
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine
from torchvision.transforms.functional import perspective
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
        self.__transform_map = list()

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
            'image', 'label', 'both', 'sync_pair'
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
                    if target in {'both'}:
                        image_transform = transform
                        label_transform = transform
                    elif target in {'image'}:
                        image_transform = transform
                    elif target in {'label'}:
                        label_transform = transform
                    if target in {'sync_pair'}:
                        image_transform, label_transform = transform()

                image_transforms.append(image_transform)
                label_transforms.append(label_transform)

            yield sequential_image_transform, sequential_label_transform

        elif self.get_mode() == 'any':
            image_transform = dummy_transform
            label_transform = dummy_transform
            transform, target, _ = choose(self.__transform_map)
            if target in {'both'}:
                image_transform = transform
                label_transform = transform
            elif target in {'image'}:
                image_transform = transform
            elif target in {'label'}:
                label_transform = transform
            if target in {'sync_pair'}:
                image_transform, label_transform = next(transform())
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


class RandomAffine(object):
    def __init__(
            self,
            degrees: Union[float, Sequence[float]],
            translate: Sequence[float] = None,
            scale: Sequence[float] = None,
            shear: Union[float, Sequence[float]] = None,
            interpolation: InterpolationMode = InterpolationMode.NEAREST,
            fill: Union[float, Sequence[float]] = 0,
            center: Sequence[int] = None
    ):
        if isinstance(degrees, (int, float)):
            degrees = (degrees, degrees)
        else:
            assert isinstance(
                degrees, Sequence
            ) and (len(degrees) == 2) and all(
                isinstance(d, (int, float)) and (-180 <= d <= 180)
                for d in degrees
            ), f"Invalid 'degrees': {degrees}"
        self.degrees = degrees

        if isinstance(translate, (int, float)):
            translate = (translate, translate)
        if translate is not None:
            assert isinstance(
                translate, Sequence
            ) and (len(translate) == 2) and all(
                isinstance(t, (int, float)) and (0 <= t <= 1)
                for t in translate
            ), f"Invalid 'translate': {translate}"
        else:
            translate = (0, 0)
        self.translate = translate

        if isinstance(scale, (int, float)):
            scale = (scale, scale)
        if scale is not None:
            assert isinstance(
                scale, Sequence
            ) and (len(scale) == 2) and all(
                isinstance(s, (int, float)) and (s > 0)
                for s in scale
            ), f"Invalid 'scale': {translate}"
        else:
            scale = (1, 1)
        self.scale = scale

        if isinstance(shear, (int, float)):
            shear = (shear, shear, 0, 0)
        if shear is not None:
            assert isinstance(
                shear, Sequence
            ) and (len(scale) in {2, 4}) and all(
                isinstance(sh, (int, float)) and (-180 <= sh <= 180)
                for sh in shear
            ), f"Invalid 'shear': {shear}"
            if len(shear) == 2:
                shear = tuple(shear) + (0, 0)
        else:
            shear = (0, 0, 0, 0)
        self.shear = shear

        self.interpolation = interpolation

        if fill is None:
            self.fill = 0
        elif isinstance(fill, (int, float)):
            self.fill = fill
        else:
            assert isinstance(
                fill, Sequence
            ) and all(
                isinstance(f, (int, float))
                for f in fill
            )
            self.fill = fill

        if center is not None:
            assert isinstance(center, Sequence), (len(center) == 2) and all(
                isinstance(c, int) and (c >= 0)
                for c in center
            )
        self.center = center

    @classmethod
    def affine_transform(
            cls,
            img: torch.Tensor,
            angle: float,
            translate: Sequence[int],
            scale: float,
            shear: Sequence[float],
            interpolation: InterpolationMode = InterpolationMode.NEAREST,
            fill: Sequence[float] = None,
            center: Sequence[int] = None
    ):
        squeeze_flag = False
        if img.ndim == 2:
            img = torch.unsqueeze(input=img, dim=0)
            squeeze_flag = True
        h, w = img.shape[-2:]
        translate = (translate[0] * h), (translate[1] * w)
        if center is not None:
            assert (0 < center[0] < h) and (0 < center[1] < w), (
                f"'center'({center}) is outside of image"
            )
        transformed_img = affine(
            img=img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
            center=center
        )
        if squeeze_flag:
            transformed_img = torch.squeeze(input=transformed_img, dim=0)
        return transformed_img

    def __call__(self, *args, **kwargs):
        angle = random.uniform(a=self.degrees[0], b=self.degrees[1])
        translate = random.uniform(
            a=-self.translate[0], b=self.translate[0]
        ), random.uniform(
            a=-self.translate[1], b=self.translate[1]
        )
        scale = random.uniform(a=self.scale[0], b=self.scale[1])
        shear = random.uniform(
            a=self.shear[0], b=self.shear[1]
        ), random.uniform(
            a=self.shear[2], b=self.shear[3]
        )
        image_transform = partial(
            RandomAffine.affine_transform,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=self.interpolation,
            fill=self.fill,
            center=self.center
        )
        label_transform = partial(
            RandomAffine.affine_transform,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=self.fill,
            center=self.center
        )
        yield image_transform, label_transform


class RandomPerspective(object):
    def __init__(
            self,
            image_shape: Sequence[int],
            distortion_scale: float = 0.5,
            interpolation: InterpolationMode = InterpolationMode.NEAREST,
            fill: Union[float, Sequence[float]] = None
    ):
        if isinstance(image_shape, int):
            image_shape = (1, image_shape, image_shape)
        if isinstance(image_shape, Sequence) and (
                len(image_shape) == 2
        ):
            image_shape = (1,) + tuple(image_shape)
        assert isinstance(image_shape, Sequence) and (
                len(image_shape) == 3
        ) and all(
            isinstance(d, int) and (d > 0)
            for d in image_shape
        ), f"Invalid 'image_shape': {image_shape}"
        self.image_shape = image_shape
        assert isinstance(
            distortion_scale, (int, float)
        ) and (
            0 <= distortion_scale <= 1
        ), (
            f"Invalid 'distortion_scale': {distortion_scale}\n" +
            "Valid range: [0, 1]"
        )
        self.distortion = distortion_scale
        self.interpolation = interpolation
        if fill is None:
            self.fill = 0
        elif isinstance(fill, (int, float)):
            self.fill = fill
        else:
            assert isinstance(
                fill, Sequence
            ) and all(
                isinstance(f, (int, float))
                for f in fill
            )
            self.fill = fill

    # noinspection SpellCheckingInspection
    @classmethod
    def perspective_transform(
            cls,
            img: torch.Tensor,
            startpoints: Sequence[Sequence[int]],
            endpoints: Sequence[Sequence[int]],
            interpolation: InterpolationMode = InterpolationMode.NEAREST,
            fill: Sequence[float] = None
    ):
        squeeze_flag = False
        if img.ndim == 2:
            img = torch.unsqueeze(img, 0)
            squeeze_flag = True
        transformed_image = perspective(
            img=img,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=interpolation,
            fill=fill
        )
        if squeeze_flag:
            transformed_image = torch.squeeze(input=transformed_image, dim=0)
        return transformed_image

    def __call__(self, *args, **kwargs):
        channels, height, width = self.image_shape
        half_height = height // 2
        half_width = width // 2
        top_left = [
            int(
                torch.randint(
                    0, int(self.distortion * half_width) + 1, size=(1,)
                ).item()
            ),
            int(
                torch.randint(
                    0, int(self.distortion * half_height) + 1, size=(1,)
                ).item()
            ),
        ]
        top_right = [
            int(
                torch.randint(
                    (
                        width - int(self.distortion * half_width) - 1
                    ), width, size=(1,)
                ).item()
            ),
            int(
                torch.randint(
                    0, int(self.distortion * half_height) + 1, size=(1,)
                ).item()
            ),
        ]
        bot_right = [
            int(
                torch.randint(
                    (
                        width - int(self.distortion * half_width) - 1
                    ), width, size=(1,)
                ).item()
            ),
            int(
                torch.randint(
                    (
                        height - int(self.distortion * half_height) - 1
                    ), height, size=(1,)
                ).item()
            ),
        ]
        bot_left = [
            int(
                torch.randint(
                    0, int(self.distortion * half_width) + 1, size=(1,)
                ).item()
            ),
            int(
                torch.randint(
                    (
                        height - int(self.distortion * half_height) - 1
                    ), height, size=(1,)
                ).item()
            ),
        ]
        start_points = [
            [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
        ]
        end_points = [top_left, top_right, bot_right, bot_left]

        fill = (self.fill,) * channels if isinstance(
            self.fill, (int, float)
        ) else self.fill

        yield partial(
            RandomPerspective.perspective_transform,
            startpoints=start_points,
            endpoints=end_points,
            interpolation=self.interpolation,
            fill=fill
        ), partial(
            RandomPerspective.perspective_transform,
            startpoints=start_points,
            endpoints=end_points,
            interpolation=InterpolationMode.NEAREST,
            fill=fill
        )
