import torch
import numpy as np
import torch.nn as tnn
from typing import Any
from typing import Dict
from typing import Union
from typing import Sequence
from collections import OrderedDict
from torch.nn.functional import pad as torch_pad


__all__ = [
    'MultilabelClassifier',
    'AutoPadding',
    'Padding',
    'PaddedConv2d',
    'Resize',
    'BranchModule',
    'StagingModule',
    'ParallelModule'
]


class MultilabelClassifier(tnn.Module):
    def __init__(self, in_channels: int, n_classes):
        super(MultilabelClassifier, self).__init__()
        self.terminator = tnn.Sequential(
            tnn.Conv2d(
                in_channels=in_channels,
                out_channels=n_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
            ),
            tnn.Softmax2d()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.terminator(x)


class Padding(tnn.Module):
    def __init__(
            self,
            mode: str = 'zeros',
            pad: Sequence[int] = None,
            value: Any = 0.0
    ):
        super(Padding, self).__init__()
        if mode == 'zeros':
            mode = 'constant'
            value = 0
        if not(mode in {'constant', 'reflect', 'replicate' or 'circular'}):
            raise NotImplementedError(
                "Unknown padding mode: {}!".format(str(mode))
            )
        else:
            self.mode = mode

        if pad is None:
            pad = (0, 0)
        self.pad = pad
        self.value = value

    def forward(self, x):
        return torch_pad(
            input=x,
            pad=self.pad,
            mode=self.mode,
            value=self.value
        )


class AutoPadding(tnn.Module):
    # noinspection SpellCheckingInspection
    r"""
        Pre-computes required padding to maintain the following relationships,
        after convolution operation.
            $$X_n = \left\lfloor \frac{X_{n-1}}{S} \right\rfloor$$
        Where $X_{n-1}$ is dimension size of the input tensor and $H_n$ is the
        dimension size after the convolution operation.

        Reference: https://kutt.it/fbT6na
        """
    def __init__(
        self,
        mode: str,
        ndim: int = 2,
        kernel_size: Union[int, Sequence[int]] = (2, 2),
        stride: Union[int, Sequence[int]] = (1, 1),
        dilation: Union[int, Sequence[int]] = (1, 1),
        value: Any = 0.0
    ) -> None:
        super(AutoPadding, self).__init__()

        if not(ndim in {1, 2, 3}):
            raise NotImplementedError(
                f"{ndim}D padding not supported"
            )
        self.ndim = ndim

        if mode == 'zeros':
            mode = 'constant'
            value = 0
        if not(mode in {'constant', 'reflect', 'replicate' or 'circular'}):
            raise NotImplementedError(
                "Unknown padding mode: {}!".format(str(mode))
            )
        else:
            self.mode = mode

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim
        assert isinstance(
            kernel_size, (tuple, list)
        ) and len(kernel_size) == ndim and all(
            isinstance(n, int) and n > 0
            for n in kernel_size
        ), f"Invalid kernel_size: {kernel_size}"
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride,) * ndim
        assert isinstance(
            stride, (tuple, list)
        ) and len(stride) == ndim and all(
            isinstance(m, int) and m > 0
            for m in stride
        ), f"Invalid stride: {stride}"
        self.stride = stride

        if isinstance(dilation, int):
            dilation = (dilation,) * ndim
        assert isinstance(
            dilation, (tuple, list)
        ) and len(dilation) == ndim and all(
            isinstance(p, int) and p > 0
            for p in dilation
        ), f"Invalid dilation: {dilation}"
        self.dilation = dilation
        self.value = value
        bi_padding = (
            np.array(
                self.dilation, dtype=int
            ) * (
                 np.array(
                     self.kernel_size, dtype=int
                 ) - 1
            )
        ) - (
            np.array(
                self.stride, dtype=int
            ) - 1
        )
        bi_padding = np.flip(bi_padding, axis=0)
        padding_a = bi_padding // 2
        padding_b = bi_padding - padding_a
        self.paddings = np.stack(
            (padding_a, padding_b), axis=-1
        ).ravel(order='C').tolist()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.ndim:
            raise AssertionError(
                'Dimension mismatch!\n' +
                f'Expected {self.ndim}D tensor got {x.ndim} tensor.'
            )
        return torch_pad(
            input=x,
            pad=self.paddings,
            mode=self.mode,
            value=self.value
        )


class PaddedConv2d(tnn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        device=None,
        dtype=None
    ):
        super(PaddedConv2d, self).__init__()
        # kwargs = dict()
        if padding_mode.lower() == 'zeros':
            pad_mode = 'constant'
            # value = 0
        else:
            pad_mode = padding_mode
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(dilation, int):
            dilation = (dilation,) * 2
        self.net = tnn.Sequential(
            AutoPadding(
                mode=pad_mode,
                ndim=2,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                value=0
            ),
            tnn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding=0,
                padding_mode='zeros',
                device=device,
                dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Resize(tnn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            target_shape: Union[int, Sequence[int]] = None,
            scale_factor: Union[float, Sequence[float]] = None,
            recompute_scale_factor: bool = False,
            mode: str = 'bilinear',
            align_corners: bool = False,
            antialias: bool = False
    ):
        super(Resize, self).__init__()
        assert not((target_shape is None) and (scale_factor is None)), (
            "'target_shape' and 'scale_factor' both cannot be None"
        )
        self.kwargs = {
            'size': target_shape,
            'scale_factor': scale_factor,
            'mode': mode,
            'align_corners': align_corners if mode in {
                'linear', 'bilinear', 'bicubic', 'trilinear'
            } else None,
            'recompute_scale_factor ': recompute_scale_factor,
            'antialias': antialias if mode in {
                'bilinear', 'bicubic'
            } else False
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tnn.functional.interpolate(
            input=x,
            **self.kwargs
        )
        return x


class BranchModule(tnn.Sequential):
    def __init__(self, *args):
        super(BranchModule, self).__init__(*args)

    def forward(self, x: Any):
        out = OrderedDict()
        for name, module in self.named_children():
            out[name] = module(x)
        return out


class StagingModule(tnn.Sequential):
    def __init__(self, *args):
        super(StagingModule, self).__init__(*args)

    def forward(self, x: Any):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            out[name] = x
        return out


class ParallelModule(tnn.Sequential):
    def __init__(self, *args):
        super(ParallelModule, self).__init__(*args)

    def forward(self, x: Union[Sequence[Any], Dict[str, Any]]):
        assert len(x) <= len(self), (
            f"Received {len(x)} inputs " +
            f"while there are {len(self)} parallel stages!"
        )
        if isinstance(x, Sequence):
            out = list()
            for i in range(len(x)):
                out.append(self[i](x[i]))
        elif isinstance(x, Dict):
            out = OrderedDict()
            for name, data in self.x.items():
                out[name] = self.get_submodule(target=name)(data)
        else:
            raise AssertionError(
                f"Received input is an instance of {type(x)}.\n" +
                "Expected input to be an instance of Sequence or Dict!"
            )
        return out
