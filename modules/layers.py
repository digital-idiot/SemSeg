import torch
import torch.nn as tnn
from typing import Tuple, List, Union
from torch.nn.functional import pad as torch_pad


__all__ = [
    'MultilabelClassifier',
    'PaddedConv2d',
    'Resize'
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
            tnn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.terminator(x)


class DynamicScalePad2D(tnn.Module):
    # noinspection SpellCheckingInspection
    r"""
        Pre-computes required padding to maintain the following relationships, after
        convolution operation.
            $$H_n = \left\lceil \frac{H_{n-1}}{S} \right\rceil$$
            $$W_n = \left\lceil \frac{W_{n-1}}{S} \right\rceil$$
        Where $H_{n-1}$ is height of the input tensor and $H_n$ is the height after
        the convolution operation. $W$ stands for width and has analogous notation.

        Reference: https://kutt.it/fbT6na
        """
    def __init__(
        self,
        mode: str,
        kernel_size: Union[int, Tuple[int, int]] = (2, 2),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        *args, ** kwargs
    ) -> None:
        super(DynamicScalePad2D, self).__init__()
        if not(mode in {'constant', 'reflect', 'replicate' or 'circular'}):
            raise NotImplementedError(
                "Unknown padding mode: {}!".format(str(mode))
            )
        else:
            self.mode = mode

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        assert isinstance(
            kernel_size, (tuple, list)
        ) and len(kernel_size) == 2 and all(
            isinstance(n, int) and n > 0
            for n in kernel_size
        ), "Invalid kernel_size: {}".format(kernel_size)
        self.ks = kernel_size

        if isinstance(stride, int):
            stride = (stride,) * 2
        assert isinstance(
            stride, (tuple, list)
        ) and len(stride) == 2 and all(
            isinstance(m, int) and m > 0
            for m in stride
        ), "Invalid stride: {}".format(stride)
        self.st = stride

        self.args = args
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise NotImplementedError(
                '{}-dimensional tensor is not supported!'.format(x.ndim)
            )

        sp_shape = x.shape[-2:]
        pads = list()
        for i in reversed(range(2)):
            res = sp_shape[i] % self.st[i]
            s = self.st[i] if res == 0 else res
            bi_pad = max(0, (self.ks[i] - s))
            pad_1 = bi_pad // 2
            pad_2 = bi_pad - pad_1
            pads.append(pad_1)
            pads.append(pad_2)
        return torch_pad(
            input=x,
            pad=pads,
            mode=self.mode,
            *self.args,
            **self.kwargs
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
        kwargs = dict()
        if padding_mode.lower() == 'zeros':
            pad_mode = 'constant'
            kwargs['value'] = 0
        else:
            pad_mode = padding_mode
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(dilation, int):
            dilation = (dilation,) * 2
        self.net = tnn.Sequential(
            DynamicScalePad2D(
                mode=pad_mode,
                kernel_size=kernel_size,
                stride=stride,
                **kwargs
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
            target_shape: Union[Tuple[int, ...], List[int]],
            mode: str = 'bilinear',
            align_corners: bool = False
    ):
        super(Resize, self).__init__()
        self.target_shape = target_shape
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tnn.functional.interpolate(
            input=x,
            size=self.target_shape,
            mode=self.mode,
            align_corners=self.align_corners
        )
        return x
