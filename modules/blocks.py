import torch
import torch.nn as tnn
from typing import Sequence, Union
from modules.layers import PaddedConv2d


class ConvolutionBlock(tnn.Module):
    def __init__(
            self,
            inc: int,
            outc: int,
            ks: Union[int, Sequence[int]] = (3, 3),
            stride: Union[int, Sequence[int]] = (1, 1),
            dilation: Union[int, Sequence[int]] = (1, 1),
            padding_mode: str = 'zeros',
            groups: int = 1,
            bias: bool = True,
            activation: tnn.Module = tnn.Mish(True)
    ):
        super().__init__()
        if isinstance(ks, int):
            ks = (ks,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(dilation, int):
            dilation = (dilation,) * 2
        self.net = tnn.Sequential(
            PaddedConv2d(
                in_channels=inc,
                out_channels=outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
                padding_mode=padding_mode,
                groups=groups,
                bias=bias
            ),
            tnn.BatchNorm2d(outc),
            activation,
        )

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out


class DeconvolutionBlock(tnn.Module):
    def __init__(
            self,
            inc: int,
            outc: int,
            ks: Union[int, Sequence[int]] = (3, 3),
            stride: Union[int, Sequence[int]] = (1, 1),
            dilation: Union[int, Sequence[int]] = (1, 1),
            padding: int = 0,
            output_padding: int = 0,
            groups: int = 1,
            bias: bool = True,
            activation: tnn.Module = tnn.Mish(True)
    ):
        super().__init__()
        if isinstance(ks, int):
            ks = (ks,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(dilation, int):
            dilation = (dilation,) * 2

        # noinspection PyTypeChecker
        self.net = tnn.Sequential(
            tnn.ConvTranspose2d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias
            ),
            tnn.BatchNorm2d(outc),
            activation,
        )

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out
