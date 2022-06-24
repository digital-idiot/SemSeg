import torch
import torch.nn as tnn
from typing import Any
from typing import Dict
from typing import Union
from typing import Sequence
from typing import FrozenSet
from modules.utils import get_shape
from modules.blocks import ConvolutionBlock
from torch.nn.functional import interpolate


__all__ = ['SimpleHead']


class UniRes(tnn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            ndim: int = 2,
            resize_mode: str = 'bilinear',
    ):
        super(UniRes, self).__init__()
        self.ndim = ndim
        self.resize_mode = resize_mode

    def forward(
            self,
            x: Sequence[torch.Tensor]
    ):
        target_shape = torch.tensor(
            [get_shape(t)[-self.ndim:] for t in x],
            device=x[0].device,
            dtype=torch.long
        ).max(dim=0).values.tolist()
        target_shape = target_shape

        # noinspection PyArgumentList
        x = tuple(
            interpolate(
                input=t,
                size=target_shape,
                scale_factor=None,
                recompute_scale_factor=None,
                mode=self.resize_mode,
                align_corners=False,
                antialias=False
            )
            for t in x
        )
        return x


class Sum(tnn.Module):
    def __init__(self):
        super(Sum, self).__init__()
        self.dim = 0

    def forward(self, x: Sequence[torch.Tensor]):
        return torch.stack(
            tensors=x, dim=self.dim
        ).sum(dim=self.dim, keepdim=False)


class Concat(tnn.Module):
    def __init__(self, dim: int = 1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x: Sequence[torch.Tensor]):
        return torch.cat(tensors=x, dim=self.dim)


class SimpleHead(tnn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            in_channels: int,
            embedding_dim: int,
            resize_mode: str = 'bilinear',
            fusion: str = 'sum',
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'relu6'}.items())
    ):
        super(SimpleHead, self).__init__()
        self.uni_res = UniRes(
            ndim=2,
            resize_mode=resize_mode
        )
        if fusion.lower() == 'sum':
            self.fuse = Sum()
        elif fusion.lower() == 'concat':
            self.fuse = Concat(dim=1)
        else:
            raise ValueError(
                f"Unknown fusion mode: {fusion}\n" +
                "Supported fusion modes: {'concat', 'sum'}"
            )
        self.conv = ConvolutionBlock(
            ndim=2,
            inc=in_channels,
            outc=embedding_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            spectral_norm=False,
            order='CNA'
        )

    def forward(self, x: Sequence[torch.Tensor]):
        x = self.uni_res(x)
        x = self.fuse(x)
        x = self.conv(x)
        return x


class Refiner(tnn.Module):
    def __init__(self):
        super(Refiner, self).__init__()
