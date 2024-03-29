import torch
import torch.nn as tnn
from typing import Any
from typing import Dict
from typing import Union
from copy import deepcopy
from typing import Sequence
from typing import FrozenSet
from modules.utils import get_shape
from modules.helpers import Registry
from modules.blocks import ConvolutionBlock
from torch.nn.functional import interpolate
from modules.helpers import NormalizationRegistry


NORMALIZATION_REGISTRY = NormalizationRegistry()


__all__ = ['SimpleHead', 'RefinerHead', 'HeadRegistry']


class SimpleAttention(tnn.Module):
    def __init__(self, k: float = 1e-2):
        super(SimpleAttention, self).__init__()
        self.k = torch.nn.Parameter(torch.tensor(k))

    def forward(self, x):
        _, _, h, w = x.size()
        y = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        y = (
            y / (
                4 * (
                    self.k + (
                        y.sum(dim=(2, 3), keepdim=True) / ((w * h) - 1)
                    )
                )
            )
        ) + 0.5
        return x * y.sigmoid()


class AttentionUniRes(tnn.Module):
    def __init__(
            self,
            resize_mode: str = 'bilinear',
            k: float = 1e-2
    ):
        super(AttentionUniRes, self).__init__()
        self.attention = SimpleAttention(k=k)
        self.uni_res = UniRes(
            ndim=2,
            resize_mode=resize_mode
        )

    def forward(self, x: Sequence[torch.Tensor]):
        x = tuple(
            self.attention(x=t)
            for t in x
        )
        return self.uni_res(x)


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


class RefinerHead(tnn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            n_heads: int,
            in_channels: int,
            embedding_dim: int,
            resize_mode: str = 'bilinear',
            attention_factor: float = 1e-2,
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'groupnorm', 'affine': True}.items()),
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'sigmoid'}.items())
    ):
        super(RefinerHead, self).__init__()
        if norm_cfg is not None:
            norm_cfg = dict(norm_cfg)
        if act_cfg is not None:
            act_cfg = dict(act_cfg)
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.uni_res = AttentionUniRes(
            resize_mode=resize_mode,
            k=attention_factor
        )
        self.group_conv = ConvolutionBlock(
            ndim=2,
            inc=(n_heads * in_channels),
            outc=(n_heads * embedding_dim),
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=n_heads,
            bias=False,
            norm_cfg=frozenset(
                {
                    'alias': 'groupnorm',
                    'num_groups': n_heads,
                    'num_channels': (n_heads * embedding_dim),
                    'affine': True
                }.items()
            ),
            act_cfg=act_cfg,
            spectral_norm=False,
            order='CAN'
        )
        self.reduce_conv = ConvolutionBlock(
            ndim=3,
            inc=n_heads,
            outc=1,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias=False,
            norm_cfg=None,
            act_cfg=None,
            spectral_norm=False,
            order='CNA'
        )
        if bool(norm_cfg):
            # noinspection SpellCheckingInspection,PyUnresolvedReferences
            if (
                norm_cfg['alias'].startswith(('batchnorm', 'instancenorm'))
            ):
                norm_cfg['num_features'] = embedding_dim
            elif norm_cfg['alias'] == 'groupnorm':
                norm_cfg['num_channels'] = embedding_dim
                norm_cfg['num_groups'] = norm_cfg.get(
                    'num_groups', embedding_dim
                )
                norm_cfg['affine'] = norm_cfg.get('affine', True)
        self.norm = NORMALIZATION_REGISTRY(
            **norm_cfg
        ) if bool(norm_cfg) else tnn.Identity()

    def forward(
            self, x: Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]]
    ):
        c, d = self.embedding_dim, self.n_heads
        x = self.uni_res(x)
        x = torch.cat(tensors=x, dim=1)
        x = self.group_conv(x)
        # Reshape to 5D
        n, _, h, w = tuple(x.size())
        x = x.view(n, c, d, h, w)
        x = self.reduce_conv(x.permute(0, 2, 1, 3, 4))
        # Return to 4D
        x = x.squeeze(dim=1)
        x = self.norm(x)
        return x


class HeadRegistry(Registry):
    __registry = {
        'simple': SimpleHead,
        'refiner': RefinerHead,
    }

    def __init__(self):
        # noinspection SpellCheckingInspection
        self._current_registry = deepcopy(self.__registry)

    @classmethod
    def register(
            cls, alias: str, layer: tnn.Module, overwrite: bool = False
    ) -> None:
        if overwrite or not(alias in cls.__registry.keys()):
            cls.__registry[alias] = layer
        else:
            raise AssertionError(
                f"Alias ({alias}) is already exist in the registry!" +
                "Try different alias or use overwrite flag."
            )

    def add(
            self, alias: str, layer: tnn.Module, overwrite: bool = False
    ) -> None:
        if overwrite or not self.exists(alias=alias):
            self._current_registry[alias] = layer
        else:
            raise AssertionError(
                f"Alias ({alias}) is already exist in the current registry!" +
                "Try different alias or use overwrite flag."
            )

    def __call__(self, alias: str, *args, **kwargs) -> Any:
        assert self.exists(alias=alias), (
            f"Alias ({alias}) does not exist in the registry!"
        )
        return self.get(alias=alias)(*args, **kwargs)

    def get(self, alias: str) -> Any:
        return self._current_registry.get(alias, None)

    @property
    def keys(self) -> tuple:
        return tuple(self._current_registry.keys())

    def exists(self, alias: str) -> bool:
        return alias in self._current_registry

    def __str__(self):
        return f"Registered Aliases: {self.keys}"
