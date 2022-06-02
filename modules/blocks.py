import torch
import torch.nn as tnn
from typing import Dict
from typing import Union
from typing import Sequence
from typing import FrozenSet
from modules.layers import Padding
from modules.layers import AutoPadding
from torch.nn.functional import interpolate
from modules.helpers import ActivationRegistry
from modules.helpers import NormalizationRegistry

ACTIVATION_REGISTRY = ActivationRegistry()
NORMALIZATION_REGISTRY = NormalizationRegistry()


class ConvolutionBlock(tnn.Module):
    def __init__(
            self,
            inc: int,
            outc: int,
            ndim: int = 2,
            kernel_size: Union[int, Sequence[int]] = (3, 3),
            stride: Union[int, Sequence[int]] = (1, 1),
            dilation: Union[int, Sequence[int]] = (1, 1),
            padding: Union[int, Sequence[int], str] = 'auto',
            padding_mode: str = 'zeros',
            groups: int = 1,
            bias: Union[bool, str] = 'auto',
            norm_cfg: Union[Dict, FrozenSet] = None,
            act_cfg: Union[Dict, FrozenSet] = None,
            spectral_norm: bool = False,
            order: str = 'CNA'
    ):
        super(ConvolutionBlock, self).__init__()
        assert isinstance(
            order, str
        ) and (
               len(order) == 3
        ) and (
               set(order.upper()) == {'A', 'C', 'N'}
        ), f"Invalid order: {order}"
        self.order = order.upper()

        if inc is not None:
            if ndim == 1:
                conv_module = tnn.Conv1d
            elif ndim == 2:
                conv_module = tnn.Conv2d
            elif ndim == 3:
                conv_module = tnn.Conv3d
            else:
                raise NotImplementedError(
                    f"{ndim}D convolution not supported!"
                )
            conv_params = {'in_channels': inc}
        else:
            if ndim == 1:
                conv_module = tnn.LazyConv1d
            elif ndim == 2:
                conv_module = tnn.LazyConv2d
            elif ndim == 3:
                conv_module = tnn.LazyConv3d
            else:
                raise NotImplementedError(
                    f"{ndim}D convolution not supported!"
                )
            conv_params = dict()
        self.ndim = ndim

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.ndim
        if isinstance(stride, int):
            stride = (stride,) * self.ndim
        if isinstance(dilation, int):
            dilation = (dilation,) * self.ndim

        if isinstance(norm_cfg, FrozenSet):
            norm_cfg = dict(norm_cfg)
        if isinstance(act_cfg, FrozenSet):
            norm_cfg = dict(act_cfg)

        # noinspection SpellCheckingInspection
        pconv_module = tnn.Sequential()
        if isinstance(padding, str):
            if padding.lower() in {'same', 'auto'}:
                if padding.lower() == 'same' and any(s != 1 for s in stride):
                    raise ValueError(
                        "padding='same' is not supported for stride > 1"
                    )
                pconv_module.add_module(
                    name='pad',
                    module=AutoPadding(
                        ndim=self.ndim,
                        mode=padding_mode,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        value=0.0
                    )
                )
            elif padding.lower() == 'valid':
                pconv_module.add_module(
                    name='pad',
                    module=Padding(pad=None)
                )
            else:
                raise NotImplementedError(
                    f"Unknown padding: {padding}"
                )
        elif isinstance(padding, int):
            padding = (padding,) * (self.ndim * 2)
            pconv_module.add_module(
                name='pad',
                module=Padding(mode=padding_mode, pad=padding)
            )
        elif isinstance(padding, Sequence) and all(
                isinstance(p, int) for p in padding
        ):
            pconv_module.add_module(
                name='pad',
                module=Padding(mode=padding_mode, pad=padding)
            )
        else:
            raise ValueError(f'Invalid padding: {padding}')
        if isinstance(bias, str) and (bias.lower() == 'auto'):
            bias = True if (norm_cfg is None) else False

        # noinspection SpellCheckingInspection
        conv_params.update(
            {
                "out_channels": outc,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": 0,
                "dilation": dilation,
                "groups": groups,
                "bias": bias,
                "padding_mode": 'zeros',
                "device": None,
                "dtype": None
            }
        )

        # noinspection SpellCheckingInspection
        pconv = conv_module(**conv_params)
        if bool(spectral_norm):
            # noinspection SpellCheckingInspection
            pconv = tnn.utils.spectral_norm(pconv)

        pconv_module.add_module(
            name='conv',
            module=pconv
        )

        # noinspection SpellCheckingInspection,PyUnresolvedReferences
        if bool(norm_cfg) and (
            norm_cfg['alias'].startswith(('batchnorm', 'instancenorm'))
        ):
            norm_cfg['num_features'] = inc if (
                    self.order.index('N') < self.order.index('C')
            ) else outc

        layer_map = {
            'C': (
                'p_conv', pconv_module
            ),
            'N': (
                'norm', NORMALIZATION_REGISTRY(
                    **norm_cfg
                ) if bool(norm_cfg) else None
            ),
            'A': (
                'act', ACTIVATION_REGISTRY(
                    **act_cfg
                ) if bool(act_cfg) else None
            )
        }

        self.net = tnn.Sequential()
        for k in self.order:
            n, m = layer_map[k]
            if not(m is None):
                self.net.add_module(
                    name=n,
                    module=m
                )

    def forward(self, x: torch.Tensor):
        return self.net(x)


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
            activation: tnn.Module = tnn.Mish(True),
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


# noinspection SpellCheckingInspection
class UpsampleBlock(tnn.Module):
    def __init__(
            self,
            inc: int,
            outc: int,
            ndim: int = 2,
            kernel_size: Union[int, Sequence[int]] = (3, 3),
            stride: Union[int, Sequence[int]] = (1, 1),
            dilation: Union[int, Sequence[int]] = (1, 1),
            padding: Union[int, Sequence[int], str] = 'auto-adjust',
            padding_mode: str = 'zeros',
            groups: int = 1,
            bias: Union[bool, str] = 'auto',
            norm_cfg: Dict = None,
            act_cfg: Dict = frozenset({'alias': 'relu'}.items()),
            spectral_norm: bool = False,
            order: str = 'CNA',
            target_shape: Union[int, Sequence[int]] = None,
            scale_factor: Union[float, Sequence[float]] = None,
            resize_mode: str = 'bilinear',
            align_corners: bool = False,
            antialias: bool = False
    ):
        super(UpsampleBlock, self).__init__()
        if isinstance(target_shape, Sequence):
            assert len(target_shape) == ndim, (
                f"'target_shape' is {len(target_shape)}D while convolution " +
                f"kernel is {ndim}D"
            )
        assert not((target_shape is None) and (scale_factor is None)), (
            "Neither 'target_shape' nor `scale_factor` is specified!"
        )
        self.target_shape = target_shape
        self.scale_factor = scale_factor
        self.resize_mode = resize_mode
        self.align_corners = align_corners
        self.antialias = antialias

        self.padding = padding
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * self.ndim
        if isinstance(stride, int):
            self.stride = (stride,) * self.ndim
        if isinstance(dilation, int):
            self.dilation = (dilation,) * self.ndim

        self.net = tnn.Sequential()
        self.net.add_module(
            name='conv',
            module=ConvolutionBlock(
                inc=inc,
                outc=outc,
                ndim=ndim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                padding=0 if (
                    isinstance(self.padding, str) and (
                        self.padding.lower() == 'auto-adjust'
                    )
                ) else padding,
                padding_mode=padding_mode,
                groups=groups,
                bias=bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                spectral_norm=spectral_norm,
                order=order
            )
        )

    def forward(self, x: torch.Tensor):
        if (self.scale_factor is not None) and (self.target_shape is None):
            shape = torch.tensor(
                data=x.shape[-self.ndim:],
                device=x.device,
                dtype=x.dtype,
                requires_grad=False
            )
            scale_factor = torch.tensor(
                data=self.scale_factor,
                device=x.device,
                dtype=x.dtype,
                requires_grad=False
            )
            target_shape = torch.round(
                input=(shape * scale_factor),
                decimals=0
            ).to(dtype=torch.long)
        else:
            target_shape = torch.tensor(
                data=self.target_shape,
                device=x.device,
                dtype=x.dtype,
                requires_grad=False
            )

        if self.paddinglower() == 'auto-adjust':
            target_shape = target_shape + (
                torch.tensor(
                    data=self.dilation,
                    device=x.device,
                    dtype=x.dtype,
                    requires_grad=False
                ) * (
                    torch.tensor(
                        data=self.kernel_size,
                        device=x.device,
                        dtype=x.dtype,
                        requires_grad=False
                    ) - 1
                )
            ) - (
                torch.tensor(
                    data=self.stride,
                    device=x.device,
                    dtype=x.dtype,
                    requires_grad=False
                ) - 1
            )

        # noinspection PyArgumentList
        x = interpolate(
            input=x,
            size=target_shape.tolist(),
            scale_factor=None,
            recompute_scale_factor=None,
            mode=self.resize_mode,
            align_corners=self.align_corners,
            antialias=self.antialias
        )
        self.net(x)
