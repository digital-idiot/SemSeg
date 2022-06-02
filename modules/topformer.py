import torch
from torch import nn
from typing import Any
from typing import Dict
from typing import Union
from copy import deepcopy
from typing import Sequence
from typing import FrozenSet
import torch.nn.functional as tnf
from collections import OrderedDict
from modules.utils import get_shape
from modules.helpers import Registry
from modules.helpers import DropPath
from modules.layers import BranchModule
from modules.utils import make_divisible
from modules.layers import StagingModule
from modules.layers import ParallelModule
from modules.blocks import ConvolutionBlock


__all__ = ['TopFormerBackBone', 'TopFormerModule']


class FeedForwardNetwork(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            drop_rate: float = 0.0,
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'relu'}.items()),
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items())
    ):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        dropout_layer = nn.Dropout(drop_rate)

        self.net = nn.Sequential()

        self.net.add_module(
            name='preconv',
            module=ConvolutionBlock(
                ndim=2,
                inc=in_features,
                outc=hidden_features,
                kernel_size=(1, 1),
                stride=(1, 1),
                dilation=(1, 1),
                padding='auto',
                padding_mode='zeros',
                groups=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None,
                spectral_norm=False,
                order='CNA'
            )
        )

        self.net.add_module(
            name='dwconv',
            module=ConvolutionBlock(
                ndim=2,
                inc=hidden_features,
                outc=hidden_features,
                kernel_size=(3, 3),
                stride=(1, 1),
                dilation=(1, 1),
                padding='auto',
                padding_mode='zeros',
                groups=hidden_features,
                bias=True,
                norm_cfg=frozenset({'alias': 'batchnorm_2d'}.items()),
                act_cfg=act_cfg,
                spectral_norm=False,
                order='CNA'
            )
        )

        self.net.add_module(name='dwdropout', module=dropout_layer)

        self.net.add_module(
            name='postconv',
            module=ConvolutionBlock(
                ndim=2,
                inc=hidden_features,
                outc=out_features,
                kernel_size=(1, 1),
                stride=(1, 1),
                dilation=(1, 1),
                padding='auto',
                padding_mode='zeros',
                groups=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None,
                spectral_norm=False,
                order='CNA'
            )
        )

        self.net.add_module(name='postdropout', module=dropout_layer)

    def forward(self, x):
        return self.net(x)


class InvertedResidual(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            inc: int,
            outc: int,
            kernel_size: int,
            stride: int,
            expand_ratio: int,
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'relu'}.items()),
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items())
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in {1, 2}

        hidden_dim = int(round(inc * expand_ratio))
        self.residual_connect = (self.stride == 1) and (inc == outc)
        self.out_channels = outc
        self.scaled = stride > 1
        self.net = nn.Sequential()
        if expand_ratio != 1:
            # noinspection PyTypeChecker
            self.net.add_module(
                name='conv_in',
                module=ConvolutionBlock(
                    ndim=2,
                    inc=inc,
                    outc=hidden_dim,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    dilation=(1, 1),
                    groups=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    padding='auto',
                    padding_mode='zeros',
                    bias=False,
                    spectral_norm=False,
                    order='CNA'
                )
            )
        # noinspection PyTypeChecker
        self.net.add_module(
            name='conv_dw',
            module=ConvolutionBlock(
                ndim=2,
                inc=hidden_dim,
                outc=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=(1, 1),
                groups=hidden_dim,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                padding='auto',
                padding_mode='zeros',
                bias=False,
                spectral_norm=False,
                order='CNA'
            )
        )
        # noinspection PyTypeChecker
        self.net.add_module(
            name='conv_out',
            module=ConvolutionBlock(
                ndim=2,
                inc=hidden_dim,
                outc=outc,
                kernel_size=(1, 1),
                stride=(1, 1),
                dilation=(1, 1),
                groups=1,
                norm_cfg=dict(norm_cfg),
                act_cfg=None,
                padding='auto',
                padding_mode='zeros',
                bias=False,
                spectral_norm=False,
                order='CNA'
            )
        )

    def forward(self, x):
        if self.residual_connect:
            return x + self.net(x)
        else:
            return self.net(x)


class TokenPyramidModule(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            stage_configs: Union[
                OrderedDict[str, Dict[str, Any]],
                Sequence[Dict[str, Any]]
            ],
            input_channels: int = 3,
            init_features: int = 16,
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'relu'}.items()),
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items()),
            width_multiplier: float = 1.0
    ):
        super().__init__()
        if isinstance(stage_configs, Sequence):
            stage_configs = OrderedDict(
                [
                    (f"stage_{i+1}", conf)
                    for i, conf in enumerate(stage_configs)
                ]
            )
        self.net = nn.Sequential()
        self.net.add_module(
            name='stem',
            module=ConvolutionBlock(
                inc=input_channels,
                outc=init_features,
                ndim=2,
                kernel_size=(3, 3),
                stride=(2, 2),
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
        )
        self.out_channels = OrderedDict()
        staging_module = StagingModule()
        for stage_name, stage_conf in stage_configs.items():
            stage = nn.Sequential()
            if isinstance(stage_conf, Sequence):
                stage_conf = OrderedDict(
                    [
                        (f"resblock_{i+1}", conf)
                        for i, conf in enumerate(stage_conf)
                    ]
                )
            for block_name, params in stage_conf.items():
                if isinstance(params, Sequence):
                    assert len(params) == 4, (
                        f"Invalid layer spec encountered: {params}\n" +
                        f"Expected 4 parameters received {len(params)}"
                    )
                    params = {
                        'kernel_size': params[0],
                        'expand_ratio': params[1],
                        'base_channels': params[2],
                        'stride': params[3]
                    }
                layer_params = {
                    "inc": init_features,
                    "act_cfg": act_cfg,
                    "norm_cfg": norm_cfg
                }
                params['outc'] = make_divisible(
                    value=(
                        params.pop('base_channels') * width_multiplier
                    ),
                    divisor=8,
                    min_value=8
                )
                layer_params.update(params)
                stage.add_module(
                    name=block_name,
                    module=InvertedResidual(**layer_params)
                )
                init_features = params['outc']
            self.out_channels[stage_name] = init_features
            staging_module.add_module(name=stage_name, module=stage)

        self.net.add_module(
            name='staging_module',
            module=staging_module
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(torch.nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            dim: int,
            key_dim: int,
            num_heads: int,
            attn_ratio: Union[float, int] = 4.0,
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = None,
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items()),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.attn_ratio = attn_ratio

        self.multi_branch = BranchModule()
        self.multi_branch.add_module(
            name='branch_q',
            module=ConvolutionBlock(
                inc=dim,
                outc=self.nh_kd,
                ndim=2,
                kernel_size=(1, 1),
                stride=(1, 1),
                dilation=(1, 1),
                padding='auto',
                padding_mode='zeros',
                groups=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None,
                spectral_norm=False,
                order='CNA'
            )
        )
        self.multi_branch.add_module(
            name='branch_k',
            module=ConvolutionBlock(
                inc=dim,
                outc=self.nh_kd,
                ndim=2,
                kernel_size=(1, 1),
                stride=(1, 1),
                dilation=(1, 1),
                padding='auto',
                padding_mode='zeros',
                groups=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None,
                spectral_norm=False,
                order='CNA'
            )
        )
        self.multi_branch.add_module(
            name='branch_v',
            module=ConvolutionBlock(
                inc=dim,
                outc=self.dh,
                ndim=2,
                kernel_size=(1, 1),
                stride=(1, 1),
                dilation=(1, 1),
                padding='auto',
                padding_mode='zeros',
                groups=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None,
                spectral_norm=False,
                order='CNA'
            )
        )

        self.projection = ConvolutionBlock(
            inc=self.dh,
            outc=dim,
            ndim=2,
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
            order='ACN'
        )

    def forward(self, x):
        b, c, h, w = get_shape(x)
        branch_outputs = self.multi_branch(x)
        x = torch.matmul(
            input=torch.matmul(
                input=branch_outputs['branch_q'].reshape(
                    b, self.num_heads, self.key_dim, (h * w)
                ).permute(0, 1, 3, 2),
                other=branch_outputs['branch_k'].reshape(
                    b, self.num_heads, self.key_dim, (h * w)
                )
            ).softmax(dim=-1),
            other=branch_outputs['branch_v'].reshape(
                b, self.num_heads, self.d, (h * w)
            ).permute(0, 1, 3, 2)
        ).permute(0, 1, 3, 2).reshape(b, self.dh, h, w)
        return self.projection(x)


class SemanticsExtractorBlock(nn.Module):

    # noinspection SpellCheckingInspection
    def __init__(
            self,
            dim: int,
            key_dim: int,
            num_heads,
            mlp_ratio=4.0,
            attn_ratio=2.0,
            drop_rate=0.0,
            drop_path=0.0,
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'relu'}.items()),
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items())
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = AttentionBlock(
            dim=dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

        self.drop_path = DropPath(
            drop_path
        ) if drop_path > 0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForwardNetwork(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=None,
            drop_rate=drop_rate,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class SemanticsExtractorModule(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            block_count: int,
            embedding_dim: int,
            key_dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            attn_ratio: float = 2.0,
            drop_rate: float = 0.0,
            drop_path: Union[float, Sequence[float]] = 0.0,
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = None
    ):
        super().__init__()
        self.block_count = block_count
        self.transformer_sequence = nn.Sequential()

        for i in range(self.block_count):
            self.transformer_sequence.add_module(
                name=f'extractor_{i+1}',
                module=SemanticsExtractorBlock(
                    dim=embedding_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    drop_rate=drop_rate,
                    drop_path=drop_path[i] if isinstance(
                        drop_path, list
                    ) else drop_path,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg
                )
            )

    def forward(self, x):
        return self.transformer_sequence(x)


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(
            self,
            inputs: Union[
                Sequence[torch.Tensor],
                OrderedDict[str, torch.Tensor]
            ]
    ):
        if isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.values())
        b, c, h, w = get_shape(inputs[-1])
        h = h // self.stride
        w = w // self.stride
        return torch.cat(
            [
                nn.functional.adaptive_avg_pool2d(
                    input=inp, output_size=(h, w)
                )
                for inp in inputs
            ],
            dim=1
        )


class InjectionDotSum(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            inc: int,
            outc: int,
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'hsigmoid'}.items())
    ) -> None:

        super(InjectionDotSum, self).__init__()

        self.local_embedding = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=None,
            spectral_norm=False,
            order='CNA'
        )

        self.global_embedding = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=None,
            spectral_norm=False,
            order='CNA'
        )

        self.global_act = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=None,
            spectral_norm=False,
            order='CNA'
        )

        self.global_act = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            spectral_norm=False,
            order='CNA'
        )

    # noinspection SpellCheckingInspection
    def forward(self, x):
        """
        x_g: global features
        x_l: local features
        """
        x_l, x_g = x
        b, c, h, w = get_shape(x_l)
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)

        # noinspection PyArgumentList
        sig_act = tnf.interpolate(
            input=global_act,
            size=(h, w),
            scale_factor=None,
            recompute_scale_factor=None,
            mode='bilinear',
            align_corners=False,
            antialias=False
        )
        global_feat = self.global_embedding(x_g)

        # noinspection PyArgumentList
        global_feat = tnf.interpolate(
            input=global_feat,
            size=(h, w),
            scale_factor=None,
            recompute_scale_factor=None,
            mode='bilinear',
            align_corners=False,
            antialias=False
        )
        out = (local_feat * sig_act) + global_feat
        return out


class InjectionDotSumCBR(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            inc: int,
            outc: int,
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'hsigmoid'}.items())
    ) -> None:
        """
        local_embedding: conv-bn-relu
        global_embedding: conv-bn-relu
        global_act: conv
        """
        super(InjectionDotSumCBR, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=None,
            spectral_norm=False,
            order='CNA'
        )

        self.global_embedding = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=frozenset({'alias': 'relu'}.items()),
            spectral_norm=False,
            order='CNA'
        )

        self.global_act = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=frozenset({'alias': 'relu'}.items()),
            spectral_norm=False,
            order='CNA'
        )

        self.global_act = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            spectral_norm=False,
            order='CNA'
        )

    # noinspection SpellCheckingInspection
    def forward(self, x):
        x_l, x_g = x
        b, c, h, w = get_shape(x_l)
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)

        # noinspection PyArgumentList
        sig_act = tnf.interpolate(
            input=global_act,
            size=(h, w),
            scale_factor=None,
            recompute_scale_factor=None,
            mode='bilinear',
            align_corners=False,
            antialias=False
        )
        global_feat = self.global_embedding(x_g)

        # noinspection PyArgumentList
        global_feat = tnf.interpolate(
            input=global_feat,
            size=(h, w),
            scale_factor=None,
            recompute_scale_factor=None,
            mode='bilinear',
            align_corners=False,
            antialias=False
        )
        out = (local_feat * sig_act) + global_feat
        return out


class FuseBlockSum(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            inc: int,
            outc: int,
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items())
    ) -> None:
        super(FuseBlockSum, self).__init__()

        self.fuse_l = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=None,
            spectral_norm=False,
            order='CNA'
        )

        self.fuse_h = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=None,
            spectral_norm=False,
            order='CNA'
        )

    # noinspection SpellCheckingInspection
    def forward(self, x):
        x_l, x_h = x
        b, c, h, w = get_shape(x_l)
        feat_l = self.fuse_l(x_l)
        # noinspection PyArgumentList
        feat_h = tnf.interpolate(
            input=self.fuse_h(x_h),
            size=(h, w),
            scale_factor=None,
            recompute_scale_factor=None,
            mode='bilinear',
            align_corners=False,
            antialias=False
        )
        out = feat_l + feat_h
        return out


class FuseBlockDot(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            inc: int,
            outc: int,
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'hsigmoid'}.items())
    ) -> None:
        super(FuseBlockDot, self).__init__()

        self.fuse_l = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=None,
            spectral_norm=False,
            order='CNA'
        )

        self.fuse_h = ConvolutionBlock(
            inc=inc,
            outc=outc,
            ndim=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            padding_mode='zeros',
            groups=1,
            bias='auto',
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            spectral_norm=False,
            order='CNA'
        )

    # noinspection SpellCheckingInspection
    def forward(self, x):
        x_l, x_h = x
        b, c, h, w = get_shape(x_l)
        feat_l = self.fuse_l(x_l)
        # noinspection PyArgumentList
        feat_h = tnf.interpolate(
            input=self.fuse_h(x_h),
            size=(h, w),
            scale_factor=None,
            recompute_scale_factor=None,
            mode='bilinear',
            align_corners=False,
            antialias=False
        )
        out = feat_l * feat_h
        return out


class SubmoduleRegistry(Registry):
    __registry = {
        'fuse_sum': FuseBlockSum,
        'fuse_dot': FuseBlockDot,
        'dot_sum': InjectionDotSum,
        'dot_sum_cbr': InjectionDotSumCBR,
    }

    def __init__(self):
        # noinspection SpellCheckingInspection
        self._current_registry = deepcopy(self.__registry)

    @classmethod
    def register(
            cls, alias: str, layer: nn.Module, overwrite: bool = False
    ) -> None:
        if overwrite or not(alias in cls.__registry.keys()):
            cls.__registry[alias] = layer
        else:
            raise AssertionError(
                f"Alias ({alias}) is already exist in the registry!" +
                "Try different alias or use overwrite flag."
            )

    def add(
            self, alias: str, layer: nn.Module, overwrite: bool = False
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


class TopFormerModule(nn.Module):

    __module_registry = SubmoduleRegistry()

    # noinspection SpellCheckingInspection
    def __init__(
            self,
            input_channels: int,
            stage_configs: Union[
                OrderedDict[str, Dict[str, Any]],
                Sequence[Dict[str, Any]]
            ],
            out_channels: Union[Sequence[int], Dict[str, int]],
            depths: int = 4,
            key_dim: int = 16,
            num_heads: int = 8,
            attn_ratios: float = 2,
            mlp_ratios: float = 2,
            c2t_stride: int = 2,
            drop_path_rate: float = 0.0,
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[Any]]
            ] = frozenset({'alias': 'relu6'}.items()),
            injection_type: str = "multi_sum",
    ):
        super().__init__()

        if isinstance(stage_configs, Sequence):
            stage_configs = OrderedDict(
                [
                    (f"stage_{i+1}", conf)
                    for i, conf in enumerate(stage_configs)
                ]
            )
        self.stage_configs = stage_configs
        self.norm_cfg = norm_cfg
        self.injection_type = injection_type

        self.tpm = TokenPyramidModule(
            stage_configs=stage_configs,
            input_channels=input_channels,
            init_features=16,
            act_cfg={'alias': 'relu'},
            norm_cfg=norm_cfg,
            width_multiplier=1.0
        )

        self.stage_names = tuple(self.tpm.out_channels.keys())
        if isinstance(out_channels, Sequence):
            out_channels = OrderedDict(
                [
                    (self.stage_names[i], n)
                    for i, n in enumerate(out_channels)
                ]
            )
        self.channel_splits = tuple(self.tpm.out_channels.values())
        self.embed_dim = sum(self.channel_splits)
        assert isinstance(out_channels, Dict), (
            f"Invalid out_channels: {type(out_channels)}" +
            f"({out_channels})\nExpected an instance of Sequnce or Dict."
        )
        injection_configs = {
            k: {
                'inc': self.tpm.out_channels[k], 'outc': n
            }
            for k, n in out_channels.items()
        }
        self.injection_keys = tuple(injection_configs.keys())

        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        self.trans = SemanticsExtractorModule(
            block_count=depths,
            embedding_dim=self.embed_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratios,
            attn_ratio=attn_ratios,
            drop_rate=0.0,
            drop_path=[
                x.item() for x in torch.linspace(0, drop_path_rate, depths)
            ],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        if self.injection_type:
            assert self.__module_registry.exists(
                alias=self.injection_type
            ), f"Unknown injection_type: {self.injection_type}"
            inj_module = self.__module_registry.get(alias=self.injection_type)
            self.post_stages = ParallelModule()
            for name, arg_dict in injection_configs.items():
                self.post_stages.add_module(
                    name=name,
                    module=inj_module(
                        inc=arg_dict['inc'],
                        outc=arg_dict['outc'],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                    )
                )
        else:
            self.post_stages = nn.Identity()

    def forward(self, x):
        pyramid_stages = self.tpm(x)
        gather = self.ppa(pyramid_stages)
        extract = self.trans(gather)
        if self.injection_type:
            groups = OrderedDict(
                [
                    (k, g)
                    for k, g in zip(
                        self.stage_names,
                        extract.split(self.channel_splits, dim=1)
                    )
                ]
            )
            extract = OrderedDict(
                [
                    (k, (pyramid_stages[k], groups[k]))
                    for k in self.injection_keys
                ]
            )
        semantics = self.post_stages(extract)
        return semantics


# noinspection SpellCheckingInspection
class TopFormerBackBone(nn.Module):
    __available_configs = {
        'T': {
            'stage_configs': {
                'stage_1': {
                    'resblock_1': {
                        'kernel_size': 3,
                        'expand_ratio': 1,
                        'base_channels': 16,
                        'stride': 1
                    },
                    'resblock_2': {
                        'kernel_size': 3,
                        'expand_ratio': 4,
                        'base_channels': 16,
                        'stride': 2
                    },
                    'resblock_3': {
                        'kernel_size': 3,
                        'expand_ratio': 3,
                        'base_channels': 16,
                        'stride': 1
                    }
                },
                'stage_2': {
                    'resblock_1': {
                        'kernel_size': 5,
                        'expand_ratio': 3,
                        'base_channels': 32,
                        'stride': 2
                    },
                    'resblock_2': {
                        'kernel_size': 5,
                        'expand_ratio': 3,
                        'base_channels': 32,
                        'stride': 1
                    }
                },
                'stage_3': {
                    'resblock_1': {
                        'kernel_size': 3,
                        'expand_ratio': 3,
                        'base_channels': 64,
                        'stride': 2
                    },
                    'resblock_2': {
                        'kernel_size': 3,
                        'expand_ratio': 3,
                        'base_channels': 64,
                        'stride': 1
                    }
                },
                'stage_4': {
                    'resblock_1': {
                        'kernel_size': 5,
                        'expand_ratio': 6,
                        'base_channels': 96,
                        'stride': 2
                    },
                    'resblock_2': {
                        'kernel_size': 5,
                        'expand_ratio': 6,
                        'base_channels': 96,
                        'stride': 1
                    }
                }
            },
            'out_channels': {
                'stage_1': 128,
                'stage_2': 128,
                'stage_3': 128,
                'stage_4': 128
            },
            'num_heads': 4,
            'c2t_stride': 2,
            'depths': 4,
            'key_dim': 16,
            'attn_ratios': 2,
            'mlp_ratios': 2,
            'drop_path_rate': 0.0,
            'norm_cfg': {'alias': 'batchnorm_2d'},
            'act_cfg': {'alias': 'relu6'},
            'injection_type': 'dot_sum'
        },
        'S': {
            'stage_configs': {
                'stage_1': {
                    'resblock_1': {
                        'kernel_size': 3,
                        'expand_ratio': 1,
                        'base_channels': 16,
                        'stride': 1
                    },
                    'resblock_2': {
                        'kernel_size': 3,
                        'expand_ratio': 4,
                        'base_channels': 24,
                        'stride': 2
                    },
                    'resblock_3': {
                        'kernel_size': 3,
                        'expand_ratio': 3,
                        'base_channels': 24,
                        'stride': 1
                    }
                },
                'stage_2': {
                    'resblock_1': {
                        'kernel_size': 5,
                        'expand_ratio': 3,
                        'base_channels': 48,
                        'stride': 2
                    },
                    'resblock_2': {
                        'kernel_size': 5,
                        'expand_ratio': 3,
                        'base_channels': 48,
                        'stride': 1
                    }
                },
                'stage_3': {
                    'resblock_1': {
                        'kernel_size': 3,
                        'expand_ratio': 3,
                        'base_channels': 96,
                        'stride': 2
                    },
                    'resblock_2': {
                        'kernel_size': 3,
                        'expand_ratio': 3,
                        'base_channels': 96,
                        'stride': 1
                    }
                },
                'stage_4': {
                    'resblock_1': {
                        'kernel_size': 5,
                        'expand_ratio': 6,
                        'base_channels': 128,
                        'stride': 2
                    },
                    'resblock_2': {
                        'kernel_size': 5,
                        'expand_ratio': 6,
                        'base_channels': 128,
                        'stride': 1
                    },
                    'resblock_3': {
                        'kernel_size': 3,
                        'expand_ratio': 6,
                        'base_channels': 128,
                        'stride': 1
                    }
                }
            },
            'out_channels': {
                'stage_1': 192,
                'stage_2': 192,
                'stage_3': 192,
                'stage_4': 192
            },
            'num_heads': 4,
            'c2t_stride': 2,
            'depths': 4,
            'key_dim': 16,
            'attn_ratios': 2,
            'mlp_ratios': 2,
            'drop_path_rate': 0.0,
            'norm_cfg': {'alias': 'batchnorm_2d'},
            'act_cfg': {'alias': 'relu6'},
            'injection_type': 'dot_sum'
        },
        'B': {
            'stage_configs': {
                'stage_1': {
                    'resblock_1': {
                        'kernel_size': 3,
                        'expand_ratio': 1,
                        'base_channels': 16,
                        'stride': 1
                    },
                    'resblock_2': {
                        'kernel_size': 3,
                        'expand_ratio': 4,
                        'base_channels': 32,
                        'stride': 2
                    },
                    'resblock_3': {
                        'kernel_size': 3,
                        'expand_ratio': 3,
                        'base_channels': 32,
                        'stride': 1
                    }
                },
                'stage_2': {
                    'resblock_1': {
                        'kernel_size': 5,
                        'expand_ratio': 3,
                        'base_channels': 64,
                        'stride': 2
                    },
                    'resblock_2': {
                        'kernel_size': 5,
                        'expand_ratio': 3,
                        'base_channels': 64,
                        'stride': 1
                    }
                },
                'stage_3': {
                    'resblock_1': {
                        'kernel_size': 3,
                        'expand_ratio': 3,
                        'base_channels': 128,
                        'stride': 2
                    },
                    'resblock_2': {
                        'kernel_size': 3,
                        'expand_ratio': 3,
                        'base_channels': 128,
                        'stride': 1
                    }
                },
                'stage_4': {
                    'resblock_1': {
                        'kernel_size': 5,
                        'expand_ratio': 6,
                        'base_channels': 160,
                        'stride': 2
                    },
                    'resblock_2': {
                        'kernel_size': 5,
                        'expand_ratio': 6,
                        'base_channels': 160,
                        'stride': 1
                    },
                    'resblock_3': {
                        'kernel_size': 3,
                        'expand_ratio': 6,
                        'base_channels': 160,
                        'stride': 1
                    }
                }
            },
            'out_channels': {
                'stage_1': 256,
                'stage_2': 256,
                'stage_3': 256,
                'stage_4': 256
            },
            'num_heads': 4,
            'c2t_stride': 2,
            'depths': 4,
            'key_dim': 16,
            'attn_ratios': 2,
            'mlp_ratios': 2,
            'drop_path_rate': 0.0,
            'norm_cfg': {'alias': 'batchnorm_2d'},
            'act_cfg': {'alias': 'relu6'},
            'injection_type': 'dot_sum'
        }
    }

    def __init__(
            self,
            config_alias: str = 'S',
            input_channels: int = 3,
            injection_type: str = 'dot_sum'
    ):
        config = self.__available_configs[config_alias]
        config['input_channels'] = input_channels
        config['injection_type'] = injection_type
        super(TopFormerBackBone, self).__init__()
        self.net = TopFormerModule(**config)

    def forward(self, x):
        return self.net(x)
