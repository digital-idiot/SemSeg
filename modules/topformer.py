import torch
from torch import nn
from typing import Any
from typing import Dict
from typing import Union
from typing import Sequence
from typing import FrozenSet
import torch.nn.functional as tnf
from collections import OrderedDict
from modules.utils import get_shape
from modules.helpers import DropPath
from modules.layers import BranchModule
from modules.utils import make_divisible
from modules.layers import StagingModule
from modules.layers import ParallelModule
from modules.blocks import ConvolutionBlock


class Mlp(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            drop_rate: float = 0.0,
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = frozenset({'alias': 'relu'}.items()),
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items())
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc_a = ConvolutionBlock(
            ndim=2,
            inc=in_features,
            outc=out_features,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            groups=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
            order='CNA'
        )

        self.dwconv = ConvolutionBlock(
            ndim=2,
            inc=hidden_features,
            outc=hidden_features,
            kernel_size=(3, 3),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            groups=hidden_features,
            norm_cfg=None,
            act_cfg=act_cfg,
            order='CNA'
        )

        self.fc_b = ConvolutionBlock(
            ndim=2,
            inc=hidden_features,
            outc=out_features,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding='auto',
            groups=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
            order='CNA'
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc_a(x)
        x = self.dwconv(x)
        x = self.dropout(x)
        x = self.fc_b(x)
        x = self.dropout(x)
        return x


class InvertedResidual(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            inc: int,
            outc: int,
            ks: int,
            stride: int,
            expand_ratio: int,
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = frozenset({'alias': 'relu'}.items()),
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
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
        self.conv = nn.Sequential()
        if expand_ratio != 1:
            # noinspection PyTypeChecker
            self.conv.add_module(
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
                    padding='auto'
                )
            )
        # noinspection PyTypeChecker
        self.conv.add_module(
            name='conv_dw',
            module=ConvolutionBlock(
                ndim=2,
                inc=hidden_dim,
                outc=hidden_dim,
                kernel_size=ks,
                stride=(1, 1),
                dilation=(1, 1),
                groups=hidden_dim,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                padding='auto'
            )
        )
        # noinspection PyTypeChecker
        self.conv.add_module(
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
                padding='auto'
            )
        )

    def forward(self, x):
        if self.residual_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TokenPyramidModule(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            stage_configs: Union[
                OrderedDict[str, Dict[str, Any]],
                Sequence[Dict[str, Any]]
            ],
            # token_subset: Union[Set[str], Set[int]] = None,
            input_channels: int = 3,
            init_features: int = 16,
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = frozenset({'alias': 'relu'}.items()),
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
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

        self.stem = nn.Sequential(
            OrderedDict(
                [
                    (
                        'stem',
                        ConvolutionBlock(
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
                ]
            )
        )

        self.staging_module = StagingModule()
        for name, params_dict in stage_configs.items():
            layer_params = {
                "inc": init_features,
                "act_cfg": act_cfg,
                "norm_cfg": norm_cfg
            }
            params_dict['outc'] = make_divisible(
                v=(params_dict.pop('basec') * width_multiplier),
                divisor=8,
                min_value=8
            )
            layer_params.update(params_dict)
            self.staging_module.add_module(
                name=name,
                module=InvertedResidual(**layer_params)
            )
            init_features = params_dict['outc']

    def forward(self, x):
        x = self.stem(x)
        return self.staging_module(x)


class Attention(torch.nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            dim: int,
            key_dim: int,
            num_heads: int,
            attn_ratio: Union[float, int] = 4.0,
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = None,
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
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

        self.proj = ConvolutionBlock(
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
        return self.proj(x)


class Block(nn.Module):

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
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = frozenset({'alias': 'relu'}.items()),
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items())
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

        self.drop_path = DropPath(
            drop_path
        ) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
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


class BasicLayer(nn.Module):
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
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = None
    ):
        super().__init__()
        self.block_count = block_count
        self.transformer_sequence = nn.Sequential()

        # self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_count):
            self.transformer_sequence.add_module(
                name=f'block_{i+1}',
                module=Block(
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
            norm_cfg=frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg=frozenset({'alias': 'hsigmoid'}.items())
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
    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
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
            norm_cfg=frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg=frozenset({'alias': 'hsigmoid'}.items())
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
    def forward(self, x_l, x_g):
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
            norm_cfg=frozenset({'alias': 'batchnorm_2d'}.items())
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
    def forward(self, x_l, x_h):
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
            norm_cfg=frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg=frozenset({'alias': 'hsigmoid'}.items())
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
    def forward(self, x_l, x_h):
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


# TODO: Create a registry
SIM_BLOCK = {
    "fuse_sum": FuseBlockSum,
    "fuse_multi": FuseBlockDot,

    "multi_sum": InjectionDotSum,
    "multi_sum_cbr": InjectionDotSumCBR,
}


class TopFormer(nn.Module):
    # noinspection SpellCheckingInspection
    def __init__(
            self,
            input_channels: int,
            stage_configs: Union[
                OrderedDict[str, Dict[str, Any]],
                Sequence[Dict[str, Any]]
            ],
            out_channels: Sequence[int],
            channel_splits: Sequence[int],
            token_subset: Sequence[int] = None,
            depths: int = 4,
            key_dim: int = 16,
            num_heads: int = 8,
            attn_ratios: float = 2,
            mlp_ratios: float = 2,
            c2t_stride: int = 2,
            drop_path_rate: float = 0.0,
            norm_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = frozenset({'alias': 'batchnorm_2d'}.items()),
            act_cfg: Union[
                Dict[str, Any], FrozenSet[Sequence[str, Any]]
            ] = frozenset({'alias': 'relu6'}.items()),
            injection_type: str = "multi_sum",
            injection: bool = True
    ):
        super().__init__()
        if token_subset is None:
            token_subset = tuple(range(len(stage_configs)))

        assert len(channel_splits) == len(token_subset) == len(
            out_channels
        ) <= len(stage_configs), (
            "Invalid config encountered!\n" +
            f"\ttoken_count: {len(token_subset)}\n" +
            f"\tupsamplig_stages: {len(channel_splits)}\n" +
            f"\tupsampling_channels: {len(out_channels)}" +
            f"\tpyramid_stages: {len(stage_configs)}"
        )

        if isinstance(stage_configs, Sequence):
            stage_configs = OrderedDict(
                [
                    (f"stage_{i+1}", conf)
                    for i, conf in enumerate(stage_configs)
                ]
            )
        stage_names = tuple(stage_configs.keys())
        self.stage_configs = stage_configs
        self.token_subset = token_subset
        self.channel_splits = channel_splits
        self.norm_cfg = norm_cfg
        self.injection = injection
        self.embed_dim = sum(channel_splits)

        self.tpm = TokenPyramidModule(
            stage_configs=stage_configs,
            input_channels=input_channels,
            init_features=16,
            act_cfg=frozenset({'alias': 'relu'}.items()),
            norm_cfg=norm_cfg,
            width_multiplier=1.0
        )
        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        self.trans = BasicLayer(
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

        inj_module = SIM_BLOCK[injection_type]
        self.upsampling_stages = ParallelModule()
        for i, inc, outc in zip(token_subset, channel_splits, out_channels):
            self.upsampling_stages.add_module(
                name=stage_names[i],
                module=inj_module(
                    inc=inc,
                    outc=outc,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

    def forward(self, x):
        outputs = self.tpm(x)
        out = self.ppa(outputs)
        out = self.trans(out)
        if self.injection:
            out = out.split(self.channel_splits, dim=1)
            results = OrderedDict()
            for i, (k, t) in enumerate(outputs.items()):
                if i in self.token_subset:
                    results[k] = (t, out[i])
            results = self.upsampling_stages(results)
            return results
        else:
            return out
