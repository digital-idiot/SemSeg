import torch
from typing import Any
import torch.nn as tnn
from typing import Dict
from modules.utils import get_shape
from collections import OrderedDict
from modules.blocks import ConvolutionBlock
from torch.nn.functional import interpolate
from modules.topformer import TopFormerBackBone
from modules.heads import HeadRegistry


HEAD_REGISTRY = HeadRegistry()


class TopFormerModel(tnn.Module):
    def __init__(
            self,
            num_classes: int,
            config_alias: str = 'B',
            input_channels: int = 3,
            injection_type: str = 'dot_sum',
            fusion: str = 'sum',
            head_cfg: Dict[str: Any] = frozenset({'alias': 'refiner'}.items())
    ):
        super(TopFormerModel, self).__init__()
        assert isinstance(fusion, str) and fusion.lower() in {
            'concat', 'sum'
        }, (
                f"Unknown fusion: {fusion}\n" +
                "Supported fusion modes are: {'concat', 'sum'}"
        )
        self.out_channels = num_classes
        self.backbone = TopFormerBackBone(
            config_alias=config_alias,
            input_channels=input_channels,
            injection_type=injection_type
        )
        head_count = len(self.backbone.net.stage_names)
        inj_channels = OrderedDict(
            [
                (k, conf['outc'])
                for k, conf in self.backbone.injection_configs.items()
            ]
        )
        if fusion.lower() == 'sum':
            inc = tuple(inj_channels.values())[0]
        else:
            inc = sum(inj_channels.values())

        head_cfg = dict(head_cfg)
        if head_cfg['alias'] == 'refiner':
            head_cfg['n_heads'] = head_count
            head_cfg['in_channels'] = inc
            head_cfg['embedding_dim'] = 2 * num_classes
            self.classifier = ConvolutionBlock(
                ndim=2,
                inc=inc,
                outc=num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                dilation=(1, 1),
                padding='auto',
                padding_mode='zeros',
                groups=num_classes,
                bias=False,
                norm_cfg=None,
                act_cfg=None,
                spectral_norm=False,
                order='CNA'
            )
        elif head_cfg['alias'] == 'simple':
            head_cfg['in_channels'] = inc
            head_cfg['embedding_dim'] = inc
            self.classifier = ConvolutionBlock(
                ndim=2,
                inc=inc,
                outc=num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                dilation=(1, 1),
                padding='auto',
                padding_mode='zeros',
                groups=1,
                bias=False,
                norm_cfg=None,
                act_cfg=None,
                spectral_norm=False,
                order='CNA'
            )
        else:
            NotImplementedError(
                f"Specified head ({head_cfg['alias']}) is unknown " +
                "or not compatible!"
            )

        self.head = HEAD_REGISTRY(**head_cfg)

    def forward(self, x: torch.Tensor):
        b, c, h, w = get_shape(x)
        x = self.backbone(x)
        x = tuple(x.values())
        x = self.head(x)
        # noinspection SpellCheckingInspection,PyArgumentList
        x = interpolate(
            input=x,
            size=(h, w),
            scale_factor=None,
            recompute_scale_factor=None,
            mode='bilinear',
            align_corners=False,
            antialias=False

        )
        return self.classifier(x)
