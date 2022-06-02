import torch
import torch.nn as tnn
from modules.utils import get_shape
from collections import OrderedDict
from modules.heads import SimpleHead
from modules.blocks import ConvolutionBlock
from torch.nn.functional import interpolate
from modules.topformer import TopFormerBackBone


class TopFormerModel(tnn.Module):
    def __init__(
            self,
            num_classes: int,
            config_alias: str = 'B',
            input_channels: int = 3,
            injection_type: str = 'dot_sum',
            fusion: str = 'sum'
    ):
        super(TopFormerModel, self).__init__()
        assert isinstance(fusion, str) and fusion.lower() in {
            'concat', 'sum'
        }, (
                f"Unknown fusion: {fusion}\n" +
                "Supported fusion modes are: {'concat', 'sum'}"
        )
        self.backbone = TopFormerBackBone(
            config_alias=config_alias,
            input_channels=input_channels,
            injection_type=injection_type
        )
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

        # noinspection SpellCheckingInspection
        self.head = SimpleHead(
            in_channels=inc,
            embedding_dim=inc,
            resize_mode='bilinear',
            fusion='sum',
            norm_cfg={'alias': 'batchnorm_2d'},
            act_cfg={'alias': 'relu6'}
        )

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
