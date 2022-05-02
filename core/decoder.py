import torch
from torch import nn as tnn
from typing import Sequence
from modules.lawin import LawinHead
from torch.nn import functional as tnf
from modules.blocks import ConvolutionBlock
from modules.blocks import DeconvolutionBlock


# Upscale ~4x
class Decoder(tnn.Module):
    def __init__(
            self,
            in_channels: list,
            out_channels: int,
            embed_dim: int = 512,
    ):
        super(Decoder, self).__init__()
        self.decoder = LawinHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            out_channels=(2 * out_channels)
        )  # ~(H/4), ~(W/4)

        self.deconvolution = DeconvolutionBlock(
            inc=(2 * out_channels),
            outc=out_channels,
            ks=(3, 3),
            stride=(4, 4),
            dilation=(1, 1),
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
            activation=tnn.Mish(True)
        )  # ~(H), ~(W)

        self.feature_reducer = ConvolutionBlock(
            inc=(2 * out_channels),
            outc=out_channels,
            ks=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding_mode='zeros',
            groups=1,
            bias=True,
            activation=tnn.Mish(True)
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        x = self.decoder(features)
        a = self.deconvolution(x)  # ~4x
        # noinspection SpellCheckingInspection
        b = tnf.interpolate(
            input=x,
            size=(a.size(-2), a.size(-1)),
            mode='bilinear',
            align_corners=False
        )  # ~4x
        b = self.feature_reducer(b)
        return a + b
