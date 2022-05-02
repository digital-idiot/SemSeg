import torch
from torch import nn as tnn
from modules.convnext import ConvNeXt


class ImageEncoder(tnn.Module):
    def __init__(
            self,
            in_channels: int,
            variant_code: str = 'T',
    ):
        super(ImageEncoder, self).__init__()
        self.encoder = ConvNeXt(
            model_name=variant_code, in_channels=in_channels
        )
        self.channels = self.encoder.channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
