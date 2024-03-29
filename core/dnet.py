import torch
from torch import nn as tnn
from core.decoder import Decoder
from modules.layers import MultilabelClassifier
from torch.nn import functional as tnf
from core.image_encoder import ImageEncoder


class DNet(tnn.Module):
    def __init__(
            self,
            image_channels: int,
            num_classes: int,
            embed_dim: int = 256,
            stem_code: str = 'T',
    ):
        super(DNet, self).__init__()

        self.in_channels = image_channels
        self.out_channels = num_classes

        self.encoder = ImageEncoder(
            in_channels=image_channels,
            variant_code=stem_code
        )
        self.decoder = Decoder(
            in_channels=self.encoder.channels,
            out_channels=(2 * num_classes),
            embed_dim=embed_dim,
        )
        self.terminator = MultilabelClassifier(
            in_channels=(2 * num_classes),
            n_classes=num_classes
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        # noinspection PyArgumentList
        h, w = x.size(-2), x.size(-1)
        x = self.encoder(x)
        x = self.decoder(x)

        # Force match spatial dimension of image and features
        x = tnf.interpolate(
            input=x,
            size=(h, w),
            mode='nearest'
        )
        x = self.terminator(x)
        return x
