import torch
from torch import nn, Tensor
from modules.helpers import DropPath


class LayerNorm(nn.Module):
    """Channel first layer norm
    """

    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Block(nn.Module):
    def __init__(self, dim, dpr=0., init_value=1e-6):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, (7, 7), (1, 1), 3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw_conv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            init_value * torch.ones((dim,)), requires_grad=True
        ) if init_value > 0 else None
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        xinput = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)  # N.C.H.W -> N.H.W.C
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)
        x = xinput + self.drop_path(x)
        return x


class Stem(nn.Sequential):
    def __init__(self, c1, c2, k, s):
        super().__init__(
            nn.Conv2d(c1, c2, k, s),
            LayerNorm(c2)
        )


class DownSample(nn.Sequential):
    def __init__(self, c1, c2, k, s):
        super().__init__(
            LayerNorm(c1),
            nn.Conv2d(c1, c2, k, s)
        )


class ConvNeXt(nn.Module):

    __available_configs = {
        'X': [[2, 2, 4, 2], [32, 64, 128, 256], 0.0],
        'T': [[3, 3, 9, 3], [96, 192, 384, 768], 0.0],  # [depths, dims, dpr]
        'S': [[3, 3, 27, 3], [96, 192, 384, 768], 0.0],
        'B': [[3, 3, 27, 3], [128, 256, 512, 1024], 0.0]
    }

    def __init__(
            self,
            model_name: str = 'T',
            in_channels: int = 3
    ) -> None:
        super().__init__()
        assert (
                model_name in self.__available_configs.keys()
        ), (
                f"Unknown ConvNeXt variant '{model_name}'! " +
                f"Supported variants:{list(self.__available_configs.keys())}"
        )
        assert in_channels >= 1, "Input channels must be a positive integer!"
        depths, embed_dims, drop_path_rate = self.__available_configs[
            model_name
        ]
        self.channels = embed_dims

        self.downsample_layers = nn.ModuleList([
            Stem(in_channels, embed_dims[0], 4, 4),
            *[
                DownSample(
                    embed_dims[i], embed_dims[i + 1], 2, 2
                ) for i in range(3)
            ]
        ])

        self.stages = nn.ModuleList()
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(*[
                Block(embed_dims[i], dpr[cur + j])
                for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        for i in range(4):
            self.add_module(f"norm{i}", LayerNorm(embed_dims[i]))

    def forward(self, x: Tensor):
        outs = list()

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            norm_layer = getattr(self, f"norm{i}")
            outs.append(norm_layer(x))
        return outs
