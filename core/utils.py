from torch import nn as tnn


def init_weights(
        m: tnn.Module,
) -> None:
    if isinstance(m, tnn.Linear):
        tnn.init.kaiming_normal_(
            a=0.01,
            tensor=m.weight,
            mode='fan_in',
            nonlinearity='leaky_relu'
        )
        if m.bias is not None:
            tnn.init.zeros_(m.bias)
    elif isinstance(m, tnn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
        if m.bias is not None:
            tnn.init.zeros_(m.bias)
    elif isinstance(m, (tnn.LayerNorm, tnn.BatchNorm2d)):
        tnn.init.ones_(m.weight)
        tnn.init.zeros_(m.bias)
