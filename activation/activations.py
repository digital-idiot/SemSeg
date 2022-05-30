from torch import nn as tnn

__all__ = ['HSigmoid']


class HSigmoid(tnn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.relu = tnn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
