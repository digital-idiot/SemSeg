import torch
from torch import nn
from abc import ABCMeta
from torch import Tensor
from copy import deepcopy
from abc import abstractmethod


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        kp = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor


class Registry(metaclass=ABCMeta):
    __registry = dict()

    @abstractmethod
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def register(cls, alias: str, layer: nn.Module, overwrite: bool = False):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, alias):
        pass

    @abstractmethod
    def exists(self, alias):
        pass


class ActivationRegistry(Registry):
    # noinspection SpellCheckingInspection
    __registry = {
            'relu': nn.ReLU,
            'relu6': nn.ReLU6,
            'selu': nn.SELU,
            'silu': nn.SiLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'mish': nn.Mish,
            'softmax2d': nn.Softmax2d,
            'elu': nn.ELU,
            'threshold': nn.Threshold,
            'glu': nn.GLU,
            'celu': nn.CELU,
            'prelu': nn.PReLU,
            'rrelu': nn.RReLU,
            'softmin': nn.Softmin,
            'softmax': nn.Softmax,
            'logsoftmax': nn.LogSoftmax,
            'softplus': nn.Softplus,
            'softsign': nn.Softsign,
            'tanh_shrink': nn.Tanhshrink,
            'soft_shrink': nn.Softshrink,
            'hard_shrink': nn.Hardshrink,
            'hard_sigmoid': nn.Hardsigmoid,
            'hard_tanh': nn.Hardtanh,
            'hard_swish': nn.Hardswish,
            'leaky_relu': nn.LeakyReLU,
            'log_sigmoid': nn.LogSigmoid,
            'adaptive_logsoftmax': nn.AdaptiveLogSoftmaxWithLoss
        }

    def __init__(self):
        # noinspection SpellCheckingInspection
        self._current_registry = deepcopy(self.__registry)

    @classmethod
    def register(cls, alias: str, layer: nn.Module, overwrite: bool = False):
        if overwrite or not(alias in cls.__registry.keys()):
            cls.__registry[alias] = layer
        else:
            raise AssertionError(
                f"Alias ({alias}) is already exist in the registry!" +
                "Try different alias or use overwrite flag."
            )

    def add(self, alias: str, layer: nn.Module, overwrite: bool = False):
        if overwrite or not self.exists(alias=alias):
            self._current_registry[alias] = layer
        else:
            raise AssertionError(
                f"Alias ({alias}) is already exist in the current registry!" +
                "Try different alias or use overwrite flag."
            )

    def __call__(self, alias: str, *args, **kwargs):
        assert self.exists(alias=alias), (
            f"Alias ({alias}) does not exist in the registry!"
        )
        return self.get(alias=alias)(*args, **kwargs)

    def get(self, alias: str):
        return self._current_registry.get(alias, None)

    @property
    def keys(self):
        return tuple(self._current_registry.keys())

    def exists(self, alias: str):
        return alias in self._current_registry

    def __str__(self):
        return f"Registered Aliases: {self.keys}"


class NormalizationRegistry(Registry):
    # noinspection SpellCheckingInspection
    __registry = {
        'batchnorm_1d': nn.BatchNorm1d,
        'batchnorm_2d': nn.BatchNorm2d,
        'batchnorm_3d': nn.BatchNorm3d,
        'lazybatchnorm_1d': nn.LazyBatchNorm1d,
        'lazybatchnorm_2d': nn.LazyBatchNorm2d,
        'lazybatchnorm_3d': nn.LazyBatchNorm3d,
        'instancenorm_1d': nn.InstanceNorm1d,
        'instancenorm_2d': nn.InstanceNorm2d,
        'instancenorm_3d': nn.InstanceNorm3d,
        'lazyinstancenorm_1d': nn.LazyInstanceNorm1d,
        'lazyinstancenorm_2d': nn.LazyInstanceNorm2d,
        'lazyinstancenorm_3d': nn.LazyInstanceNorm3d,
        'batchnorm_sync': nn.SyncBatchNorm,
        'groupnorm': nn.GroupNorm,
        'layernorm': nn.LayerNorm,
        'localresponsenorm': nn.LocalResponseNorm
    }

    def __init__(self):
        # noinspection SpellCheckingInspection
        self._current_registry = deepcopy(self.__registry)

    @classmethod
    def register(cls, alias: str, layer: nn.Module, overwrite: bool = False):
        if overwrite or not(alias in cls.__registry.keys()):
            cls.__registry[alias] = layer
        else:
            raise AssertionError(
                f"Alias ({alias}) is already exist in the registry!" +
                "Try different alias or use overwrite flag."
            )

    def add(self, alias: str, layer: nn.Module, overwrite: bool = False):
        if overwrite or not self.exists(alias=alias):
            self._current_registry[alias] = layer
        else:
            raise AssertionError(
                f"Alias ({alias}) is already exist in the current registry!" +
                "Try different alias or use overwrite flag."
            )

    def __call__(self, alias: str, *args, **kwargs):
        assert self.exists(alias=alias), (
            f"Alias ({alias}) does not exist in the registry!"
        )
        return self.get(alias=alias)(*args, **kwargs)

    def get(self, alias: str):
        return self._current_registry.get(alias, None)

    @property
    def keys(self):
        return tuple(self._current_registry.keys())

    def exists(self, alias: str):
        return alias in self._current_registry

    def __str__(self):
        return f"Registered Aliases: {self.keys}"


class ConvolutionRegistry(Registry):
    __registry = {
        'conv1d': nn.Conv1d,
        'conv2d': nn.Conv2d,
        'conv3d': nn.Conv3d,
        'lazy_conv1d': nn.LazyConv1d,
        'lazy_conv2d': nn.LazyConv2d,
        'lazy_conv3d': nn.LazyConv3d
    }

    def __init__(self):
        # noinspection SpellCheckingInspection
        self._current_registry = deepcopy(self.__registry)

    @classmethod
    def register(cls, alias: str, layer: nn.Module, overwrite: bool = False):
        if overwrite or not(alias in cls.__registry.keys()):
            cls.__registry[alias] = layer
        else:
            raise AssertionError(
                f"Alias ({alias}) is already exist in the registry!" +
                "Try different alias or use overwrite flag."
            )

    def add(self, alias: str, layer: nn.Module, overwrite: bool = False):
        if overwrite or not self.exists(alias=alias):
            self._current_registry[alias] = layer
        else:
            raise AssertionError(
                f"Alias ({alias}) is already exist in the current registry!" +
                "Try different alias or use overwrite flag."
            )

    def __call__(self, alias: str, *args, **kwargs):
        assert self.exists(alias=alias), (
            f"Alias ({alias}) does not exist in the registry!"
        )
        return self.get(alias=alias)(*args, **kwargs)

    def get(self, alias: str):
        return self._current_registry.get(alias, None)

    @property
    def keys(self):
        return tuple(self._current_registry.keys())

    def exists(self, alias: str):
        return alias in self._current_registry

    def __str__(self):
        return f"Registered Aliases: {self.keys}"
