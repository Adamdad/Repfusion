import copy
from turtle import forward
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from ..losses.norm_l2 import MSE_Norm_Loss

from ..builder import (CLASSIFIERS, build_backbone, build_head, build_loss,
                       build_neck)
from ..utils.augment import Augments
from .base import BaseClassifier
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from torch.distributions import Categorical

def transpose(x):
    return x.transpose(-2, -1)

def check_if_wrapped(model):
    return isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel))


def get_module(module, module_path):
    # if check_if_wrapped():
    #     module = module.module
    for name, m in module.named_modules():
        # print(name)
        if name == module_path:
            return m

    AssertionError('No module name {} found'.format(module_path))


class ComposeLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        assert len(losses) == len(
            weights), 'Number of loss must be the same as weights'
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        self.number_loss = len(weights)

    def forward(self, pred, target):
        loss_value = 0.0
        for i in range(self.number_loss):
            loss_value += self.losses[i](pred, target).mean() * self.weights[i]
        return loss_value


class FeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feat = None

    def hook_fn(self, module, input, output):
        # print(output)
        self.feat = output

    def close(self):
        self.hook.remove()


class ForwardHookManager(object):
    def __init__(self):
        self.hook_list = list()

    def add_hook(self, module, module_path, prefix=None):
        unwrapped_module = module.module if check_if_wrapped(
            module) else module
        if prefix is not None:
            module_path = f'{prefix}.{module_path}'
        sub_module = get_module(unwrapped_module, module_path)
        handle = FeatureHook(sub_module)
        self.hook_list.append(handle)

    def clear(self):
        self.hook_list.clear()


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~torch.nn.SiLU` for more details.
    """

    return torch.sigmoid(input) * input


class SiLU(nn.Module):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SiLU.png

    Examples::

        >>> m = nn.SiLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return silu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


torch.nn.SiLU = SiLU


