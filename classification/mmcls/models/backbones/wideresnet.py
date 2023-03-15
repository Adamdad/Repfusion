import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer)

from .resnet import ResNet
from ..builder import BACKBONES


class WideBasicBlock(nn.Module):
    """BasicBlock for ResNet.
    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module, optional): downsample operation on identity
            branch. Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict, optional): dictionary to construct and config conv
            layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 dropout=0,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN',
                               momentum=0.001,
                               requires_grad=True)):
        super(WideBasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dropout = dropout
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if self.dropout > 0:
            self.drop = nn.Dropout2d(p=self.dropout)
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.in_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            out = self.norm1(x)
            out = self.relu(out)
            out = self.conv1(out)

            if self.dropout > 0:
                out = self.drop(out)

            out = self.norm2(out)
            out = self.relu(out)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        # out = self.relu(out)

        return out

@BACKBONES.register_module()
class WideResNet_CIFAR(ResNet):
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of
    channels which is twice larger in every block. The number of channels
    in outer 1x1 convolutions is the same, e.g. last block in ResNet-50
    has 2048-512-2048 channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    arch_settings = {
        28: (WideBasicBlock, (4, 4, 4)),
    }

    def __init__(self, depth, out_channel, deep_stem=False,
                 norm_cfg=dict(type='BN',
                               momentum=0.1,
                               requires_grad=True),
                 **kwargs):
        super(WideResNet_CIFAR, self).__init__(
            depth,
            deep_stem=deep_stem,
            norm_cfg=norm_cfg, **kwargs)
        assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'
        self.norm_cfg = norm_cfg
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, out_channel, postfix=1)
        self.add_module(self.norm1_name, norm1)

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                if i == self.out_indices[-1]:
                    x = self.relu(self.norm1(x))
                else:
                    x = self.relu(x)
                outs.append(x)
        else:
            return tuple(outs)