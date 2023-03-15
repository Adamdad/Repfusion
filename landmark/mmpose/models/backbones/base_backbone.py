# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn
from mmcv.utils import print_log
# from .utils import load_checkpoint


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    def init_weights(self, pretrained=None):
        from mmcv.runner import (_load_checkpoint_with_prefix, load_checkpoint,
                    load_state_dict)
        """Init backbone weights.

        Args:
            pretrained (str | None): If pretrained is a string, then it
                initializes backbone weights by loading the pretrained
                checkpoint. If pretrained is None, then it follows default
                initializer or customized initializer in subclasses.
        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif isinstance(pretrained, dict):
            logger = logging.getLogger()
            checkpoint = pretrained.get('checkpoint', None)
            prefix = pretrained.get('prefix', None)
            map_location = pretrained.get('map_location', None)
            if prefix is None:
                print_log(f'load model from: {checkpoint}', logger=logger)
                load_checkpoint(
                    self,
                    checkpoint,
                    map_location=map_location,
                    strict=False,
                    logger=logger)
            else:
                print_log(
                    f'load {prefix} in model from: {checkpoint}',
                    logger=logger)
                state_dict = _load_checkpoint_with_prefix(
                    prefix, checkpoint, map_location=map_location)
                load_state_dict(self, state_dict, strict=False, logger=logger)
        elif pretrained is None:
            # use default initializer or customized initializer in subclasses
            pass
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')

    @abstractmethod
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
