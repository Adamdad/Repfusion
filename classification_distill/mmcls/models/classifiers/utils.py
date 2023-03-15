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


@CLASSIFIERS.register_module()
class KDDDPM_ImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 model_id = "google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1.0]],
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(KDDDPM_ImageClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        return_tuple = backbone.pop('return_tuple', True)
        self.build_model(backbone=backbone, neck=neck, head=head)

        # load model and scheduler
        self.init_ddpm_teacher(model_id)

        self.criterionCls = F.cross_entropy

        # assert isinstance(distill_fn, list)
        self.init_distill_func(distill_fn)
        self.init_distill_layers(teacher_layers, student_layers)

        if return_tuple is False:
            warnings.warn(
                'The `return_tuple` is a temporary arg, we will force to '
                'return tuple in the future. Please handle tuple in your '
                'custom neck or head.', DeprecationWarning)
        self.return_tuple = return_tuple
        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)
            else:
                # Considering BC-breaking
                mixup_cfg = train_cfg.get('mixup', None)
                cutmix_cfg = train_cfg.get('cutmix', None)
                assert mixup_cfg is None or cutmix_cfg is None, \
                    'If mixup and cutmix are set simultaneously,' \
                    'use augments instead.'
                if mixup_cfg is not None:
                    warnings.warn('The mixup attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(mixup_cfg)
                    cfg['type'] = 'BatchMixup'
                    # In the previous version, mixup_prob is always 1.0.
                    cfg['prob'] = 1.0
                    self.augments = Augments(cfg)
                if cutmix_cfg is not None:
                    warnings.warn('The cutmix attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(cutmix_cfg)
                    cutmix_prob = cfg.pop('cutmix_prob')
                    cfg['type'] = 'BatchCutMix'
                    cfg['prob'] = cutmix_prob
                    self.augments = Augments(cfg)

    def init_ddpm_teacher(self,model_id):
        self.ddpm = DDPMPipeline.from_pretrained(model_id).unet
        for param in self.ddpm.parameters():
            param.requires_grad = False
    def init_distill_func(self, distill_fn):
        distill_loss_list = []
        distill_loss_weight = []
        for dfn, weight in distill_fn:
            if dfn == 'l2':
                distill_loss_list.append(nn.MSELoss())
            if dfn == 'l2_norm':
                distill_loss_list.append(MSE_Norm_Loss())
            elif dfn == 'l1':
                distill_loss_list.append(nn.L1Loss())
            elif dfn == 'cosine':
                distill_loss_list.append(nn.CosineSimilarity(dim=1, eps=1e-6))
            else:
                AssertionError(f'{dfn} distillation loss is not implemented.')
            distill_loss_weight.append(weight)

        self.distill_fn = ComposeLoss(distill_loss_list, distill_loss_weight)

    def init_distill_layers(self, teacher_layers, student_layers):
        assert len(student_layers) == len(
            teacher_layers), "len(student_layers) must be equal to len(teacher_layers)"
        distill_layer = []
        self.teacher_hoods = ForwardHookManager()
        self.student_hoods = ForwardHookManager()
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_layer_name, t_layer_channel = t_layer
            s_layer_name, s_layer_channel = s_layer
            distill_layer.append(nn.Sequential(
                nn.Conv2d(s_layer_channel, t_layer_channel, 1),
            ))
            self.student_hoods.add_hook(self.student, s_layer_name)
            self.teacher_hoods.add_hook(self.ddpm, t_layer_name)

        # exit()
        self.distill_layer = nn.ModuleList(distill_layer)

    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck),
                'head': build_head(head)
            }
        )

        
    ###########################

    def get_logits(self, model, img):
        """Directly extract features from the backbone + neck."""
        x = self.extract_feat(model, img)
        if isinstance(x, tuple):
            x = x[-1]
        logit = model['head'].fc(x)  # head
        return logit

    def extract_feat(self, model, img):
        """Directly extract features from the backbone + neck."""
        x = model['backbone'](img)
        if self.return_tuple:
            if not isinstance(x, tuple):
                x = (x, )
                warnings.simplefilter('once')
                warnings.warn(
                    'We will force all backbones to return a tuple in the '
                    'future. Please check your backbone and wrap the output '
                    'as a tuple.', DeprecationWarning)
        else:
            if isinstance(x, tuple):
                x = x[-1]
        # if self.with_neck:
        x = model['neck'](x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        bs = img.shape[0]
        timesteps = torch.LongTensor([10]).repeat(bs).to(img.device)
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        student_logit = self.get_logits(self.student, img)

        loss_cls = self.criterionCls(student_logit, gt_label)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat, layer in zip(teacher_feat, student_feat, self.distill_layer):
            s_feat = layer(s_feat)
            # if s_feat.shape != t_feat.shape:
            s_feat = F.adaptive_avg_pool2d(s_feat, (1, 1)).view(bs, -1)
            t_feat = F.adaptive_avg_pool2d(t_feat, (1, 1)).view(bs, -1)

            # F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.distill_fn(s_feat, t_feat.detach()))
        # print(feat_loss)
        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_cls=loss_cls,
                      loss_feat=feat_loss)

        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(self.student, img)

        try:
            res = self.student['head'].simple_test(x)
        except TypeError as e:
            if 'not tuple' in str(e) and self.return_tuple:
                return TypeError(
                    'Seems the head cannot handle tuple input. We have '
                    'changed all backbones\' output to a tuple. Please '
                    'update your custom head\'s forward function. '
                    'Temporarily, you can set "return_tuple=False" in '
                    'your backbone config to disable this feature.')
            raise e

        return res
