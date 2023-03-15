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


@CLASSIFIERS.register_module()
class KDDDPM_ImageClassifierv2(KDDDPM_ImageClassifier):
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
            feat_loss.append(-self.distill_fn(s_feat, t_feat.detach()))
        # print(feat_loss)
        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_cls=loss_cls,
                      loss_feat=feat_loss)

        return losses


@CLASSIFIERS.register_module()
class KDDDPM_Policy_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self, backbone, teacher_layers, student_layers, model_id="google/ddpm-cifar10-32", distill_fn=[['l2', 1]], neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.policy_net = PolicyNet2()

    def get_logits(self, model, img):
        """Directly extract features from the backbone + neck."""
        x = self.extract_feat(model, img)
        if isinstance(x, tuple):
            x = x[-1]
        logit = model['head'].fc(x)  # head
        return logit, x
    
    def init_distill_layers(self, teacher_layers, student_layers):
        distill_layer = []
        self.teacher_hoods = ForwardHookManager()
        self.student_hoods = ForwardHookManager()
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_layer_name, t_layer_channel = t_layer
            s_layer_name, s_layer_channel = s_layer
            distill_layer.append(nn.Sequential(
                nn.Conv2d(s_layer_channel, s_layer_channel, 3, groups=s_layer_channel),
                nn.Conv2d(s_layer_channel, t_layer_channel, 1),
            ))
            self.student_hoods.add_hook(self.student, s_layer_name)
            self.teacher_hoods.add_hook(self.ddpm, t_layer_name)

        # exit()
        self.distill_layer = nn.ModuleList(distill_layer)

    def init_distill_func(self, distill_fn):
        distill_loss_list = []
        distill_loss_weight = []
        for dfn, weight in distill_fn:
            if dfn == 'l2':
                distill_loss_list.append(nn.MSELoss(reduction='none'))
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

        student_logit, x = self.get_logits(self.student, img)
        bs = img.shape[0]

        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布
        timesteps = m.sample()           # 从分布中采样
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
    
        loss_cls = self.criterionCls(student_logit, gt_label, reduction='none')

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
            feat_loss.append(F.mse_loss(s_feat, t_feat.detach(), reduction="none") * 0.1)
        # print(feat_loss)
        feat_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with smallest classification loss
        reward = -(feat_loss.mean(-1).detach())
        loss_policy = -(m.log_prob(timesteps) * reward).mean()


        losses = dict(loss_policy=loss_policy,
                      loss_cls=loss_cls,
                      reward = reward,
                      timesteps= timesteps.float(),
                      loss_feat=feat_loss)

        return losses
@CLASSIFIERS.register_module()
class KDDDPM_AC_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self, backbone, teacher_layers, student_layers, model_id="google/ddpm-cifar10-32", distill_fn=[['l2', 1]], neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.actor_net = PolicyNet2()
        self.critic_net = PolicyNet2()

    def get_logits(self, model, img):
        """Directly extract features from the backbone + neck."""
        x = self.extract_feat(model, img)
        if isinstance(x, tuple):
            x = x[-1]
        logit = model['head'].fc(x)  # head
        return logit, x
    
    def init_distill_layers(self, teacher_layers, student_layers):
        distill_layer = []
        self.teacher_hoods = ForwardHookManager()
        self.student_hoods = ForwardHookManager()
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_layer_name, t_layer_channel = t_layer
            s_layer_name, s_layer_channel = s_layer
            distill_layer.append(nn.Sequential(
                nn.Conv2d(s_layer_channel, s_layer_channel, 3, groups=s_layer_channel),
                nn.Conv2d(s_layer_channel, t_layer_channel, 1),
            ))
            self.student_hoods.add_hook(self.student, s_layer_name)
            self.teacher_hoods.add_hook(self.ddpm, t_layer_name)

        # exit()
        self.distill_layer = nn.ModuleList(distill_layer)

    def init_distill_func(self, distill_fn):
        distill_loss_list = []
        distill_loss_weight = []
        for dfn, weight in distill_fn:
            if dfn == 'l2':
                distill_loss_list.append(nn.MSELoss(reduction='none'))
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

        student_logit, x = self.get_logits(self.student, img)
        bs = img.shape[0]

        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布
        timesteps = m.sample()           # 从分布中采样
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
    
        loss_cls = self.criterionCls(student_logit, gt_label, reduction='none')

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
            feat_loss.append(F.mse_loss(s_feat, t_feat.detach(), reduction="none") * 0.1)
        # print(feat_loss)
        feat_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with smallest classification loss
        reward = -(feat_loss.mean(-1).detach())
        loss_policy = -(m.log_prob(timesteps) * reward).mean()


        losses = dict(loss_policy=loss_policy,
                      loss_cls=loss_cls,
                      reward = reward,
                      timesteps= timesteps.float(),
                      loss_feat=feat_loss)

        return losses
    

@CLASSIFIERS.register_module()
class KDDDPM_PolicyStrongWeak_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self, backbone, teacher_layers, student_layers, model_id="google/ddpm-cifar10-32", distill_fn=[['l2', 1]], neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.policy_net = PolicyNet2()

    def get_logits(self, model, img):
        """Directly extract features from the backbone + neck."""
        x = self.extract_feat(model, img)
        if isinstance(x, tuple):
            x = x[-1]
        logit = model['head'].fc(x)  # head
        return logit, x
    
    def init_distill_layers(self, teacher_layers, student_layers):
        distill_layer = []
        self.teacher_hoods = ForwardHookManager()
        self.student_hoods = ForwardHookManager()
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_layer_name, t_layer_channel = t_layer
            s_layer_name, s_layer_channel = s_layer
            distill_layer.append(nn.Sequential(
                nn.Conv2d(s_layer_channel, s_layer_channel, 3, groups=s_layer_channel),
                nn.Conv2d(s_layer_channel, t_layer_channel, 1),
            ))
            self.student_hoods.add_hook(self.student, s_layer_name)
            self.teacher_hoods.add_hook(self.ddpm, t_layer_name)

        # exit()
        self.distill_layer = nn.ModuleList(distill_layer)

    def init_distill_func(self, distill_fn):
        distill_loss_list = []
        distill_loss_weight = []
        for dfn, weight in distill_fn:
            if dfn == 'l2':
                distill_loss_list.append(nn.MSELoss(reduction='none'))
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

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img']['img'].data))

        return outputs

    def forward_train(self, img, strong, **kwargs):
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

        gt_label = img['gt_label']
        img = img['img']
        strong_img = strong['img']
        strong_label = strong['gt_label']

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        student_logit, x = self.get_logits(self.student, strong_img)
        bs = img.shape[0]

        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布
        timesteps = m.sample()           # 从分布中采样
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
    
        loss_cls = self.criterionCls(student_logit, gt_label, reduction='none')

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
            feat_loss.append(F.mse_loss(s_feat, t_feat.detach(), reduction="none") * 0.1)
        # print(feat_loss)
        feat_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with smallest classification loss
        reward = -(feat_loss.mean(-1).detach())
        loss_policy = -(m.log_prob(timesteps) * reward).mean()


        losses = dict(loss_policy=loss_policy,
                      loss_cls=loss_cls,
                      reward = reward,
                      timesteps= timesteps.float(),
                      loss_feat=feat_loss)

        return losses


@CLASSIFIERS.register_module()
class KDDDPM_Noise_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 distill_fn=[['l2', 1]],
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

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
        device = img.device
        bs = img.shape[0]
        # timesteps = torch.randint(0, 100, (bs,)).to(device)
        timesteps = torch.LongTensor([10]).repeat(bs).to(device)
        if torch.rand(1) > 0.5:
            noise = torch.randn(img.shape).to(device)
            noise_img = self.noise_scheduler.add_noise(
                img, noise, timesteps).to(device)
        else:
            noise_img = img

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        student_logit = self.get_logits(self.student, noise_img)

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

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_cls=loss_cls,
                      loss_feat=feat_loss)

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_ImageClassifierv2(KDDDPM_ImageClassifier):
    def __init__(self, 
                 backbone, 
                 teacher_layers, 
                 student_layers, 
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32", 
                 distill_fn=[['l2', 1]], neck=None, 
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck)
            }
        )

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
        device = img.device
        bs = img.shape[0]
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)
        # if torch.rand(1) > 0.5:
        noise = torch.randn(img.shape).to(device)
        noise_img = self.noise_scheduler.add_noise(
            img, noise, timesteps).to(device)
        # else:
        #     noise_img = img

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, noise_img)

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

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses
    

@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self, 
                 backbone, 
                 teacher_layers, 
                 student_layers, 
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32", 
                 distill_fn=[['l2', 1]], neck=None, 
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck)
            }
        )

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
        device = img.device
        bs = img.shape[0]
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)
        if torch.rand(1) > 0.5:
            noise = torch.randn(img.shape).to(device)
            noise_img = self.noise_scheduler.add_noise(
                img, noise, timesteps).to(device)
        else:
            noise_img = img

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, noise_img)

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

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_PretrainTriplet_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self, 
                 backbone, 
                 teacher_layers, 
                 student_layers, 
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32", 
                 distill_fn=[['l2', 1]], neck=None, 
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.distill_fn = nn.TripletMarginLoss(margin=10.0, p=2)

    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck)
            }
        )

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
        device = img.device
        bs = img.shape[0]
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)
        if torch.rand(1) > 0.5:
            noise = torch.randn(img.shape).to(device)
            noise_img = self.noise_scheduler.add_noise(
                img, noise, timesteps).to(device)
        else:
            noise_img = img

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, noise_img)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat, layer in zip(teacher_feat, student_feat, self.distill_layer):
            s_feat = layer(s_feat)
            # if s_feat.shape != t_feat.shape:
            s_feat = F.adaptive_avg_pool2d(s_feat, (1, 1)).view(bs, -1)
            s_feat_neg = s_feat[torch.randperm(s_feat.size(0))]
            t_feat = F.adaptive_avg_pool2d(t_feat, (1, 1)).view(bs, -1)
            # F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.distill_fn(t_feat.detach(), s_feat, s_feat_neg))

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_ImageClassifier_SingleT(KDDDPM_ImageClassifier):
    def __init__(self, 
                 backbone, 
                 teacher_layers, 
                 student_layers, 
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32", 
                 distill_fn=[['l2', 1]], neck=None, 
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck)
            }
        )

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
        device = img.device
        bs = img.shape[0]
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)
        if torch.rand(1) > 0.5:
            noise = torch.randn(img.shape).to(device)
            noise_img = self.noise_scheduler.add_noise(
                img, noise, timesteps).to(device)
        else:
            noise_img = img

        with torch.no_grad():
            t = torch.LongTensor([self.max_time_step]).repeat(bs).to(device)
            _ = self.ddpm(img, t, return_dict=False)
        _ = self.extract_feat(self.student, noise_img)

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

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Dense_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self, 
                 backbone, 
                 teacher_layers, 
                 student_layers, 
                 model_id="google/ddpm-cifar10-32", 
                 max_time_step = 1000,
                 distill_fn=[['l2', 1]], neck=None,
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck)
            }
        )

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
        device = img.device
        bs = img.shape[0]
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)
        if torch.rand(1) > 0.5:
            noise = torch.randn(img.shape).to(device)
            noise_img = self.noise_scheduler.add_noise(
                img, noise, timesteps).to(device)
        else:
            noise_img = img

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, noise_img)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat, layer in zip(teacher_feat, student_feat, self.distill_layer):
            s_feat = layer(s_feat)
            if s_feat.shape != t_feat.shape:
                s_feat = F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            
            assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.distill_fn(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Dense_StrongWeak_ImageClassifier(KDDDPM_Pretrain_Dense_ImageClassifier):
    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img']['img'].data))

        return outputs

    def forward_train(self, img, strong, **kwargs):
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

        gt_label = img['gt_label']
        img = img['img']
        strong_img = strong['img']
        strong_label = strong['gt_label']


        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)
        device = img.device
        bs = img.shape[0]
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, strong_img)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat, layer in zip(teacher_feat, student_feat, self.distill_layer):
            s_feat = layer(s_feat)
            if s_feat.shape != t_feat.shape:
                s_feat = F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            
            assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.distill_fn(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Dense_StrongWeak_ImageClassifierv2(KDDDPM_Pretrain_Dense_StrongWeak_ImageClassifier):
    def __init__(self, backbone, teacher_layers, student_layers, model_id="google/ddpm-cifar10-32", max_time_step=1000, distill_fn=[['l2', 1]], neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, max_time_step, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.distill_fn = nn.TripletMarginLoss(margin=10.0, p=2)
        
    def forward_train(self, img, strong, **kwargs):
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

        gt_label = img['gt_label']
        img = img['img']
        strong_img = strong['img']
        strong_label = strong['gt_label']


        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)
        device = img.device
        bs = img.shape[0]
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, strong_img)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat, layer in zip(teacher_feat, student_feat, self.distill_layer):
            s_feat = layer(s_feat)
            if s_feat.shape != t_feat.shape:
                s_feat = F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            
            s_feat = s_feat.view(s_feat.size(0)*s_feat.size(2)*s_feat.size(3), -1)
            s_feat_neg = s_feat[torch.randperm(s_feat.size(0))]
            t_feat = t_feat.view(t_feat.size(0)*t_feat.size(2)*t_feat.size(3), -1)
            assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.distill_fn(t_feat.detach(), s_feat, s_feat_neg))

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Policy_Dense_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self, 
                 backbone, 
                 teacher_layers, 
                 student_layers, 
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]], 
                 neck=None, 
                 head=None, 
                 pretrained=None, 
                 train_cfg=None, 
                 init_cfg=None):
        super().__init__(backbone, 
                         teacher_layers, 
                         student_layers, 
                         model_id, 
                         distill_fn, 
                         neck, 
                         head, 
                         pretrained, 
                         train_cfg, 
                         init_cfg)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.policy_net = PolicyNet2()

    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck)
            }
        )

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
        device = img.device
        bs = img.shape[0]

        _ = self.extract_feat(self.student, img)
        
        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布

        timesteps = m.sample()           # 从分布中采样
        
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)


        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat, layer in zip(teacher_feat, student_feat, self.distill_layer):
            s_feat = layer(s_feat)
            if s_feat.shape != t_feat.shape:
                s_feat = F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            
            assert s_feat.shape == t_feat.shape
            feat_loss.append(self.distill_fn(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)
        # Find the time with largest MI
        t_feat = t_feat.view(t_feat.size(0)*t_feat.size(2)*t_feat.size(3), -1)
        reward = - (self.info_nce(t_feat.detach(), t_feat.detach()) ) # + feat_loss
        # reward_norm = reward - reward.mean()
        loss_policy = -(m.log_prob(timesteps) * reward).mean()


        losses = dict(loss_feat=feat_loss,
                      reward = reward,
                      timesteps=timesteps.float(),
                      loss_policy=loss_policy)

        return losses


@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Dense_ImageClassifierv2(KDDDPM_ImageClassifier):
    def __init__(self, backbone, teacher_layers, student_layers, max_time_step=1000, 
                 model_id="google/ddpm-cifar10-32", distill_fn=[['l2', 1]], 
                 neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.distill_fn = nn.TripletMarginLoss(margin=10.0, p=2)

    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck)
            }
        )

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
        device = img.device
        bs = img.shape[0]
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)
        if torch.rand(1) > 0.5:
            noise = torch.randn(img.shape).to(device)
            noise_img = self.noise_scheduler.add_noise(
                img, noise, timesteps).to(device)
        else:
            noise_img = img

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, noise_img)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat, layer in zip(teacher_feat, student_feat, self.distill_layer):
            s_feat = layer(s_feat)
            if s_feat.shape != t_feat.shape:
                s_feat = F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            
            s_feat = s_feat.view(s_feat.size(0)*s_feat.size(2)*s_feat.size(3), -1)
            s_feat_neg = s_feat[torch.randperm(s_feat.size(0))]
            t_feat = t_feat.view(t_feat.size(0)*t_feat.size(2)*t_feat.size(3), -1)
            assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.distill_fn(t_feat.detach(), s_feat, s_feat_neg))

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses


@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_ImageClassifier_WeakStrong(KDDDPM_Pretrain_ImageClassifier):
    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img']['img'].data))

        return outputs

    def forward_train(self, img, strong, **kwargs):
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

        gt_label = img['gt_label']
        img = img['img']
        strong_img = strong['img']
        strong_label = strong['gt_label']

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)
        device = img.device
        bs = img.shape[0]
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)
        if torch.rand(1) > 0.5:
            noise = torch.randn(img.shape).to(device)
            noise_img = self.noise_scheduler.add_noise(
                strong_img, noise, timesteps).to(device)
        else:
            noise_img = strong_img

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, noise_img)

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

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses


class Time_distill_layer(nn.Module):
    def __init__(self, s_layer_channel, t_layer_channel, block_out_channels=64, flip_sin_to_cos=True, freq_shift=0) -> None:
        super().__init__()
        self.time_proj = Timesteps(block_out_channels, flip_sin_to_cos, freq_shift)
        self.time_embeding = TimestepEmbedding(block_out_channels, block_out_channels * 4)

        self.layers = nn.Sequential(
                nn.Linear(s_layer_channel, t_layer_channel),
                nn.GELU(),
                nn.Linear(s_layer_channel, t_layer_channel),
            )
    def forward(self, x, timestep):
        bs = x.shape[0]
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(bs, -1)
        x = self.layers(x)
        t_emb = self.time_proj(timestep)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        t_emb = self.time_embedding(t_emb)
        x = x + t_emb
        return


@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_ImageClassifier_Timecond(KDDDPM_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 distill_fn=[['l2', 1]],
                 max_time_step=1000,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def init_distill_layers(self, teacher_layers, student_layers):
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
                'neck': build_neck(neck)
            }
        )

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
        device = img.device
        bs = img.shape[0]
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)
        if torch.rand(1) > 0.5:
            noise = torch.randn(img.shape).to(device)
            noise_img = self.noise_scheduler.add_noise(
                img, noise, timesteps).to(device)
        else:
            noise_img = img

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, noise_img)

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

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_T_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self, backbone, teacher_layers, student_layers, model_id="google/ddpm-cifar10-32", distill_fn=[['l2', 1]], neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.policy_net = PolicyNet()
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.distill_fn = nn.MSELoss(reduction='none')

    def get_logits(self, model, img):
        """Directly extract features from the backbone + neck."""
        x = self.extract_feat(model, img)
        if isinstance(x, tuple):
            x = x[-1]
        return x
    
    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck)
            }
        )
        
    def info_nce(self, query, positive_key, temperature=0.1, reduction='none'):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')

        # Normalize to unit vectors
        # query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
   
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)
    
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
        device = img.device
        bs = img.shape[0]

        x = self.get_logits(self.student, img)
        
        probs = self.policy_net(x.detach())
        m = Categorical(probs)      # 生成分布
        # epsilon greedy
        if torch.rand() > 0.1:
            timesteps = m.sample()           # 从分布中采样
        else:
            timesteps = torch.randint(0, 1000, (bs,)).to(device)
        
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)

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
            feat_loss.append(self.info_nce(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with largest MI
        reward = - (self.info_nce(t_feat.detach(), t_feat.detach()) ) # + feat_loss
        reward_norm = reward - reward.mean()
        loss_policy = -(m.log_prob(timesteps) * reward_norm).mean()

        print(timesteps, reward)

        losses = dict(loss_feat=feat_loss,
                      reward = reward,
                      loss_policy=loss_policy)

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_T_ImageClassifierv2(KDDDPM_T_ImageClassifier):
    def __init__(self, backbone, teacher_layers, student_layers, model_id="google/ddpm-cifar10-32", distill_fn=[['l2', 1]], neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.policy_net = PolicyNet2()

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
        device = img.device
        bs = img.shape[0]

        x = self.get_logits(self.student, img)
        
        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布

        timesteps = m.sample()           # 从分布中采样
        
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)

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
            feat_loss.append(self.info_nce(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with largest MI
        reward = - (self.info_nce(t_feat.detach(), t_feat.detach()) ) # + feat_loss
        # reward_norm = reward - reward.mean()
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        # print(timesteps, reward)

        losses = dict(loss_feat=feat_loss,
                      reward = reward,
                      timesteps = timesteps.float(),
                      loss_policy=loss_policy)

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_T_ImageClassifierv3(KDDDPM_T_ImageClassifier):
    def __init__(self, backbone, teacher_layers, student_layers, model_id="google/ddpm-cifar10-32", distill_fn=[['l2', 1]], neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.policy_net = PolicyNet2()

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
        device = img.device
        bs = img.shape[0]

        x = self.get_logits(self.student, img)
        
        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布

        timesteps = m.sample()           # 从分布中采样
        
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)

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
            feat_loss.append(self.info_nce(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with largest MI
        reward = - (self.info_nce(t_feat.detach(), t_feat.detach()) - feat_loss.detach())
        # reward_norm = reward - reward.mean()
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        # print(timesteps, reward)

        losses = dict(loss_feat=feat_loss,
                      reward = reward,
                      timesteps = timesteps.float(),
                      loss_policy=loss_policy)

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_T_ImageClassifierv2_Strongweak(KDDDPM_T_ImageClassifierv2):
    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img']['img'].data))

        return outputs

    def forward_train(self, img, strong, **kwargs):
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

        gt_label = img['gt_label']
        img = img['img']
        strong_img = strong['img']
        strong_label = strong['gt_label']

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)
        device = img.device
        bs = img.shape[0]

        x = self.get_logits(self.student, strong_img)
        
        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布

        timesteps = m.sample()           # 从分布中采样
        
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)

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
            feat_loss.append(self.info_nce(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with largest MI
        reward = - (self.info_nce(t_feat.detach(), t_feat.detach()) ) # + feat_loss
        # reward_norm = reward - reward.mean()
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        losses = dict(loss_feat=feat_loss,
                      reward = reward,
                      timesteps = timesteps.float(),
                      loss_policy=loss_policy)

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_T_ImageClassifierv3_Strongweak(KDDDPM_T_ImageClassifierv2_Strongweak):
     def forward_train(self, img, strong, **kwargs):
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

        gt_label = img['gt_label']
        img = img['img']
        strong_img = strong['img']
        strong_label = strong['gt_label']

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)
        device = img.device
        bs = img.shape[0]

        x = self.get_logits(self.student, strong_img)
        
        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布

        timesteps = m.sample()           # 从分布中采样
        
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)

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
            feat_loss.append(self.info_nce(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with largest MI
        reward = - (self.info_nce(t_feat.detach(), t_feat.detach()) - feat_loss.detach()) # + feat_loss
        # reward_norm = reward - reward.mean()
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        losses = dict(loss_feat=feat_loss,
                      reward = reward,
                      timesteps = timesteps.float(),
                      loss_policy=loss_policy)

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_ImageClassifier_T_Strongweak(KDDDPM_T_ImageClassifier):
    def __init__(self, backbone, teacher_layers, student_layers, model_id="google/ddpm-cifar10-32", distill_fn=[['l2', 1]], neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)

    
    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img']['img'].data))

        return outputs

    def forward_train(self, img, strong, **kwargs):
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

        gt_label = img['gt_label']
        img = img['img']
        strong_img = strong['img']
        strong_label = strong['gt_label']

        if self.augments is not None:
            img, gt_label = self.augments(strong_img, strong_label)
        device = img.device
        bs = img.shape[0]

        x = self.get_logits(self.student, strong_img)

        probs = self.policy_net(x.detach())
        m = Categorical(probs)      # 生成分布
        # epsilon greedy
        if torch.rand(1) > 0.1:
            timesteps = m.sample()           # 从分布中采样
        else:
            timesteps = torch.randint(0, 1000, (bs,)).to(device)
        
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)

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
            feat_loss.append(self.info_nce(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with largest MI
        reward = - (self.info_nce(t_feat.detach(), t_feat.detach()) + feat_loss) # 
        reward_norm = reward - reward.mean()
        loss_policy = -(m.log_prob(timesteps) * reward_norm).mean()

        # print(probs, reward, m.log_prob(timesteps))
        print(timesteps, reward)

        losses = dict(loss_feat=feat_loss,
                      reward = reward,
                      loss_policy=loss_policy)

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_ImageClassifier_T_Strongweak2(KDDDPM_Pretrain_ImageClassifier_T_Strongweak):
    def forward_train(self, img, strong, **kwargs):
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

        gt_label = img['gt_label']
        img = img['img']
        strong_img = strong['img']
        strong_label = strong['gt_label']

        if self.augments is not None:
            img, gt_label = self.augments(strong_img, strong_label)
        device = img.device
        bs = img.shape[0]

        x = self.get_logits(self.student, strong_img)

        probs = self.policy_net(x.detach())
        m = Categorical(probs)      # 生成分布
        timesteps = m.sample()           # 从分布中采样
        
        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)

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

        feat_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with largest MI
        reward = - self.info_nce(t_feat.detach(), t_feat.detach())
        reward_norm = reward - reward.mean()
        loss_policy = -(m.log_prob(timesteps) * reward_norm).mean()
        print(timesteps)
        losses = dict(loss_feat=feat_loss,
                      reward = reward,
                      loss_policy=loss_policy)

        return losses

class PolicyNet(nn.Module):
    def __init__(self, input_dim=512, action_num=1000):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_num)  # Prob of Left

    def forward(self, x):

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class PolicyNet2(nn.Module):
    def __init__(self, input_dim=3072, action_num=1000):
        super(PolicyNet2, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_num)  # Prob of Left

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class CriticNet(nn.Module):
    def __init__(self, input_dim=3072):
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Prob of Left

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x