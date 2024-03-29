import copy
import warnings
from shutil import ExecError

import torch
import torch.nn.functional as F
from torch import nn

from ..builder import (CLASSIFIERS, build_backbone, build_head, build_loss,
                       build_neck)
from ..utils.augment import Augments
from .base import BaseClassifier
from ..losses import AT,RKD, CRDLoss


@CLASSIFIERS.register_module()
class KDImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 kd_loss,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(KDImageClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        assert 'student' in backbone.keys(), 'student network should be specified'
        assert 'teacher' in backbone.keys(), 'teacher network should be specified'

        return_tuple = backbone.pop('return_tuple', True)
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone['student']),
                'neck': build_neck(neck['student']),
                'head': build_head(head['student'])
            }
        )
        self.teacher = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone['teacher']),
                'neck': build_neck(neck['teacher']),
                'head': build_head(head['teacher'])
            }
        )
        self.criterionCls = F.cross_entropy
        self.criterionKD = build_loss(kd_loss)

        self.lambda_kd = train_cfg['lambda_kd']
        self.teacher_ckpt = train_cfg['teacher_checkpoint']
        if return_tuple is False:
            warnings.warn(
                'The `return_tuple` is a temporary arg, we will force to '
                'return tuple in the future. Please handle tuple in your '
                'custom neck or head.', DeprecationWarning)
        self.return_tuple = return_tuple
        self.load_teacher()
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

    def load_teacher(self):
        try:
            self.teacher.load_state_dict(
                torch.load(self.teacher_ckpt)['state_dict'])
            print(
                f'Teacher pretrained model has been loaded {self.teacher_ckpt}')
        except:
            ExecError('Teacher model not loaded')
        for param in self.teacher.parameters():
            param.requires_grad = False

    ###########################

    def get_logit(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.extract_feat(self.student, img)
        if isinstance(x, tuple):
            x = x[-1]
        logit = self.student['head'].fc(x)  # head
        return logit

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

        with torch.no_grad():
            teacher_logit = self.get_logits(self.teacher, img)
        student_logit = self.get_logits(self.student, img)

        loss_cls = self.criterionCls(student_logit, gt_label)
        loss_kd = self.criterionKD(
            student_logit, teacher_logit.detach()) * self.lambda_kd

        losses = dict(loss_cls=loss_cls,
                      loss_kd=loss_kd)

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
class HintImageClassifier(KDImageClassifier):
    def __init__(self, backbone, kd_loss, neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, kd_loss, neck, head, pretrained, train_cfg, init_cfg)
        self.lambda_feat = train_cfg['lambda_feat']
        self.feat_channels_student = train_cfg['feat_channels']['student']
        self.feat_channels_teacher = train_cfg['feat_channels']['teacher']
        feat_fcs = []
        for i in range(len(self.feat_channels_student)):
            feat_fcs.append(nn.Linear(
                    self.feat_channels_teacher[i], self.feat_channels_student[i])
                    )
        self.feat_fcs = nn.ModuleList(feat_fcs)
    def get_logits(self, model, img):
        """Directly extract features from the backbone + neck."""
        x = self.extract_feat(model, img)
        if isinstance(x, tuple):
            last_x = x[-1]
        logit = model['head'].fc(last_x)  # head
        return logit, x
    
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

        with torch.no_grad():
             _, teacher_x = self.get_logits(self.teacher, img)
        student_logit, student_x = self.get_logits(self.student, img)

        loss_cls = self.criterionCls(student_logit, gt_label)
        loss_kd = 0.0
        assert len(teacher_x) == len(student_x)
        for layer_id, (teacher_x_layer, student_x_layer) in enumerate(zip(teacher_x, student_x)):
            loss_kd += F.mse_loss(self.feat_fcs[layer_id](teacher_x_layer),
                                 student_x_layer) * self.lambda_feat

        losses = dict(loss_cls=loss_cls,
                      loss_kd=loss_kd)

        return losses

@CLASSIFIERS.register_module()
class CRDImageClassifier(KDImageClassifier):
    def __init__(self, backbone, kd_loss, neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, kd_loss, neck, head, pretrained, train_cfg, init_cfg)
        self.lambda_feat = train_cfg['lambda_feat']
        self.feat_channels_student = train_cfg['feat_channels']['student']
        self.feat_channels_teacher = train_cfg['feat_channels']['teacher']
        self.n_data = train_cfg['n_data']
        self.distll_fn = []
        for i in range(len(self.feat_channels_student)):
            self.distll_fn.append(CRDLoss(s_dim=self.feat_channels_student[i], t_dim=self.feat_channels_teacher[i], n_data=self.n_data))
    def get_logits(self, model, img):
        """Directly extract features from the backbone + neck."""
        x = self.extract_feat(model, img)
        if isinstance(x, tuple):
            last_x = x[-1]
        logit = model['head'].fc(last_x)  # head
        return logit, x
    
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

        with torch.no_grad():
             _, teacher_x = self.get_logits(self.teacher, img)
        student_logit, student_x = self.get_logits(self.student, img)

        loss_cls = self.criterionCls(student_logit, gt_label)
        loss_kd = 0.0
        assert len(teacher_x) == len(student_x)
        for layer_id, (teacher_x_layer, student_x_layer) in enumerate(zip(teacher_x, student_x)):
            loss_kd += self.distll_fn[layer_id](student_x_layer, teacher_x_layer) * self.lambda_feat

        losses = dict(loss_cls=loss_cls,
                      loss_kd=loss_kd)

        return losses


@CLASSIFIERS.register_module()
class ATImageClassifier(KDImageClassifier):
    def __init__(self, backbone, kd_loss, neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, kd_loss, neck, head, pretrained, train_cfg, init_cfg)
        self.lambda_feat = train_cfg['lambda_feat']
        self.distill_fn = AT(p=2)

    def get_logits(self, model, img):
        """Directly extract features from the backbone + neck."""
        pool_x, x = self.extract_feat(model, img)
        if isinstance(pool_x, tuple):
            last_x = pool_x[-1]
        logit = model['head'].fc(last_x)  # head
        return logit, x
    
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
        pool_x = model['neck'](x)
        return pool_x, x
    
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

        with torch.no_grad():
             _, teacher_x = self.get_logits(self.teacher, img)
        student_logit, student_x = self.get_logits(self.student, img)

        loss_cls = self.criterionCls(student_logit, gt_label)
        loss_kd = 0.0
        assert len(teacher_x) == len(student_x)
        for layer_id, (teacher_x_layer, student_x_layer) in enumerate(zip(teacher_x, student_x)):
            loss_kd += self.distill_fn(teacher_x_layer, student_x_layer) * self.lambda_feat

        losses = dict(loss_cls=loss_cls,
                      loss_kd=loss_kd)

        return losses
    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x, _ = self.extract_feat(self.student, img)

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
class RKDImageClassifier(KDImageClassifier):
    def __init__(self, backbone, kd_loss, neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, kd_loss, neck, head, pretrained, train_cfg, init_cfg)
        self.distill_fn = RKD()
        self.lambda_feat = train_cfg['lambda_feat']

    def get_logits(self, model, img):
        """Directly extract features from the backbone + neck."""
        x = self.extract_feat(model, img)
        if isinstance(x, tuple):
            last_x = x[-1]
        logit = model['head'].fc(last_x)  # head
        return logit, x
    
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

        with torch.no_grad():
             _, teacher_x = self.get_logits(self.teacher, img)
        student_logit, student_x = self.get_logits(self.student, img)

        loss_cls = self.criterionCls(student_logit, gt_label)
        loss_kd = 0.0
        assert len(teacher_x) == len(student_x)
        for layer_id, (teacher_x_layer, student_x_layer) in enumerate(zip(teacher_x, student_x)):
            loss_kd += self.distill_fn(teacher_x_layer, student_x_layer) * self.lambda_feat

        losses = dict(loss_cls=loss_cls,
                      loss_kd=loss_kd)

        return losses
