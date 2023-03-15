import random
from curses import noecho

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from mmcls.models.guided_diffusion.load_model import create_imagenet256, create_imagenet64
from torch import nn, no_grad
from torch.distributions import Categorical

from ..builder import (CLASSIFIERS, build_backbone, build_head, build_loss,
                       build_neck)
from ..losses import AT, RKD
from ..losses.norm_l2 import MSE_Norm_Loss
from .kd_ddpm import (ComposeLoss, ForwardHookManager, KDDDPM_ImageClassifier,
                      transpose)


@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_CleanDense_AT_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]], neck=None,
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.kd_weight = train_cfg['kd_weight']
        # self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck)
            }
        )
    def init_distill_func(self, distill_fn):
        self.distill_fn = AT(p=2)
    def init_distill_layers(self, teacher_layers, student_layers):
        assert len(student_layers) == len(
            teacher_layers), "len(student_layers) must be equal to len(teacher_layers)"
        # distill_layer = []
        self.teacher_hoods = ForwardHookManager()
        self.student_hoods = ForwardHookManager()
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_layer_name, _ = t_layer
            s_layer_name, _ = s_layer
            self.student_hoods.add_hook(self.student, s_layer_name)
            self.teacher_hoods.add_hook(self.ddpm, t_layer_name)

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

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, img)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat in zip(teacher_feat, student_feat):
            # s_feat = layer(s_feat)
            # if s_feat.shape != t_feat.shape:
            #     s_feat = F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            
            # assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.distill_fn(s_feat, t_feat.detach()))


        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight

        losses = dict(loss_feat=feat_loss)

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_CleanDense_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]], neck=None,
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
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

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, img)

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
class KDDDPM_Pretrain_CleanDense_StrongWeak_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]], neck=None,
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def build_model(self, backbone, neck, head):
        self.student = nn.ModuleDict(
            {
                'backbone': build_backbone(backbone),
                'neck': build_neck(neck)
            }
        )

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
        # timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)
        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布
        timesteps = m.sample()           # 从分布中采样

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, img)

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

        reward = -(feat_loss.mean(-1).detach())
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_policy=loss_policy,
                loss_cls=loss_cls,
                reward = reward,
                timesteps= timesteps.float(),
                loss_feat=feat_loss)

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_CleanDense_HRNet_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]], neck=None,
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
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

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        student_feat = self.extract_feat(self.student, img)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        # student_feat = [hook.feat for hook in self.student_hoods.hook_list]
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
class KDDDPM_Pretrain_CleanDenseContrast_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]], neck=None,
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

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
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, img)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat, layer in zip(teacher_feat, student_feat, self.distill_layer):
            s_feat = layer(s_feat)
            if s_feat.shape != t_feat.shape:
                s_feat = F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            
            s_feat = s_feat.view(s_feat.size(0)*s_feat.size(2)*s_feat.size(3), -1)
            t_feat = t_feat.view(t_feat.size(0)*t_feat.size(2)*t_feat.size(3), -1)
            assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.info_nce(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_feat=feat_loss)

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Clean_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]], neck=None,
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
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

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, img)

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
class KDDDPM_Pretrain_Clean_TaskOriented_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=256,
                 num_class=10,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.num_class = num_class
        self.task_head = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim),
            nn.ReLU(True),
            nn.Linear(teacher_dim, teacher_dim),
            nn.ReLU(True),
            nn.Linear(teacher_dim, num_class)
        )
        self.policy_net = PolicyNet_Cov()
        self.kd_weight = train_cfg['kd_weight']
        if 'entropy_reg' in train_cfg.keys():
            self.entropy_reg = train_cfg['entropy_reg']
        else:
            self.entropy_reg = 0.1
        # self.policy_net2 = PolicyNet_Cov()

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

        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布
        loss_entropy = -m.entropy() * self.entropy_reg
        timesteps = m.sample()           # 从分布中采样
        _ = self.extract_feat(self.student, img)

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
            feat_loss.append(F.mse_loss(s_feat, t_feat.detach()))

        teacher_logit = self.task_head(t_feat)
        loss_teacher = F.cross_entropy(
            teacher_logit, gt_label, reduction='none')

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight

        # Find the time with smallest classification loss
        reward = -(loss_teacher.detach())
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        losses = dict(loss_feat=feat_loss,
                      loss_teacher=loss_teacher,
                      loss_policy=loss_policy,
                      loss_entropy=loss_entropy,
                      timesteps=timesteps.float())

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Imagenet_TaskOriented_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=1024,
                 num_class=1000,
                 teacher_ckp=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        self.teacher_ckp = teacher_ckp
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.num_class = num_class
        self.task_head = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim//4),
            nn.ReLU(True),
            nn.Linear(teacher_dim//4, teacher_dim//4),
            nn.ReLU(True),
            nn.Linear(teacher_dim//4, num_class)
        )
        
        self.policy_net = Policy_AlexNet()
        self.kd_weight = train_cfg['kd_weight']
        if 'entropy_reg' in train_cfg.keys():
            self.entropy_reg = train_cfg['entropy_reg']
        else:
            self.entropy_reg = 0.1
        # self.policy_net2 = PolicyNet_Cov()
    def init_ddpm_teacher(self,model_id):
        self.ddpm = create_imagenet256(self.teacher_ckp)
        for param in self.ddpm.parameters():
            param.requires_grad = False

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

        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布
        loss_entropy = -m.entropy() * self.entropy_reg
        timesteps = m.sample()           # 从分布中采样
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps)

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
            feat_loss.append(F.mse_loss(s_feat, t_feat.detach()))

        teacher_logit = self.task_head(t_feat)
        loss_teacher = F.cross_entropy(
            teacher_logit, gt_label, reduction='none')

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight

        # Find the time with smallest classification loss
        reward = -(loss_teacher.detach())
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        losses = dict(loss_feat=feat_loss,
                      loss_teacher=loss_teacher,
                      loss_policy=loss_policy,
                      loss_entropy=loss_entropy,
                      timesteps=timesteps.float())

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Imagenet64_TaskOriented_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=1024,
                 num_class=1000,
                 teacher_ckp=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        self.teacher_ckp = teacher_ckp
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.num_class = num_class
        self.task_head = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim//2),
            nn.ReLU(True),
            nn.Linear(teacher_dim//2, teacher_dim//2),
            nn.ReLU(True),
            nn.Linear(teacher_dim//2, num_class)
        )
        
        self.policy_net = PolicyNet_Cov64()
        self.kd_weight = train_cfg['kd_weight']
        if 'entropy_reg' in train_cfg.keys():
            self.entropy_reg = train_cfg['entropy_reg']
        else:
            self.entropy_reg = 0.1
        # self.policy_net2 = PolicyNet_Cov()
    def init_ddpm_teacher(self,model_id):
        self.ddpm = create_imagenet64(self.teacher_ckp)
        for param in self.ddpm.parameters():
            param.requires_grad = False

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

        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布
        loss_entropy = -m.entropy() * self.entropy_reg
        timesteps = m.sample()           # 从分布中采样
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, gt_label)

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
            feat_loss.append(F.mse_loss(s_feat, t_feat.detach()))

        teacher_logit = self.task_head(t_feat)
        loss_teacher = F.cross_entropy(
            teacher_logit, gt_label, reduction='none')

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight

        # Find the time with smallest classification loss
        reward = -(loss_teacher.detach())
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        losses = dict(loss_feat=feat_loss,
                      loss_teacher=loss_teacher,
                      loss_policy=loss_policy,
                      loss_entropy=loss_entropy,
                      timesteps=timesteps.float())

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Imagenet256_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=50,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=1024,
                 num_class=1000,
                 teacher_ckp=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        self.teacher_ckp = teacher_ckp
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.kd_weight = train_cfg['kd_weight']
        # self.policy_net2 = PolicyNet_Cov()
    def init_ddpm_teacher(self,model_id):
        self.ddpm = create_imagenet256(self.teacher_ckp)
        for param in self.ddpm.parameters():
            param.requires_grad = False

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
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps)

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
            feat_loss.append(F.mse_loss(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight


        losses = dict(loss_feat=feat_loss,
                      timesteps=timesteps.float())

        return losses
@CLASSIFIERS.register_module()
class KDDDPM_AT_Imagenet64_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=50,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=1024,
                 num_class=1000,
                 teacher_ckp=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        self.teacher_ckp = teacher_ckp
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.kd_weight = train_cfg['kd_weight']
        # self.policy_net2 = PolicyNet_Cov()
    def init_ddpm_teacher(self,model_id):
        self.ddpm = create_imagenet64(self.teacher_ckp)
        for param in self.ddpm.parameters():
            param.requires_grad = False

    def init_distill_func(self, distill_fn):
        self.distill_fn = AT(p=2)
    def init_distill_layers(self, teacher_layers, student_layers):
        assert len(student_layers) == len(
            teacher_layers), "len(student_layers) must be equal to len(teacher_layers)"
        # distill_layer = []
        self.teacher_hoods = ForwardHookManager()
        self.student_hoods = ForwardHookManager()
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_layer_name, _ = t_layer
            s_layer_name, _ = s_layer
            self.student_hoods.add_hook(self.student, s_layer_name)
            self.teacher_hoods.add_hook(self.ddpm, t_layer_name)

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
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, gt_label)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat in zip(teacher_feat, student_feat):
            feat_loss.append(self.distill_fn(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight


        losses = dict(loss_feat=feat_loss,
                      timesteps=timesteps.float())

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_AT_Imagenet256_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=50,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=1024,
                 num_class=1000,
                 teacher_ckp=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        self.teacher_ckp = teacher_ckp
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.kd_weight = train_cfg['kd_weight']
        # self.policy_net2 = PolicyNet_Cov()
    def init_ddpm_teacher(self,model_id):
        self.ddpm = create_imagenet256(self.teacher_ckp)
        for param in self.ddpm.parameters():
            param.requires_grad = False

    def init_distill_func(self, distill_fn):
        self.distill_fn = AT(p=2)
    def init_distill_layers(self, teacher_layers, student_layers):
        assert len(student_layers) == len(
            teacher_layers), "len(student_layers) must be equal to len(teacher_layers)"
        # distill_layer = []
        self.teacher_hoods = ForwardHookManager()
        self.student_hoods = ForwardHookManager()
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_layer_name, _ = t_layer
            s_layer_name, _ = s_layer
            self.student_hoods.add_hook(self.student, s_layer_name)
            self.teacher_hoods.add_hook(self.ddpm, t_layer_name)

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
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat in zip(teacher_feat, student_feat):
            feat_loss.append(self.distill_fn(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight

        losses = dict(loss_feat=feat_loss,
                      timesteps=timesteps.float())

        return losses
    

@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Imagenet64_PolicyInfo_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=50,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=1024,
                 num_class=1000,
                 teacher_ckp=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        self.teacher_ckp = teacher_ckp
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.kd_weight = train_cfg['kd_weight']
        self.policy_net = Policy_AlexNet()
    def init_ddpm_teacher(self,model_id):
        self.ddpm = create_imagenet64(self.teacher_ckp)
        for param in self.ddpm.parameters():
            param.requires_grad = False

    def info_nce(self, query, positive_key, temperature=0.2, reduction='none'):
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

        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布

        timesteps = m.sample()           # 从分布中采样

        # timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, gt_label)

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
            feat_loss.append(F.mse_loss(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight
        
        reward = - (self.info_nce(t_feat.detach(), t_feat.detach()) ) # + feat_loss
        # reward_norm = reward - reward.mean()
        loss_policy = -(m.log_prob(timesteps) * reward)

        losses = dict(loss_feat=feat_loss,
                      loss_policy=loss_policy,
                      timesteps=timesteps.float())

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Imagenet64_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=50,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=1024,
                 num_class=1000,
                 teacher_ckp=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        self.teacher_ckp = teacher_ckp
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.kd_weight = train_cfg['kd_weight']
        # self.policy_net2 = PolicyNet_Cov()
    def init_ddpm_teacher(self,model_id):
        self.ddpm = create_imagenet64(self.teacher_ckp)
        for param in self.ddpm.parameters():
            param.requires_grad = False

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
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, gt_label)

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
            feat_loss.append(F.mse_loss(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight


        losses = dict(loss_feat=feat_loss,
                      timesteps=timesteps.float())

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Imagenet64_ImageClassifier_Minus(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=50,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=1024,
                 num_class=1000,
                 teacher_ckp=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        self.teacher_ckp = teacher_ckp
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.kd_weight = train_cfg['kd_weight']
        # self.policy_net2 = PolicyNet_Cov()
    def init_ddpm_teacher(self,model_id):
        self.ddpm = create_imagenet64(self.teacher_ckp)
        for param in self.ddpm.parameters():
            param.requires_grad = False

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
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, gt_label)

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
            feat_loss.append(-F.mse_loss(F.normalize(s_feat), F.normalize(t_feat.detach())))

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight


        losses = dict(loss_feat=feat_loss,
                      timesteps=timesteps.float())

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_RKD_Imagenet64_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=50,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=1024,
                 num_class=1000,
                 teacher_ckp=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        self.teacher_ckp = teacher_ckp
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.kd_weight = train_cfg['kd_weight']
        # self.policy_net2 = PolicyNet_Cov()
    def init_ddpm_teacher(self,model_id):
        self.ddpm = create_imagenet64(self.teacher_ckp)
        for param in self.ddpm.parameters():
            param.requires_grad = False

    def init_distill_func(self, distill_fn):
        self.distill_fn = RKD()

    def init_distill_layers(self, teacher_layers, student_layers):
        assert len(student_layers) == len(
            teacher_layers), "len(student_layers) must be equal to len(teacher_layers)"
        # distill_layer = []
        self.teacher_hoods = ForwardHookManager()
        self.student_hoods = ForwardHookManager()
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_layer_name, _ = t_layer
            s_layer_name, _ = s_layer
            self.student_hoods.add_hook(self.student, s_layer_name)
            self.teacher_hoods.add_hook(self.ddpm, t_layer_name)
        # pass

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
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, gt_label)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat in zip(teacher_feat, student_feat):
            # s_feat = layer(s_feat)
            # if s_feat.shape != t_feat.shape:
            s_feat = F.adaptive_avg_pool2d(s_feat, (1, 1)).view(bs, -1)
            t_feat = F.adaptive_avg_pool2d(t_feat, (1, 1)).view(bs, -1)

            # F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            # assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.distill_fn(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight


        losses = dict(loss_feat=feat_loss,
                      timesteps=timesteps.float())

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Dense_Imagenet64_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=50,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=1024,
                 num_class=1000,
                 teacher_ckp=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        self.teacher_ckp = teacher_ckp
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.kd_weight = train_cfg['kd_weight']
        # self.policy_net2 = PolicyNet_Cov()
    def init_ddpm_teacher(self,model_id):
        self.ddpm = create_imagenet64(self.teacher_ckp)
        for param in self.ddpm.parameters():
            param.requires_grad = False

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
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, gt_label)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat, layer in zip(teacher_feat, student_feat, self.distill_layer):
            s_feat = layer(s_feat)
            # if s_feat.shape != t_feat.shape:
            # s_feat = F.adaptive_avg_pool2d(s_feat, (1, 1)).view(bs, -1)
            # t_feat = F.adaptive_avg_pool2d(t_feat, (1, 1)).view(bs, -1)

            s_feat = F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(F.mse_loss(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight


        losses = dict(loss_feat=feat_loss,
                      timesteps=timesteps.float())

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_AT_TaskOriented_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=256,
                 num_class=10,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.num_class = num_class
        self.task_head = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim),
            nn.ReLU(True),
            nn.Linear(teacher_dim, teacher_dim),
            nn.ReLU(True),
            nn.Linear(teacher_dim, num_class)
        )
        self.policy_net = PolicyNet_Cov()
        self.distill_layer = None
        self.kd_weight = train_cfg['kd_weight']
        if 'entropy_reg' in train_cfg.keys():
            self.entropy_reg = train_cfg['entropy_reg']
        else:
            self.entropy_reg = 1.0
        
        # self.policy_net2 = PolicyNet_Cov()
    def init_distill_func(self, distill_fn):
        self.distill_fn = AT(p=2)

    def init_distill_layers(self, teacher_layers, student_layers):
        assert len(student_layers) == len(
            teacher_layers), "len(student_layers) must be equal to len(teacher_layers)"
        # distill_layer = []
        self.teacher_hoods = ForwardHookManager()
        self.student_hoods = ForwardHookManager()
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_layer_name, _ = t_layer
            s_layer_name, _ = s_layer
            self.student_hoods.add_hook(self.student, s_layer_name)
            self.teacher_hoods.add_hook(self.ddpm, t_layer_name)
        # pass

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

        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布
        loss_entropy = -m.entropy() * self.entropy_reg
        timesteps = m.sample()           # 从分布中采样
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat in zip(teacher_feat, student_feat):
            # s_feat = layer(s_feat)
            # if s_feat.shape != t_feat.shape:

            s_feat = F.interpolate(s_feat, size=(t_feat.size(2), t_feat.size(3)), mode='bilinear')
            # assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.distill_fn(s_feat, t_feat.detach()))

        t_feat = F.adaptive_avg_pool2d(t_feat, (1, 1)).view(bs, -1)
        teacher_logit = self.task_head(t_feat)
        loss_teacher = F.cross_entropy(
            teacher_logit, gt_label, reduction='none')

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight

        # Find the time with smallest classification loss
        reward = -(loss_teacher.detach())
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        losses = dict(loss_feat=feat_loss,
                      loss_teacher=loss_teacher,
                      loss_policy=loss_policy,
                      loss_entropy=loss_entropy,
                      timesteps=timesteps.float())

        return losses
    
@CLASSIFIERS.register_module()
class KDDDPM_RKD_TaskOriented_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=256,
                 num_class=10,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.num_class = num_class
        self.task_head = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim),
            nn.ReLU(True),
            nn.Linear(teacher_dim, teacher_dim),
            nn.ReLU(True),
            nn.Linear(teacher_dim, num_class)
        )
        self.policy_net = PolicyNet_Cov()
        self.distill_layer = None
        self.kd_weight = train_cfg['kd_weight']
        if 'entropy_reg' in train_cfg.keys():
            self.entropy_reg = train_cfg['entropy_reg']
        else:
            self.entropy_reg = 1.0
        
        # self.policy_net2 = PolicyNet_Cov()
    def init_distill_func(self, distill_fn):
        self.distill_fn = RKD()

    def init_distill_layers(self, teacher_layers, student_layers):
        assert len(student_layers) == len(
            teacher_layers), "len(student_layers) must be equal to len(teacher_layers)"
        # distill_layer = []
        self.teacher_hoods = ForwardHookManager()
        self.student_hoods = ForwardHookManager()
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_layer_name, _ = t_layer
            s_layer_name, _ = s_layer
            self.student_hoods.add_hook(self.student, s_layer_name)
            self.teacher_hoods.add_hook(self.ddpm, t_layer_name)
        # pass

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

        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布
        loss_entropy = -m.entropy() * self.entropy_reg
        timesteps = m.sample()           # 从分布中采样
        _ = self.extract_feat(self.student, img)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        for t_feat, s_feat in zip(teacher_feat, student_feat):
            # s_feat = layer(s_feat)
            # if s_feat.shape != t_feat.shape:
            t_feat = F.adaptive_avg_pool2d(t_feat, (1, 1)).view(bs, -1)
            s_feat = F.adaptive_avg_pool2d(s_feat, (1, 1)).view(bs, -1)
            # assert s_feat.shape == t_feat.shape
            # print(s_feat.shape, t_feat.shape)
            feat_loss.append(self.distill_fn(s_feat, t_feat.detach()))

        teacher_logit = self.task_head(t_feat)
        loss_teacher = F.cross_entropy(
            teacher_logit, gt_label, reduction='none')

        feat_loss = sum(feat_loss)/len(feat_loss) * self.kd_weight

        # Find the time with smallest classification loss
        reward = -(loss_teacher.detach())
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        losses = dict(loss_feat=feat_loss,
                      loss_teacher=loss_teacher,
                      loss_policy=loss_policy,
                      loss_entropy=loss_entropy,
                      timesteps=timesteps.float())

        return losses
    


@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Clean_TaskOriented2_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 teacher_dim=256,
                 num_class=10,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.num_class = num_class
        self.policy_net = PolicyNet_Cov()
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

        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布
        loss_entropy = -m.entropy() * 0.1
        timesteps = m.sample()           # 从分布中采样
        _ = self.extract_feat(self.student, img)

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
        one_hot_gt = F.one_hot(gt_label, num_classes=self.num_class).to(device)
        # Find the time with smallest classification loss
        reward = - torch.linalg.lstsq(t_feat, one_hot_gt).residuals
        loss_policy = -(m.log_prob(timesteps) * reward).mean()

        losses = dict(loss_feat=feat_loss,
                      loss_policy=loss_policy,
                      loss_entropy=loss_entropy,
                      timesteps=timesteps.float())

        return losses

@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Clean_StudentOriented_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 student_dim=512,
                 num_class=10,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.task_head = nn.Sequential(
            nn.Linear(student_dim, student_dim),
            nn.ReLU(True),
            nn.Linear(student_dim, student_dim),
            nn.ReLU(True),
            nn.Linear(student_dim, num_class)
        )
        self.policy_net = PolicyNet_Cov()
        self.last_state = None
        self.last_action = None
        self.last_target = None

    def get_reward(self):
       
        probs = self.policy_net(self.last_state)
        m = Categorical(probs)      # 生成分布
        loss_entropy = -m.entropy() * 0.1
        with torch.no_grad():
            feat = self.extract_feat(self.student, self.last_state)
            if isinstance(feat, tuple):
                feat = feat[-1]
            # student_logit = self.task_head(feat)
        reward = - torch.linalg.lstsq(feat, F.one_hot(self.last_target)).residuals
        # reward = - F.cross_entropy(student_logit, self.last_target, reduction='none').detach()
        loss_policy = -(m.log_prob(self.last_action) * reward).mean()

        return loss_policy, loss_entropy

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


        feat = self.extract_feat(self.student, img)

        with torch.no_grad():
            probs = self.policy_net(img)
            m = Categorical(probs)      # 生成分布
            timesteps = m.sample()           # 从分布中采样
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

        # Train the task head
        if isinstance(feat, tuple):
            feat = feat[-1]
        # student_logit = self.task_head(feat.detach())
        # loss_student = F.cross_entropy(student_logit, gt_label, reduction='none')

        if self.last_state is not None:
            loss_policy, loss_entropy = self.get_reward()
            # Find the time with smallest classification loss
            
        else:
            loss_entropy = torch.zeros(1).to(device)
            loss_policy= torch.zeros(1).to(device)
        
        self.last_target = gt_label
        self.last_state = img
        self.last_action = timesteps
        # print(loss_policy)

        losses = dict(loss_feat=feat_loss,
                    #   loss_student=loss_student,
                      loss_policy=loss_policy,
                      loss_entropy=loss_entropy,
                      timesteps=timesteps.float())

        return losses


@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Clean_TaskCommon_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]],
                 entropy_reg=0.1,
                 teacher_dim=256,
                 num_class=10,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, max_time_step,
                         model_id, distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.task_head = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim),
            nn.ReLU(True),
            nn.Linear(teacher_dim, teacher_dim),
            nn.ReLU(True),
            nn.Linear(teacher_dim, num_class)
        )
        self.common_head = nn.Sequential(
            # nn.Linear(teacher_dim, teacher_dim),
            nn.Identity()
        )
        self.entropy_reg = entropy_reg
        self.policy_net_task = PolicyNet_Cov()
        self.policy_net_common = PolicyNet_Cov()

    def info_nce(self, query, positive_key, temperature=0.1, reduction='none'):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')

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

        _ = self.extract_feat(self.student, img)

        # Task-oriented
        probs = self.policy_net_task(img)
        m = Categorical(probs)      # 生成分布
        loss_entropy = -m.entropy() * self.entropy_reg
        timesteps_task = m.sample()      # 从分布中采样

        with torch.no_grad():
            _ = self.ddpm(img, timesteps_task, return_dict=False)

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

        teacher_logit = self.task_head(t_feat.detach())
        loss_teacher = F.cross_entropy(
            teacher_logit, gt_label, reduction='none')

        feat_task_loss = sum(feat_loss)/len(feat_loss)

        # Find the time with smallest classification loss
        reward = -(loss_teacher.detach())
        loss_policy_task = -(m.log_prob(timesteps_task) * reward).mean()

        # Mutual information
        probs = self.policy_net_common(img)
        m = Categorical(probs)      # 生成分布
        loss_entropy += -m.entropy() * self.entropy_reg
        timesteps_common = m.sample()      # 从分布中采样

        with torch.no_grad():
            _ = self.ddpm(img, timesteps_common, return_dict=False)

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

        feat_loss_common = sum(feat_loss)/len(feat_loss)

        t_feat = self.common_head(t_feat.detach())
        loss_infonce = self.info_nce(t_feat, t_feat)
        # Find the time with smallest classification loss
        reward = -(loss_infonce.detach())
        loss_policy_common = -(m.log_prob(timesteps_common) * reward).mean()

        losses = dict(loss_feat_common=feat_loss_common,
                      loss_feat_task=feat_task_loss,
                      loss_teacher=loss_teacher,
                      loss_policy_task=loss_policy_task,
                      loss_infonce = loss_infonce,
                      loss_policy_common = loss_policy_common,
                      loss_entropy=loss_entropy,
                      timesteps_task = timesteps_task.float(),
                      timesteps_common=timesteps_common.float())

        return losses


@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_Clean_SingleT_ImageClassifier(KDDDPM_Pretrain_Clean_ImageClassifier):
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
        timesteps = torch.LongTensor(
            [self.max_time_step]).repeat(bs).to(device)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, img)

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
class KDDDPM_Pretrain_CleanContrast_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=1000,
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]], neck=None,
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.max_time_step = max_time_step
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

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
        timesteps = torch.randint(0, self.max_time_step, (bs,)).to(device)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, img)

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

        losses = dict(loss_feat=feat_loss)

        return losses


@CLASSIFIERS.register_module()
class KDDDPM_Pretrain_MutiStep_ImageClassifier(KDDDPM_ImageClassifier):
    def __init__(self,
                 backbone,
                 teacher_layers,
                 student_layers,
                 max_time_step=[1000],
                 model_id="google/ddpm-cifar10-32",
                 distill_fn=[['l2', 1]], neck=None,
                 head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        assert len(max_time_step) == len(
            student_layers), "len(max_time_step) should == len(student_layers)."
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
        max_time_step_index = random.randint(0, len(self.max_time_step)-1)
        # print(max_time_step_index)
        if max_time_step_index == len(self.max_time_step)-1:
            timesteps = torch.randint(
                0, self.max_time_step[max_time_step_index], (bs,)).to(device)
        else:
            timesteps = torch.randint(
                self.max_time_step[max_time_step_index+1], self.max_time_step[max_time_step_index], (bs,)).to(device)

        with torch.no_grad():
            _ = self.ddpm(img, timesteps, return_dict=False)
        _ = self.extract_feat(self.student, img)

        teacher_feat = [hook.feat for hook in self.teacher_hoods.hook_list]
        student_feat = [hook.feat for hook in self.student_hoods.hook_list]
        feat_loss = []
        t_feat, s_feat, layer = teacher_feat[max_time_step_index], student_feat[
            max_time_step_index], self.distill_layer[max_time_step_index]

        s_feat = layer(s_feat)

        s_feat = F.adaptive_avg_pool2d(s_feat, (1, 1)).view(bs, -1)
        t_feat = F.adaptive_avg_pool2d(t_feat, (1, 1)).view(bs, -1)

        assert s_feat.shape == t_feat.shape
        feat_loss = self.distill_fn(s_feat, t_feat.detach())

        losses = dict(loss_feat=feat_loss)

        return losses


@CLASSIFIERS.register_module()
class KDDDPM_PolicyNoise(KDDDPM_ImageClassifier):
    def __init__(self, backbone, teacher_layers, student_layers, model_id="google/ddpm-cifar10-32", distill_fn=[['l2', 1]], neck=None, head=None, pretrained=None, train_cfg=None, init_cfg=None):
        super().__init__(backbone, teacher_layers, student_layers, model_id,
                         distill_fn, neck, head, pretrained, train_cfg, init_cfg)
        self.policy_net = PolicyNet()
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

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

        bs = img.shape[0]
        device = img.device
        probs = self.policy_net(img)
        m = Categorical(probs)      # 生成分布

        timesteps = m.sample()           # 从分布中采样

        noise = torch.randn(img.shape).to(device)
        noise_img = self.noise_scheduler.add_noise(
            img, noise, timesteps).to(device)

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
            feat_loss.append(-self.distill_fn(s_feat, t_feat.detach()))

        feat_loss = sum(feat_loss)/len(feat_loss)

        losses = dict(loss_cls=loss_cls,
                      loss_feat=feat_loss)

        return losses


class PolicyNet(nn.Module):
    def __init__(self, input_dim=3072, action_num=1000):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_num)  # Prob of Left

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class PolicyNet_Cov64(nn.Module):
    def __init__(self, action_num=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_num)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class PolicyNet_Cov(nn.Module):
    def __init__(self, action_num=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_num)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x
    
class Policy_AlexNet(nn.Module):
    def __init__(self, action_num=1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_num),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.softmax(x, dim=-1)
        return x
