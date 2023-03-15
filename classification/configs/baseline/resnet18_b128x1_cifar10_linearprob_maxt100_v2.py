_base_ = [
    '../_base_/datasets/cifar10_bs128.py'
]

# checkpoint saving
checkpoint_config = dict(interval=100)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

optimizer_config = dict(grad_clip=None)
# learning policy
lr = 10.0
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0)
lr_cfg = dict(  # passed to adjust_learning_rate()
    type='MultiStep',
    lr=lr,
    decay_rate=0.1,
    decay_steps=[60, 80],
)
runner = dict(type='EpochBasedRunner', max_epochs=100)


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=4, 
        norm_eval=True,
        init_cfg=dict(type='Pretrained', 
                      prefix='student.backbone.',
                      checkpoint='work_dirs/ddpm_resnet18_b128x1_cifar10_pretrain_maxtime/latest.pth')),
    neck=dict(
        type='GlobalAveragePooling'
    ),
    head=dict(
        type='LinearClsHead',
            num_classes=10,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
    )
)
