_base_ = [
    '../_base_/datasets/cifar10_bs128.py'
]

# checkpoint saving
checkpoint_config = dict(interval=50)
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

# optimizer
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 128 * 1 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr_ratio=1e-2)

runner = dict(type='EpochBasedRunner', max_epochs=200)


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(type='Pretrained', 
                      prefix='student.backbone.',
                      checkpoint='work_dirs/ddpm_resnet18_b128x1_cifar10_deepsup_pretrain_adamw/latest.pth')),
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
