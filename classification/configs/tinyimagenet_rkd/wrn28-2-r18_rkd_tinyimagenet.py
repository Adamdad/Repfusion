_base_ = [
    '../_base_/datasets/tinyimagenet_bs128.py'
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    val=dict(
        data_prefix='data/tiny-imagenet-200/val/'),
    test=dict(
        data_prefix='data/tiny-imagenet-200/val/'))

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
optimizer = dict(type='SGD',
                 lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)


# model settings
model = dict(
    type='RKDImageClassifier',
    kd_loss=dict(type='Logits'), # not used
    train_cfg=dict(lambda_feat=1.0,
                   lambda_kd=1.0,
                   feat_channels=dict(student=[512],
                                      teacher=[128]),
                   teacher_checkpoint='work_dirs/wideresnet28-2_b64x2_tinyimagenet/latest.pth'),
    backbone=dict(
        # return_tuple=False,
        student=dict(
            type='ResNet_CIFAR',
            depth=18,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        teacher=dict(
            type='WideResNet_CIFAR',
            depth=28,
            stem_channels=16,
            base_channels=16 * 2,
            num_stages=3,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            out_indices=(2, ),
            out_channel=128,
            style='pytorch'),
    ),
    neck=dict(
        student=dict(type='GlobalAveragePooling'),
        teacher=dict(type='GlobalAveragePooling')
    ),
    head=dict(
        teacher=dict(
            type='LinearClsHead',
            num_classes=200,
            in_channels=128,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ),
        student=dict(
            type='LinearClsHead',
            num_classes=200,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        )
    )
)