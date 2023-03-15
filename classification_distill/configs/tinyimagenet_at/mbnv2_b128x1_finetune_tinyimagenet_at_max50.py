_base_ = [
    '../_base_/datasets/tinyimagenet_bs128.py'
]

# data = dict(
#     samples_per_gpu=64,
#     workers_per_gpu=4)
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
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2_CIFAR',
                     out_indices=(7, ),
                     widen_factor=1.0,
                    init_cfg=dict(type='Pretrained', 
                                prefix='student.backbone.',
                                checkpoint='work_dirs/ddpm-mbnv2_at_b64x1_tinyimagenet_max50/latest.pth')),
    neck=dict(
        type='GlobalAveragePooling'
    ),
    head=dict(
        type='LinearClsHead',
            num_classes=200,
            in_channels=1280,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
    )
)
