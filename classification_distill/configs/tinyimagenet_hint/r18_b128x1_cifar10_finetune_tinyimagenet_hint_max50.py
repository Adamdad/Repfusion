_base_ = [
    '../_base_/datasets/tinyimagenet_bs128.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py'
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
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 64 * 1 / 512)

# optimizer
# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     min_lr_ratio=1e-2)
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=100,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=200)


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
            type='ResNet_CIFAR',
            depth=18,
            num_stages=4,
            out_indices=(3,),
        style='pytorch',
        init_cfg=dict(type='Pretrained', 
                      prefix='student.backbone.',
                      checkpoint='work_dirs/ddpm-r18_hint_tinyimagenet_max50/latest.pth')),
    neck=dict(
        type='GlobalAveragePooling'
    ),
    head=dict(
        type='LinearClsHead',
            num_classes=200,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
    )
)
