# dataset settings
dataset_type = 'TinyImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=64),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(64, -1)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/tiny-imagenet-200/train',
        pipeline=train_pipeline),
    val=dict(
        type='ImageNet',
        data_prefix='data/tiny-imagenet-200/val/images',
        ann_file=None,
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type='ImageNet',
        data_prefix='data/tiny-imagenet-200/val/images',
        ann_file=None,
        pipeline=test_pipeline))
evaluation = dict(interval=5, metric='accuracy')