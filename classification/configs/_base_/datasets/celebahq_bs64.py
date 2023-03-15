# dataset settings
dataset_type = 'ImageList'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=256),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/celebaHQ_mask/CelebAMask-HQ/CelebA-HQ-img',
        ann_file=None,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/celebaHQ_mask/CelebAMask-HQ/CelebA-HQ-img',
        ann_file=None,
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/celebaHQ_mask/CelebAMask-HQ/CelebA-HQ-img',
        ann_file=None,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')