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
# optimizer
optimizer = dict(type='SGD',
                 lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)

# learning policy
# lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)

fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='KDDDPM_Pretrain_Imagenet64_PolicyInfo_ImageClassifier',
    teacher_layers=[["middle_block.2.out_layers.3", 768]],
    student_layers=[['backbone.relu', 128]],
    distill_fn=[['l2', 1.0]],
    train_cfg=dict(kd_weight=1.0),
    teacher_ckp="/root/autodl-tmp/64x64_diffusion.pt",
    backbone=dict(
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
    neck=dict(
        type='GlobalAveragePooling'
    ),
    head=None
)

evaluation = dict(interval=200, metric='accuracy')
