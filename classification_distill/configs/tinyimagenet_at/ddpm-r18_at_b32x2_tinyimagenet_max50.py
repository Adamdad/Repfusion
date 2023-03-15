_base_ = [
    '../_base_/datasets/tinyimagenet_bs128.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py'
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2)
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
    warmup_iters=1000,
    warmup_by_epoch=False)

# learning policy
# lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)

fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='KDDDPM_Pretrain_Imagenet64_ImageClassifier',
    teacher_layers=[["middle_block.2.out_layers.3", 768]],
    student_layers=[['backbone.layer4.1.relu', 512]],
    distill_fn=[['l2', 1.0]],
    train_cfg=dict(kd_weight=1.0),
    max_time_step=50,
    teacher_ckp="/home/yangxingyi/guided-diffusion/64x64_diffusion.pt",
    backbone=dict(
            type='ResNet_CIFAR',
            depth=18,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
    neck=dict(
        type='GlobalAveragePooling'
    ),
    head=None
)

evaluation = dict(interval=200, metric='accuracy')
