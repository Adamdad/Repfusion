_base_ = [
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py'
]


data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    val=dict(ann_file=None),
    test=dict(ann_file=None))

# checkpoint saving
checkpoint_config = dict(interval=10)
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

optimizer = dict(
    type='AdamW',
    lr=5e-4 * 32 * 8 / 512)

# optimizer
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=5 * 1252,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=100)

fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='KDDDPM_AT_Imagenet256_ImageClassifier',
    teacher_layers=[["middle_block.2.out_layers.3", 1024]],
    student_layers=[['backbone.layer4.1.relu', 512]],
    distill_fn=[['l2', 1000.0]],
    teacher_dim=1024,
    num_class=1000,
    max_time_step=1,
    teacher_ckp="/home/yangxingyi/guided-diffusion/256x256_diffusion_uncond.pt",
    train_cfg=dict(kd_weight=1000.),
    backbone=dict(type='ResNet',
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
