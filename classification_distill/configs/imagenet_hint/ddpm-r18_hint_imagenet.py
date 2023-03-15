_base_ = [
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py'
]


data = dict(
    samples_per_gpu=32,
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

optimizer = dict(type='SGD',
                 lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)

fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='KDDDPM_Pretrain_Imagenet256_ImageClassifier',
    teacher_layers=[["middle_block.2.out_layers.3", 1024]],
    student_layers=[['backbone.layer4.1.relu', 512]],
    distill_fn=[['l2', 1.0]],
    teacher_dim=1024,
    num_class=1000,
    teacher_ckp="/home/yangxingyi/guided-diffusion/256x256_diffusion_uncond.pt",
    train_cfg=dict(kd_weight=1.),
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
