_base_ = [
    '../_base_/datasets/tinyimagenet_bs128.py',
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

optimizer = dict(type='SGD',
                 lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)

fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='KDDDPM_AT_Imagenet64_ImageClassifier',
    teacher_layers=[["middle_block.2.out_layers.3", 768]],
    student_layers=[['backbone.conv2.activate', 1280]],
    distill_fn=[['l2', 1.0]],
    train_cfg=dict(kd_weight=1000.0),
    max_time_step=50,
    teacher_ckp="guided-diffusion/64x64_diffusion.pt",
    backbone=dict(type='MobileNetV2_CIFAR',
                     out_indices=(7, ),
                     widen_factor=1.0),
    neck=dict(
        type='GlobalAveragePooling'
    ),
    head=None
)

evaluation = dict(interval=200, metric='accuracy')
