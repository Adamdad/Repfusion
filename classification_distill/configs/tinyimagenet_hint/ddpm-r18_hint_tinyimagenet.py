_base_ = [
    '../_base_/datasets/tinyimagenet_bs128.py'
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4)
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
                 lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)

fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='Repfusion_Imagenet64_ImageClassifier',
    teacher_layers=[["middle_block.2.out_layers.3", 768]],
    student_layers=[['backbone.layer4.1.relu', 512]],
    distill_fn=[['l2', 1.0]],
    teacher_dim=768,
    num_class=200,
    teacher_ckp="guided-diffusion/64x64_diffusion.pt",
    train_cfg=dict(kd_weight=1.,
                entropy_reg=0.1),
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


custom_hooks = [
    dict(type='EntropyDecayHook', init_entropy_reg=0.00, end_epoch=100),
    dict(type='EMAHook', momentum=0.001, priority='ABOVE_NORMAL')
]

evaluation = dict(interval=200, metric='accuracy')
