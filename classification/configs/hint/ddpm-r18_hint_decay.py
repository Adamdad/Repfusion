_base_ = [
    '../_base_/datasets/cifar10_bs128.py'
]

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
    type='KDDDPM_Pretrain_Clean_TaskOriented_ImageClassifier',
    teacher_layers=[["mid_block.resnets.1.conv2", 256]],
    student_layers=[['backbone.layer4.1.relu', 512]],
    distill_fn=[['l2', 1.0]],
    train_cfg=dict(kd_weight=1.,
                entropy_reg=0.1),
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(
        type='GlobalAveragePooling'
    ),
    head=None
)

custom_hooks = [
    dict(type='EntropyDecayHook', init_entropy_reg=0.1, end_epoch=100),
]

evaluation = dict(interval=200, metric='accuracy')