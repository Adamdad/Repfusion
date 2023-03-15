_base_ = [
    '../_base_/datasets/celebahq_bs32.py'
]

data = dict(samples_per_gpu=8,
    workers_per_gpu=2)

# checkpoint saving
checkpoint_config = dict(interval=50,max_keep_ckpts=3)
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

# fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='KDDDPM_Pretrain_CleanDense_ImageClassifier',
    teacher_layers=[["mid_block.resnets.1.conv2", 512]],
    student_layers=[['backbone.layer4.2.relu', 2048]],
    distill_fn=[['l1', 10.0]],
    model_id='google/ddpm-ema-celebahq-256',
    max_time_step=1,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(
        type='GlobalAveragePooling'
    ),
    head=None
)

evaluation = dict(interval=200, metric='accuracy')