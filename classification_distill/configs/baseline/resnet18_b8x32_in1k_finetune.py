_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

data = dict(
    val=dict(ann_file=None),
    test=dict(ann_file=None))

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', 
                prefix='student.backbone.',
                checkpoint=''
                )
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            num_classes=1000),
    ))

custom_hooks = [dict(type='EMAHook', momentum=0.01, priority='ABOVE_NORMAL')]