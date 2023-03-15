_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/celebahqmask.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
data = dict(samples_per_gpu=8)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(num_classes=18),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=18,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=18,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ])
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.005)
