

_base_ = './bisenetv1_r18-d32_lr5e-3_1x16_448x448_160k_coco-celebahq_mask.py'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', 
                checkpoint='' , # Put the disilled checkpoint hear
                prefix='student.backbone.')
            )
        ), 
    )
