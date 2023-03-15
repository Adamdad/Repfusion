# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CelebAHQMask(CustomDataset):
    """Face Occluded dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('skin', 'l_brow', 'r_brow', 
            'l_eye', 'r_eye', 'eye_g', 
            'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 
            'l_lip', 'neck', 'neck_l', 
            'cloth', 'hair', 'hat')

    PALETTE = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
               [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
               [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
               [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
               [0, 32, 192], [128, 128, 224]]

    def __init__(self, split, **kwargs):
        super(CelebAHQMask, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
    