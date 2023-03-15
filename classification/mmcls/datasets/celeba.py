import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class CelebA(MultiLabelDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.
    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        difficult_as_postive (Optional[bool]): Whether to map the difficult
            labels as positive. If it set to True, map difficult examples to
            positive ones(1), If it set to False, map difficult examples to
            negative ones(0). Defaults to None, the difficult labels will be
            set to '-1'.
    """

    CLASSES = ('5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
               'Bald Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
               'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
               'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
               'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
               'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
               'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
               'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
               'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 
               'Wearing_Necktie', 'Young')

    def __init__(self, split_fille, split='train', **kwargs):
        super(CelebA, self).__init__(**kwargs)
        assert split in ['train', 'val', 'test']
        
        self.split = split
        self.split_value = dict(train=0, val=1, test=2)[split]
        self.split_fille = split_fille

    def load_annotations(self):
        """Load annotations.
        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        split_list = open(self.ann_file).readlines()
        split_dict = dict()
        for line in split_list:
            filename, value = line.split()
            split_dict[filename] = int(value)
        label_list = open(self.ann_file).readlines()[2:]
        for label in label_list:
            label = label.split()
            filename = label[0]
            if split_dict[filename] == self.split_value:
                gt_label = np.zeros(len(self.CLASSES))
                for i, n in enumerate(label[1:]):
                    if n == '-1':
                        gt_label[i] = 0
                    else:
                        gt_label[i] = 1
                
                info = dict(
                    img_prefix=self.data_prefix,
                    img_info=dict(filename=filename),
                    gt_label=gt_label.astype(np.int8))
                data_infos.append(info)
        print(f'CelebA dataset {self.split} subset, of {len(data_infos)} images.')

        return data_infos
