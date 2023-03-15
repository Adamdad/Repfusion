import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2
import random
from tqdm import tqdm

def split_train_val(root, face_data):
    train_num = 24183
    val_num = 2993
    test_num = 2824
    file_names = [f.split('.')[0] for f in os.listdir(
        face_data)]  # if f.endswith('.png')
    print("CelebA-HQ-Mask contains {} images in total".format(len(file_names)))
    assert train_num + val_num + test_num == len(file_names)
    random.shuffle(file_names)

    train_names = file_names[:train_num]
    val_names = file_names[train_num:train_num+val_num]
    test_names = file_names[train_num+val_num:]

    list2txt(train_names, os.path.join(root, 'train_celebahqmask.txt'))
    list2txt(val_names, os.path.join(root, 'val_celebahqmask.txt'))
    list2txt(test_names, os.path.join(root, 'test_celebahqmask.txt'))


def list2txt(list_values, name):
    print("Save {} files to {}".format(len(list_values), name))
    # Open a file in write mode
    with open(name, "w") as file:
        # Write each item in the list to a new line in the file
        file.write("\n".join(list_values))


def raw_to_labels(face_data, face_sep_mask, mask_path):
    counter = 0
    total = 0
    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    for i in range(15):
        # files = os.listdir(osp.join(face_sep_mask, str(i)))
        print("Preprocess folder {}".format(i))
        pbar = tqdm(range(i*2000, (i+1)*2000))
        for j in pbar:
            
            mask = np.zeros((512, 512))

            for l, att in enumerate(atts, 1):
                total += 1
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(face_sep_mask, str(i), file_name)

                if os.path.exists(path):
                    counter += 1
                    sep_mask = np.array(Image.open(path).convert('P'))

                    mask[sep_mask == 225] = l
            cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
            # print(j)
            pbar.set_description(f"File {j}")

    print(counter, total)


if __name__ == "__main__":
    root = '/Checkpoint/yangxingyi/data/celebaHQ_mask/CelebAMask-HQ'
    face_data = '/Checkpoint/yangxingyi/data/celebaHQ_mask/CelebAMask-HQ/CelebA-HQ-img'
    face_sep_mask = '/Checkpoint/yangxingyi/data/celebaHQ_mask/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    mask_path = '/Checkpoint/yangxingyi/data/celebaHQ_mask/CelebAMask-HQ/mask'
    # split_train_val(root, face_data)
    raw_to_labels(face_data, face_sep_mask, mask_path)

