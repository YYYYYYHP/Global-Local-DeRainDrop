import os.path

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import cv2
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='./data/test_b/predict/', type=str)
    parser.add_argument("--img_name", default='0_predict.jpg', type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--output_dir", default='./data/test_b/sam/', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    args = get_args()
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=f"cuda:{args.gpu}" if args.gpu >=0 else 'cpu')

    input_list = sorted(os.listdir(args.input_dir))
    for i in input_list:
        args.img_name = i
        global_derained_img = cv2.cvtColor(cv2.imread(os.path.join(args.input_dir, args.img_name)), cv2.COLOR_BGR2RGB)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(global_derained_img)

        bbox_list = []
        local_img_list = []
        for idx in range(0, len(masks)):
            mask = masks[idx]['bbox']
            mask[2] = mask[0] + mask[2]
            mask[3] = mask[1] + mask[3]
            if ((mask[2] - mask[0]) <= 64 and (mask[3] - mask[1]) <= 64):
                bbox_list.append(mask)
                local_img_list.append(global_derained_img[mask[1]:mask[3], mask[0]:mask[2]])
            else:
                continue
        with open(os.path.join(args.output_dir, args.img_name.split('.')[:-1][0] + '.pkl'), 'wb') as f:
            pickle.dump({'bbox': np.array(bbox_list), 'imgs': local_img_list}, f)
