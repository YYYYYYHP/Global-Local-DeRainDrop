# PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
# Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
# Models lib
from models import *
# Metrics lib
import cv2
from skimage.measure import compare_psnr, compare_ssim


def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)


def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='test', type=str)
    parser.add_argument("--input_dir", default='./data/test_b/data/', type=str)
    parser.add_argument("--sam", default='./data/test_b/predict/', type=str)
    parser.add_argument("--gt_dir", default='./data/test_b/gt/', type=str)
    args = parser.parse_args()
    return args


def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    # align to four
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


def predict(image):
    image = np.array(image, dtype='float32') / 255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()

    out = model(image)[-1]

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :] * 255.

    return out


if __name__ == '__main__':
    args = get_args()

    model = Generator().cuda()
    model.load_state_dict(torch.load('./checkpoints/global_model.pkl'))

    input_list = sorted(os.listdir(args.input_dir))
    gt_list = sorted(os.listdir(args.gt_dir))
    num = len(input_list)
    cumulative_psnr = 0
    cumulative_ssim = 0
    for i in range(num):
        print('Processing image: %s' % (input_list[i]))
        img = cv2.imread(args.input_dir + input_list[i])
        gt = cv2.imread(args.gt_dir + gt_list[i])
        img = align_to_four(img)
        gt = align_to_four(gt)
        result = predict(img)
        result = np.array(result, dtype='uint8')
        cur_psnr = calc_psnr(result, gt)
        cur_ssim = calc_ssim(result, gt)
        print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
        # cv2.imwrite(args.output_dir + input_list[i].replace('rain', 'predict'), result)

    print('In testing dataset, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / num, cumulative_ssim / num))
