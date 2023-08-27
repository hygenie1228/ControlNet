import glob
import numpy as np
import torch
import os
import os.path as osp
import argparse
from tqdm import tqdm
import cv2

import lpips
from metric import psnr, ssim

parser = argparse.ArgumentParser()
parser.add_argument('--exp-dir', default='experiment/exp_08-27_01:30:18', required=False, type=str)
args = parser.parse_args()

# Linearly calibrated models (LPIPS)
lpips_fn = lpips.LPIPS(net='alex', spatial=False) 
# lpips_fn = lpips.LPIPS(net='vgg', spatial=False)
img_list = sorted(glob.glob('experiment/evaluation/target/*'))

total_psnr = 0.0
total_ssim = 0.0
total_lpips = 0.0
total_num = len(img_list)
for target_path in tqdm(img_list):
    imgname = target_path.split('/')[-1]
    output_image =  lpips.load_image(osp.join(args.exp_dir, imgname))
    target_image = lpips.load_image(target_path)

    image_resolution = output_image.shape[:2][::-1]
    target_image = cv2.resize(target_image, image_resolution)

    total_psnr += psnr(output_image, target_image)
    total_ssim += ssim(output_image, target_image)
    
    output_image = lpips.im2tensor(output_image)
    target_image = lpips.im2tensor(target_image)
    total_lpips += lpips_fn.forward(target_image, output_image).item()

print(f"PSNR:{total_psnr/total_num:0.4f}")
print(f"SSIM:{total_ssim/total_num:0.4f}")
print(f"LPIPS:{total_lpips/total_num:0.4f}")