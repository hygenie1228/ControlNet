import json
import cv2
import numpy as np
import os
import os.path as osp
import random

from torch.utils.data import Dataset

import glob

def get_aug_config():
    scale = np.clip(np.random.randn(), -1.0, 1.0) * 0.3 + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * 30 if random.random() <= 0.6 else 0
    shift = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))
    c_up = 1.0 + 0.2
    c_low = 1.0 - 0.2
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    do_flip = False
    return scale, rot, shift, color_scale, do_flip

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, shift=(0, 0), inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    t_x, t_y = src_w*shift[0], src_h*shift[1]
    src_center = np.array([c_x+t_x, c_y+t_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        inv_trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

    trans, inv_trans = trans.astype(np.float32), inv_trans.astype(np.float32)
    return trans, inv_trans

def generate_patch_image(cvimg, bbox, scale, rot, shift, do_flip, out_shape):
    img = cvimg.copy()
   
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans, inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, shift)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    
    if do_flip:
        img_patch = img_patch[:, ::-1, :]

    return img_patch, trans, inv_trans



class MyDataset(Dataset):
    def __init__(self):
        self.db = json.load(open('data/deepfashion/deepfashion_train.json'))


    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        data = self.db[idx]
        path, prompt = data['path'], data['prompt']

        name = path.split('/')[-1]
        path = osp.join('data/deepfashion', path)
        source_filename = path
        target_filename = path.replace('_segm', '')

        source = cv2.imread(source_filename, 0)
        source = cv2.resize(source, (512,512))[:,:,None]
        target = cv2.imread(target_filename, -1)
        target = cv2.resize(target, (512, 512))
        target_mask = target[:,:,[3]]
        target = target[:,:,:3]
        
        # augmentation
        scale, rot, shift, color_scale, do_flip = get_aug_config()
        target, trans, inv_trans = generate_patch_image(target, [0,0,512,512], scale, rot, shift, do_flip, (512,512))
        target = target * color_scale[None,None,:]
        target = np.clip(target, 0, 255)
        source = cv2.warpAffine(source, trans, (512,512), flags=cv2.INTER_LINEAR)
        target_mask = cv2.warpAffine(target_mask, trans, (512,512), flags=cv2.INTER_LINEAR)

        target_mask = (target_mask>128) * 1.0
        bg_colors = np.random.uniform(low=0.0, high=256, size=3).astype(np.uint8)
        bg_colors = np.ones_like(target) * bg_colors[None,None,:]
        target = target * target_mask[:,:,None] + bg_colors * (1-target_mask[:,:,None] )

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        source = source[:,:,None]

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        return dict(jpg=target, txt=prompt, hint=source)

