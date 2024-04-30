import sys
import os
import os.path as osp
this_dir = osp.dirname(__file__)
path = osp.join(this_dir, '..')
if path not in sys.path: sys.path.insert(0, path)

from share import *
import datetime
import argparse
import json
from tqdm import tqdm
import cv2
import einops
import numpy as np
import torch
import random
import datetime
import math
from torchvision import transforms
from scipy.spatial.transform import Rotation

from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import MyDataset
from annotator.util import resize_image, HWC3
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


parser = argparse.ArgumentParser()
parser.add_argument('--strength', required=False, type=float, default=1.0)
parser.add_argument('--scale', required=False, type=float, default=3.0)
parser.add_argument('--ddim_steps', required=False, type=float, default=20)
args = parser.parse_args()


# Configs
image_resolution = 256
model = create_model('./models/zero123.yaml').cpu()
model_path = 'experiment/exp_08-26_20:15:15/controlnet/version_0/checkpoints/epoch=21-step=274999.ckpt'
 
model.load_state_dict(load_state_dict(model_path, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

num_samples = 1
strength = args.strength
scale = args.scale
guess_mode = False
eta = 0.0
ddim_steps = args.ddim_steps
text_prompt = "human, isolated on white background, best quality, extremely detailed"
seed_everything(0) 


# Logging
KST = datetime.timezone(datetime.timedelta(hours=9))
save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-13]
save_folder = save_folder.replace(" ", "_")
save_folder_path = 'experiment/{}'.format(save_folder)
os.makedirs(save_folder_path, exist_ok=True)

img_dir = 'data/THuman2.0/parsed'
db = json.load(open('data/THuman2.0/thuman_test.json', 'r'))
datalist = []
for subject in db.keys(): datalist.extend(db[subject])


for i, data in tqdm(enumerate(datalist)):
    img_path = osp.join(img_dir, data['input_img_path'])
    input_img = cv2.imread(img_path)

    taget_img_path = osp.join(img_dir, data['target_img_path'])

    depth_path = taget_img_path.replace('/img', '/depth').replace('.jpg', '.png')
    control_img = cv2.imread(depth_path)
    # normal_path = taget_img_path.replace('/img', '/normal').replace('.jpg', '.png')
    # control_img = cv2.imread(normal_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    control_img = cv2.cvtColor(control_img, cv2.COLOR_BGR2RGB)

    input_img = cv2.resize(input_img, (image_resolution, image_resolution))
    control_img = cv2.resize(control_img, (image_resolution, image_resolution))

    R_1 = np.load(img_path.replace('/img', '/camera').replace('.jpg', '.npz'))['R']
    R_2 = np.load(taget_img_path.replace('/img', '/camera').replace('.jpg', '.npz'))['R']
    R = R_2 @ R_1.transpose()
    R = Rotation.from_matrix(R)
    R = R.as_euler('yxz', degrees=True)
    x, y, z = R[1] * -1, R[0] * -1, 0        

    with torch.no_grad():
        input_image = HWC3(input_img)
        img = resize_image(input_image, image_resolution)
        img = img.astype(np.float32) / 255.0
        img = transforms.ToTensor()(img).unsqueeze(0).cuda()
        img = img * 2 - 1
        H, W = image_resolution, image_resolution

        c_cat = model.encode_first_stage(img).mode()
        T = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), 0.0])
        T = T.unsqueeze(0).cuda()
        c = model.get_learned_conditioning(img)
        c = torch.cat([c, T[:,None]], dim=-1)
        c_crossattn = model.cc_projection(c)


        depth = control_img
        depth = cv2.resize(depth, (image_resolution, image_resolution))
        c_control = depth.astype(np.float32) / 255.0
        c_control = transforms.ToTensor()(c_control).unsqueeze(0).cuda()
        c_control = c_control * 2 - 1      


        uc_cat = torch.zeros_like(c_cat).to(c_cat.device)
        uc_crossattn = torch.zeros_like(c_crossattn).to(c_crossattn.device)
        cond = {"c_concat": [c_cat.repeat(num_samples,1,1,1)], "c_crossattn": [c_crossattn.repeat(num_samples,1,1)], "c_control": [c_control.repeat(num_samples,1,1,1)]}
        un_cond = {"c_concat": [uc_cat.repeat(num_samples,1,1,1)], "c_crossattn": [uc_crossattn.repeat(num_samples,1,1)]}
        shape = (4, H // 8, W // 8)

        model.control_scales = ([strength] * 13)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)


        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_folder_path, f'{i:04d}.png'), x_samples[0][:,:,::-1])