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
parser.add_argument('--scale', required=False, type=float, default=9.0)
parser.add_argument('--ddim_steps', required=False, type=float, default=20)
args = parser.parse_args()


# Configs
image_resolution = 256
model = create_model('./models/depthmap.yaml').cpu()
model_path = 'experiment/exp_08-22_17:26:04/controlnet/version_0/checkpoints/epoch=31-step=399999.ckpt'
 
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

    input_img = cv2.resize(input_img, (image_resolution, image_resolution))
    control_img = cv2.resize(control_img, (image_resolution, image_resolution))

    with torch.no_grad():
        input_image = HWC3(input_img)
        control_image = HWC3(control_img)
        H, W, C = input_img.shape

        input_image = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_NEAREST)

        control_image = np.concatenate((input_image, control_image), -1)
        control = torch.from_numpy(control_image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_cross = model.get_unconditional_conditioning(1)
        if text_prompt is not None:
            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([text_prompt] * num_samples)]}
        else:
            cond = {"c_concat": [control], "c_crossattn": [uc_cross]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [uc_cross]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_folder_path, f'{i:04d}.png'), x_samples[0][:,:,::-1])