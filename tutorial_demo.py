from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_openpose = OpenposeDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./lightning_logs/version_2/checkpoints/epoch=2-step=37499.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


input_image = cv2.imread('demo/input.png')
text_prompt = "best quality, extremely detailed"

image_resolution, detect_resolution = 512, 512
num_samples = 1
strength = 1.0
scale = 9.0
eta = 0.0
ddim_steps = 20
guess_mode = True

with torch.no_grad():
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    input_image = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_NEAREST)

    control = torch.from_numpy(input_image.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([text_prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([''] * num_samples)]}
    shape = (4, H // 8, W // 8)

    # cond['c_concat']: control_image, cond['c_crossattn']: positive prompt
    # cond['c_concat']: control_image, cond['c_crossattn']: negative prompt

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
    cv2.imwrite('debug.png', x_samples[0][:,:,::-1])