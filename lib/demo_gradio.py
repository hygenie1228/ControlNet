import sys
import os
import os.path as osp
this_dir = osp.dirname(__file__)
path = osp.join(this_dir, '..')
if path not in sys.path: sys.path.insert(0, path)

from share import *
import config

import cv2
import einops
import gradio as gr
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import random
import trimesh
from pyrender.constants import RenderFlags
import math
os.environ['PYOPENGL_PLATFORM']  = 'egl'
import pyrender

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# Configs
image_resolution = 256
model = create_model('./models/zero123.yaml').cpu()
model_path = 'experiment/exp_08-26_20:15:15/controlnet/version_0/checkpoints/epoch=21-step=274999.ckpt'
model.load_state_dict(load_state_dict(model_path, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R


class Renderer:
    def __init__(self, resolution=(512,512), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, img, mesh, cam=[5000,5000,256,256], angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):
        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        fx, fy, cx, cy = cam
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, depth = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)
        
        d_min = np.sort(np.unique(depth))[1]
        d_max = depth.max()
        depth = (depth - d_min) / (d_max - d_min)
        depth = (depth * 255).astype(np.uint8)

        depth = 255 - depth
        depth = depth * 0.9 * valid_mask + (1 - valid_mask) * np.ones_like(depth)*255

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)
        return depth #, normal



def process(input_image, debug_txt, x, y, num_samples, ddim_steps, strength, scale, seed, eta):
    if input_image is None: return
    seed_everything(seed) 

    with torch.no_grad():
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        img = img.astype(np.float32) / 255.0
        img = transforms.ToTensor()(img).unsqueeze(0).cuda()
        img = img * 2 - 1
        H, W = image_resolution, image_resolution

        if debug_txt == "1":
            mesh_path = 'demo/demo_1.obj'
        elif debug_txt == "2":
            mesh_path = 'demo/demo_2.obj'
        elif debug_txt == "3":
            mesh_path = 'demo/demo_3.obj'
        else:
            assert 0 

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=True)

        mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
        mesh = mesh.copy()
        R = np.matmul(make_rotate(math.radians(x), 0, 0), make_rotate(0, math.radians(y), 0))
        mesh.vertices = mesh.vertices @ R.T
        mesh.vertices = mesh.vertices @ np.array([[ 1., -0.,  0.],[ 0., -1., -0.],[ 0.,  0., -1.]])
        mesh.vertices[:,2] += 10

        c_cat = model.encode_first_stage(img).mode()
        T = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), 0.0])
        T = T.unsqueeze(0).cuda()
        c = model.get_learned_conditioning(img)
        c = torch.cat([c, T[:,None]], dim=-1)
        c_crossattn = model.cc_projection(c)

        renderer = Renderer()
        depth = np.ones((512,512,3)) * 255
        depth = renderer.render(depth, mesh)
        depth = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
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

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)    
        results = [x_samples[i][:,:,::-1].astype(np.uint8) for i in range(len(x_samples))]

    return [depth] + results



block = gr.Blocks().queue()
with block:
    with gr.Row(): 
        gr.Markdown("## Novel View Synthesis for Human")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            debug_txt = gr.Textbox(label="", visible=False)

            gr.Examples(
                examples=[["1", "demo/demo_1.jpg"], ["2", "demo/demo_2.jpg"], ["3", "demo/demo_3.jpg"]],
                inputs=[debug_txt, input_image],
                outputs=[],
                fn=process,
                cache_examples=False,
            )

            polar_slider = gr.Slider(
                -90, 90, value=0, step=5, label='Polar angle (vertical rotation in degrees)')
            azimuth_slider = gr.Slider(
                -180, 180, value=0, step=5, label='Azimuth angle (horizontal rotation in degrees)')

            run_button = gr.Button(label="Run")

            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=20.0, value=3.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)

        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')

    ips = [input_image, debug_txt, polar_slider, azimuth_slider, num_samples, ddim_steps, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0', server_port=12345, share=True)
