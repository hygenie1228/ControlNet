import sys
import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'
this_dir = osp.dirname(__file__)
path = osp.join(this_dir, '..')
if path not in sys.path: sys.path.insert(0, path)

from share import *
import datetime
import argparse
import json
from tqdm import tqdm
import cv2
import math
import einops
import numpy as np
import torch
from torchvision import transforms
import random
import datetime
import trimesh
import pyrender
from pyrender.constants import RenderFlags

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

        # normal = converter.convert(depth)
        # normal = normal * valid_mask[:,:,None] + (1 - valid_mask[:,:,None]) * img
        depth = 255 - depth
        depth = depth * 0.9 * valid_mask + (1 - valid_mask) * np.ones_like(depth)*255

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)
        return depth #, normal



# Configs
image_resolution = 256
model = create_model('./models/zero123.yaml').cpu()
model_path = 'experiment/exp_08-30_22:49:02/controlnet/version_0/checkpoints/epoch=21-step=274999.ckpt'
 
model.load_state_dict(load_state_dict(model_path, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

num_samples = 1
strength = args.strength
scale = args.scale
guess_mode = False
eta = 0.0
ddim_steps = args.ddim_steps
seed_everything(0) 
renderer = Renderer()

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
data = datalist[0]

img_path = osp.join(img_dir, data['input_img_path'])
input_img = cv2.imread(img_path)
input_img = cv2.resize(input_img, (image_resolution, image_resolution))
cv2.imwrite('debug.png', input_img)
input_image = HWC3(input_img)
H, W, C = input_img.shape

img_name = img_path.split('/')[-1]
mesh_path = img_path.replace('/img', '/smpl_param').replace(img_name, '') + 'smpl_mesh.obj'
orig_mesh = trimesh.load(mesh_path, process=False, maintain_order=True)


img = input_img.astype(np.float32) / 255.0
img = transforms.ToTensor()(img).unsqueeze(0).cuda()
img = img * 2 - 1
H, W = image_resolution, image_resolution


for i in tqdm(range(180)):
    mesh = orig_mesh.copy()
    R = make_rotate(0, math.radians(2*i), 0)

    mesh.vertices = mesh.vertices @ R.T
    mesh.vertices = mesh.vertices @ np.array([[ 1., -0.,  0.],[ 0., -1., -0.],[ 0.,  0., -1.]])
    mesh.vertices[:,2] += 10

    with torch.no_grad():
        depth = np.ones((512,512,3)) * 255
        depth = renderer.render(depth, mesh)
        depth = cv2.resize(depth, (image_resolution, image_resolution)).astype(np.uint8)
        depth = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        c_control = input_img.astype(np.float32) / 255.0
        c_control = transforms.ToTensor()(c_control).unsqueeze(0).cuda()
        c_control = c_control * 2 - 1 
    
        x, y, z = 0, 2*i, 0
        c_cat = model.encode_first_stage(img).mode()
        T = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), 0.0])
        T = T.unsqueeze(0).cuda()
        c = model.get_learned_conditioning(img)
        c = torch.cat([c, T[:,None]], dim=-1)
        c_crossattn = model.cc_projection(c)

        
        uc_cat = torch.zeros_like(c_cat).to(c_cat.device)
        uc_crossattn = torch.zeros_like(c_crossattn).to(c_crossattn.device)
        cond = {"c_concat": [c_cat.repeat(num_samples,1,1,1)], "c_crossattn": [c_crossattn.repeat(num_samples,1,1)], "c_control": [c_control.repeat(num_samples,1,1,1)]}
        un_cond = {"c_concat": [uc_cat.repeat(num_samples,1,1,1)], "c_crossattn": [uc_crossattn.repeat(num_samples,1,1)]}
        shape = (4, H // 8, W // 8)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        cv2.imwrite(osp.join('debug', f'{i:04d}.png'), x_samples[0][:,:,::-1])

