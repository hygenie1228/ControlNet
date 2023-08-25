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
parser.add_argument('--scale', required=False, type=float, default=9.0)
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
model = create_model('./models/depthmap.yaml').cpu()
model_path = 'experiment/exp_08-22_17:26:04/controlnet/version_0/checkpoints/epoch=25-step=324999.ckpt'

model_dict = load_state_dict('models/zero123.ckpt', location='cuda')
target_dict = {}
for k, v in model_dict.items():
    if k.startswith('model_ema.'): continue
    else: target_dict[k] = v

model.load_state_dict(target_dict, strict=False)
# model.load_state_dict(load_state_dict(model_path, location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

num_samples = 1
n_samples = num_samples
strength = args.strength
scale = args.scale
guess_mode = False
eta = 0.0
ddim_steps = args.ddim_steps
text_prompt = "human, isolated on white background, best quality, extremely detailed"
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
data = datalist[747]


img_path = osp.join(img_dir, data['input_img_path'])
input_img = cv2.imread('/home/namhj/ControlNet/input.png')[:,:,::-1]
input_img = cv2.resize(input_img, (image_resolution, image_resolution))
input_im = (input_img / 255.0).astype(np.float32)
# input_image = HWC3(input_img)
H, W, C = input_img.shape

img_name = img_path.split('/')[-1]
mesh_path = img_path.replace('/img', '/smpl_param').replace(img_name, '') + 'smpl_mesh.obj'
orig_mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
img = np.ones((512,512,3)) * 255

for i in tqdm(range(180)):
    mesh = orig_mesh.copy()
    R = make_rotate(0, math.radians(2*i), 0)

    mesh.vertices = mesh.vertices @ R.T
    mesh.vertices = mesh.vertices @ np.array([[ 1., -0.,  0.],[ 0., -1., -0.],[ 0.,  0., -1.]])
    mesh.vertices[:,2] += 10

    with torch.no_grad():
        depth = renderer.render(img, mesh)
        depth = cv2.resize(depth, (image_resolution, image_resolution)).astype(np.uint8)
        control_image = HWC3(depth)

        input_im = transforms.ToTensor()(input_im).unsqueeze(0)
        # input_im = input_im[0].permute(1,2,0).cpu().numpy()
        # cv2.imwrite('debug2.png', input_im[:,:,::-1]*255)
        # import pdb; pdb.set_trace()
        input_im = input_im * 2 - 1
        input_im = input_im.to(model.device)
        
        R_1 = np.load('/home/namhj/ControlNet/data/THuman2.0/parsed/0000/camera/0000.npz')['R']
        R_2 = np.load('/home/namhj/ControlNet/data/THuman2.0/parsed/0000/camera/0003.npz')['R']
        R = R_2 @ R_1.transpose()
        R = Rotation.from_matrix(R)
        R = R.as_euler('yxz', degrees=True)
        x, y, z = R[1] * -1, R[0] * -1, 0        

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
        T = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), z])
        T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
        c = torch.cat([c, T], dim=-1)
        c = model.cc_projection(c)
        
        cond = {}
        cond['c_crossattn'] = [c]
        cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach().repeat(n_samples, 1, 1, 1)]

        un_cond = {}
        un_cond['c_concat'] = [torch.zeros(n_samples, 4, H // 8, W // 8).to(c.device)]
        un_cond['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
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
        cv2.imwrite('debug.png', x_samples[0][:,:,::-1])
        import pdb; pdb.set_trace()
