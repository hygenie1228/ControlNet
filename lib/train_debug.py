import sys
import os
import os.path as osp
this_dir = osp.dirname(__file__)
path = osp.join(this_dir, '..')
if path not in sys.path: sys.path.insert(0, path)

from share import *
import datetime

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 1000
learning_rate = 1e-4
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/depthmap.yaml').cpu()
import pdb; pdb.set_trace()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Logging
KST = datetime.timezone(datetime.timedelta(hours=9))
save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-13]
save_folder = save_folder.replace(" ", "_")
save_folder_path = 'experiment/{}'.format(save_folder)
save_codes_path = osp.join(save_folder_path, 'codes')
os.makedirs(save_folder_path, exist_ok=True)
os.system(f'cp -r lib {save_codes_path}')


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_folder_path, name='controlnet')
ckpt_callback = ModelCheckpoint(every_n_epochs=2, save_top_k=-1)
trainer = pl.Trainer(gpus=1, precision=32, logger=tb_logger, callbacks=[ckpt_callback, logger])

# Train!
trainer.fit(model, dataloader)
