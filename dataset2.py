import json
import cv2
import numpy as np
import os
import os.path as osp

from torch.utils.data import Dataset

import glob

class MyDataset(Dataset):
    def __init__(self):
        self.db = json.load(open('data/deepfashion/deepfashion_train.json'))


    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        data = self.db[idx]
        path, prompt = data['path'], data['prompt']

        name = path.split('/')[-1]
        source_filename = path
        target_filename = path.replace(name, 'image.png')
        prompt = prompt

        source = cv2.imread(source_filename, 0)
        source = cv2.resize(source, (512,512))[:,:,None]
        target = cv2.imread(target_filename)
        target = cv2.resize(target, (512, 512))
        target = (target * (source>128))

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

