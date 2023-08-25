import json
import cv2
import numpy as np
import json
import os
import os.path as osp

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.img_dir = 'data/THuman2.0/parsed'
        self.datalist = json.load(open('data/THuman2.0/thuman_train.json'))
        self.subject_list = list(self.datalist.keys())
        self.subject_len = len(self.subject_list)

    def __len__(self):
        return 50000

    def __getitem__(self, idx):
        idx = idx // self.subject_len
        subject = self.subject_list[idx]
        data = self.datalist[subject]

        i1, i2 = np.random.choice(len(data), size=2, replace=True)
        img_path_1 = osp.join(self.img_dir, data[i1]['img_path'])
        img_path_2 = osp.join(self.img_dir, data[i2]['img_path'])
        depth_path_2 = osp.join(self.img_dir, data[i2]['img_path'].replace('/img', '/depth').replace('.jpg', '.png'))

        source_1 = cv2.imread(img_path_1)
        source_2 = cv2.imread(depth_path_2)
        target = cv2.imread(img_path_2)

        # Do not forget that OpenCV read images in BGR order.
        source_1 = cv2.cvtColor(source_1, cv2.COLOR_BGR2RGB)
        source_2 = cv2.cvtColor(source_2, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # source_1 = cv2.resize(source_1, (256, 256))
        # source_2 = cv2.resize(source_2, (256, 256))
        # target = cv2.resize(target, (256, 256))

        source = np.concatenate((source_1 ,source_2), -1)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        text_prompt = 'human, isolated on white background, best quality, extremely detailed'
        return dict(jpg=target, txt=text_prompt, hint=source, pose=np.zeros((10,)))

