import json
import glob
import os
import os.path as osp
from tqdm import tqdm

dir_list = sorted(glob.glob('data/deepfashion/deepfashion_train/*'))

save_list = []
for dir_path in tqdm(dir_list):
    prompts = json.load(open(osp.join(dir_path, 'prompt.json')))
    for cloth_type in ['upper-cloth', 'jacket', 'pants', 'skirt', 'shoes']:
        if not cloth_type in prompts.keys():
            continue
        
        path = osp.join(dir_path, 'png', f'{cloth_type}_segm.png')
        if osp.isfile(path):
            save_data = {}
            save_data['path'] = path
            save_data['prompt'] = prompts[cloth_type]
            save_list.append(save_data)

json.dump(save_list, open('/home/namhj/ControlNet/data/deepfashion/deepfashion_train.json', 'w'))
import pdb; pdb.set_trace()