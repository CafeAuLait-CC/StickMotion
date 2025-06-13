import torch
from collections import OrderedDict
import os

path = 'stickman/weight/real_init/t2m'
os.makedirs(path, exist_ok=True)
weight_path = "stickman/logs/human_ml3d/fix_init/last.ckpt"
wei = torch.load(weight_path, map_location='cpu')['state_dict']

model_set = set()
for key in list(wei.keys()):
    model_set.add(key.split('.')[0])

for name in model_set:
    new_wei = OrderedDict()
    for key in list(wei.keys()):
        if key.startswith(name):
            new_wei[key[len(name)+1:]] = wei[key]
    torch.save(new_wei, f'{path}/{name}.ckpt')
# new_wei = OrderedDict()
# for key in list(wei.keys()):
#     if key.startswith('stickman_encoder'):
#         new_wei[key[17:]] = wei[key]