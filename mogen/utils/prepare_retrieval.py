import torch
from mmcv import Config, DictAction

from mogen import __version__
from mogen.datasets import build_dataset
from mogen.models.transformers.remodiffuse import RetrievalDatabase
import numpy as np 
from tqdm import tqdm
import clip

device = "cuda:7"
# cfg = Config.fromfile("configs/remodiffuse/remodiffuse_t2m.py")
cfg = Config.fromfile("configs/remodiffuse/remodiffuse_kit.py")
save_pth = cfg.model.model.retrieval_cfg.reidx_file
cfg.data.train.times = 1
cfg.model.model.retrieval_cfg.num_retrieval = 3

dataset = build_dataset(cfg.data.train)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.data.workers_per_gpu)


retrievaldb = RetrievalDatabase(**cfg.model.model.retrieval_cfg)
clip_model, _ = clip.load('ViT-B/32', device)

retrievaldb.to(device)
retrievaldb.tensor_to(device)
retrievaldb.eval()
clip_model.eval()

all_ret_indexs = []
# for dl_idx, data in tqdm(enumerate(dataloader), total=len(dataset)):
for db_idx in tqdm(range(len(dataset))):
    data = dataset[db_idx]
    motion, motion_mask = data['motion'].float(), data['motion_mask'].float()
    text = data['motion_metas'].data['text']
    text_idx = data['text_idx'].item()
    clip_feat = data['clip_feat']
    motion_length = data['motion_length'].item()
    sample_idx = data['sample_idx'].item()
    assert db_idx == sample_idx
    with torch.no_grad():
        ret_indexs = retrievaldb.retrieve(text, motion_length, clip_model, device)
        all_ret_indexs.append(ret_indexs)

np.save(save_pth, np.array(all_ret_indexs))

