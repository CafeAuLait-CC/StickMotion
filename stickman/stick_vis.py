import os
from os.path import join as pjoin
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
torch.set_float32_matmul_precision('high')
from mmcv import Config
from  lightning import Trainer, LightningModule
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from stickman.dataset import StickmanDataset
from stickman.model import StickmanEncoder, MotionEncoder, StickmanDecoder
from stickman.loss import SLoss
from stickman.eval_with_eye import eval_vis
import numpy as np

if __name__ == '__main__':
    poses = np.load('vis_kit_pose.npy') 
    tracks = np.load('vis_kit_tracks.npy')
    eval_vis(tracks[20:30], poses[20:30], joints_num=21)
    # cfg_file = "configs/remodiffuse/remodiffuse_kit.py"
    # # cfg_file = "configs/remodiffuse/remodiffuse_t2m.py"
    # cfg = Config.fromfile(cfg_file)
    # train_ratio = 0.8
    # train_dataset = StickmanDataset(cfg, train=True, train_ratio=train_ratio)
    # val_dataset = StickmanDataset(cfg, train=False, train_ratio=train_ratio)
    # # test_dataset = StickmanDataset(cfg, test=True)
    # cfg = cfg.stick_set
    # batch_size = cfg.train.batch_size
    # epochs = cfg.train.epochs
    # workers = cfg.train.workers
    # train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=workers)
    # val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=workers)
    # for batch in train_loader:
    #     poses = batch['norm_pose'].cpu().numpy()
    #     tracks = batch['track'].cpu().numpy()
    #     top = 10
    #     eval_vis(tracks[:top], poses[:top], joints_num=21)