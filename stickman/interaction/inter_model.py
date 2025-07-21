from os.path import join as pjoin
import os
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
from stickman.eval_with_eye import eval_vis, pose_vis_online
from stickman.utils import resample_polyline

class StickModel(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.stickman_encoder = StickmanEncoder(cfg.stickman_encoder)
        # self.motion_encoder = MotionEncoder(cfg.motion_encoder)
        self.stickman_decoder = StickmanDecoder(cfg.stickman_decoder)
        self.loss = SLoss(cfg)
        self.cfg = cfg
        self.save_hyperparameters(cfg)

    def forward(self, batch):
        track = batch['track']

        stickman_feat = self.stickman_encoder(track)
        predict_pose = self.stickman_decoder(stickman_feat)
        
        return stickman_feat, predict_pose

    def training_step(self, batch, batch_idx):
        norm_pose = batch['norm_pose']
        # stickman_feat, motion_feat, predict_pose = self(batch)
        stickman_feat, predict_pose = self(batch)
        loss_dict = self.loss(stickman_feat,  predict_pose, norm_pose)
        loss = 0
        for k,v in loss_dict.items():
            loss += v
            self.log(f'train_{k}', v, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.trainer.max_epochs-1], gamma=0.1)
        return [optimizer], [lr_scheduler]
    
    def predict_step(self, stickman):
        input_stickman = torch.tensor([stickman]).to(self.device).to(torch.float32)
        batch1 = {
            'track': input_stickman,
            'motion': None,
            'norm_pose': None,
        }
        stickman_feat, predict_pose = self(batch1)
        predict_pose = predict_pose.detach().cpu().numpy()[0] # [4, 21, 3]
        buf = pose_vis_online(predict_pose, joints_num=21)
        
        return buf
        
    
    def validation_step(self, batch, batch_idx):
        
        # norm_pose = batch['norm_pose']
        # poses = batch['norm_pose'].cpu().numpy()
        # tracks = batch['track'].cpu().numpy()
        # top = 10
        # eval_vis(tracks[:top], poses[:top], joints_num=21)
        # os._exit(0)
        # stickman_feat, predict_pose = self(batch)
        
        ''' 
        ### only vis track and pose
        gt_poses = batch['norm_pose'].cpu().numpy()
        tracks = batch['track'].cpu().numpy()
        stickman_feat, predict_pose = self(batch)
        predict_poses = predict_pose.cpu().numpy()
        top = 10
        eval_vis(tracks[:top], predict_poses[:top,0], joints_num=22)
        
        
        ### dataset input
        import numpy as np
        id = 0
        batch1 = {
            'track': batch['track'][id, None],
            'motion': None,
            'norm_pose': None,
        }
        stickman_input = batch['track'][id].cpu().numpy()
        stickman_feat, predict_pose = self(batch1)
        eval_vis([stickman_input], [predict_pose[0,0].cpu().numpy()], joints_num=22)
        
        ## web input
        import numpy as np
        stickman_input = np.load(".vscode/interaction/stickman_input.npy", allow_pickle=True)
        batch1 = {
            'track': torch.tensor([stickman_input]).to('cuda').to(torch.float32),
            'motion': None,
            'norm_pose': None,
        }
        stickman_feat, predict_pose = self(batch1)
        np.save(".vscode/interaction/ppose.npy", predict_pose[0].cpu().numpy())
        ppose = predict_pose[0].cpu().numpy()
        eval_vis([stickman_input], ppose, joints_num=21)
        '''

        norm_pose = batch['norm_pose']
        stickman_feat, predict_pose = self(batch)
        loss_dict = self.loss(stickman_feat, predict_pose, norm_pose)
        loss = 0
        for k,v in loss_dict.items():
            loss += v
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

def get_pose_model(
    cfg_path='configs/remodiffuse/remodiffuse_kit.py',
    weight_path='stickman/logs/kit_ml/fix_init/last.ckpt',
    device='cuda'):
    cfg = Config.fromfile(cfg_path)
    cfg = cfg.stick_set

    ckpt = os.path.abspath(weight_path)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'checkpoint not found: {ckpt}')

    model = StickModel.load_from_checkpoint(ckpt, map_location=device, cfg=cfg)
    model.to(device)
    model.eval()
    
    return model.predict_step

