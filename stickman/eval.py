import os
from os.path import join as pjoin
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
from copy import deepcopy
from mogen.models.transformers.remodiffuse import MultiStickEncoder
torch.set_float32_matmul_precision('high')
from mmcv import Config
from  lightning import Trainer, LightningModule
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from stickman.dataset import RealStickmanDataset
from stickman.model import StickmanEncoder, MotionEncoder, StickmanDecoder
from stickman.loss import SLoss
from stickman.eval_with_eye import motion_vis

class StickModel(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.multistick_encoder = MultiStickEncoder(**cfg) 
        self.num_joints = 21
        self.sup_pose = (self.num_joints - 1) * 3
        self.proj = torch.nn.Linear(cfg.d_model, self.sup_pose)

    def forward(self, track):

        stickman_feat = self.multistick_encoder(track)
        
        return stickman_feat

    def training_step(self, batch, batch_idx):
        norm_pose = batch['norm_pose']
        track = batch['track']
        motion = batch['motion']
        motion = motion[:,:,4:4+self.sup_pose]
        stickman_feat = self(track)
        pred_motion = self.proj(stickman_feat[:,[0,-1]])
        loss = (pred_motion - motion).abs().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0)
        return [optimizer], [lr_scheduler]
    
    def validation_step(self, batch, batch_idx):
        norm_pose = batch['norm_pose']
        track = batch['track']
        motion = batch['motion']
        motion = motion[:,:,4:4+self.sup_pose]
        stickman_feat = self(track)
        pred_motion = self.proj(stickman_feat[:,[0,-1]])
        loss = (pred_motion - motion).abs().mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
        # norm_pose = batch['norm_pose']
        # track = batch['track']
        # motion = batch['motion']
        # motion = motion[:,:,4:4+self.sup_pose]
        # stickman_feat = self(track)
        # pred_motion = self.proj(stickman_feat[:,[0,-1]])
        # _mean = torch.tensor(self.trainer.val_dataloaders.dataset.mean).to(pred_motion.device)
        # _std = torch.tensor(self.trainer.val_dataloaders.dataset.std).to(pred_motion.device)
        # mean = _mean[4:4+self.sup_pose]
        # std = _std[4:4+self.sup_pose]
        # gt_motion = _mean + _std * batch['motion']
        # pred_motion = pred_motion * std + mean
        # idx = 30
        # _gt_motion = gt_motion[idx][0]
        # _pred_motion = deepcopy(_gt_motion)
        # _pred_motion[4:4+self.sup_pose] = pred_motion[idx][0]
        # motion_vis(_pred_motion[None].to(torch.float32), _gt_motion[None].to(torch.float32))
        # loss = (pred_motion - motion).abs().mean()
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # return loss



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('config', type=str)
    parser.add_argument('--version', type=str, default='test')
    args = parser.parse_args()
    cfg_file = "configs/remodiffuse/remodiffuse_kit.py"
    cfg = Config.fromfile(cfg_file)
    model_cfg = cfg.model.model.multistick_encoder
    train_ratio = 0.8
    train_dataset = RealStickmanDataset(cfg, train=True, train_ratio=train_ratio)
    val_dataset = RealStickmanDataset(cfg, train=False, train_ratio=train_ratio)

    cfg = cfg.stick_set

    batch_size = cfg.train.batch_size
    epochs = cfg.train.epochs
    workers = cfg.train.workers

    cwd_path = os.path.abspath(os.path.dirname(__file__))
    log_path = pjoin(cwd_path, 'logs')

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=workers)
    logger = TensorBoardLogger(save_dir=log_path, name=train_dataset.dataset_name, version=args.version)
    os.makedirs(logger.log_dir, exist_ok=True)
    os.system(f"cp {cfg_file} {logger.log_dir}/")

    checkpoint_callback = ModelCheckpoint(
            dirpath = logger.log_dir,
            monitor = "val_loss_epoch",
            mode = 'min',
            save_top_k = 3,
            every_n_epochs = 3,
            save_last=True
        )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model = StickModel(model_cfg)
    trainer = Trainer(accelerator="gpu", 
                      strategy=DDPStrategy(),
                      check_val_every_n_epoch=1,
                    #   devices=[1,2],
                      devices=[7],
                    #   max_epochs=1,
                    #   max_steps=7,
                      max_epochs=epochs,
                    #   precision='16-mixed',
                    #   gradient_clip_algorithm="norm",
                    #   gradient_clip_val=0.5,
                      logger=logger,
                      callbacks=[
                          checkpoint_callback,
                          lr_monitor,
                          ]
                      )     
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # path = "stickman/logs/kit_ml/crop/last.ckpt"
    # model = StickModel.load_from_checkpoint(path, cfg=model_cfg)
    # trainer.validate(model, dataloaders=val_loader)