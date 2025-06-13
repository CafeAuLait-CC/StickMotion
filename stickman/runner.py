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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.trainer.max_epochs-1], gamma=0.1)
        return [optimizer], [lr_scheduler]
    
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



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('config', type=str)
    parser.add_argument('version', nargs='?', default='debug', help='Set version')
    args = parser.parse_args()
    # cfg_file = "configs/remodiffuse/remodiffuse_kit.py"
    cfg_file = "configs/remodiffuse/remodiffuse_t2m.py"
    cfg = Config.fromfile(cfg_file)
    train_ratio = 0.8
    train_dataset = StickmanDataset(cfg, train=True, train_ratio=train_ratio)
    val_dataset = StickmanDataset(cfg, train=False, train_ratio=train_ratio)
    # test_dataset = StickmanDataset(cfg, test=True)

    cfg = cfg.stick_set

    batch_size = cfg.train.batch_size
    epochs = cfg.train.epochs
    workers = cfg.train.workers

    cwd_path = os.path.abspath(os.path.dirname(__file__))
    log_path = pjoin(cwd_path, 'logs')

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True, num_workers=workers)
    logger = TensorBoardLogger(save_dir=log_path, name=train_dataset.dataset_name, version=args.version)
    # test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=workers)
    # logger = TensorBoardLogger(save_dir=log_path, name=test_dataset.dataset_name, version=args.version)
    os.makedirs(logger.log_dir, exist_ok=True)
    os.system(f"cp {cfg_file} {logger.log_dir}/")

    checkpoint_callback = ModelCheckpoint(
            dirpath = logger.log_dir,
            monitor = "train_loss",
            mode = 'min',
            save_top_k = -1,
            every_n_epochs = max(1, epochs//10),
            save_last=True
        )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model = StickModel(cfg)
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
                    # num_sanity_val_steps=0,
                      logger=logger,
                      callbacks=[
                          checkpoint_callback,
                          lr_monitor,
                          ]
                      )     
    
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    path = "stickman/logs/human_ml3d/fix_init/last.ckpt"
    model = StickModel.load_from_checkpoint(path, cfg=cfg)
    trainer.validate(model, dataloaders=val_loader)