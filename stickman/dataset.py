# stick man algorithm by wangtao@bupt.edu.cn

import my_tools
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from mmcv import Config

from stickman.utils import StickMan
from mogen import __version__
from mogen.datasets import build_dataset
from mogen.utils.plot_utils import (
    recover_from_ric,
    plot_3d_motion,
    t2m_kinematic_chain,
    plot_3d_motion_kit,
    kit_kinematic_chain
)

class StickmanDataset(Dataset):
    def __init__(self, cfg, train_ratio=0.8, train=True, test=False):
        super().__init__()
        device = "cpu"
        cfg.data.train.times = 1
        if test:
            dataset = build_dataset(cfg.data.test)
            dataset_name = dataset.dataset_name
        else:
            dataset = build_dataset(cfg.data.train)
            dataset_name = dataset.dataset.dataset_name
        self.dataset_name = dataset_name
        self.stickman = StickMan(dataset_name)
        if dataset_name == 'human_ml3d':
            #assert dataset_name == "human_ml3d"
            mean_path = "data/datasets/human_ml3d/mean.npy"
            std_path = "data/datasets/human_ml3d/std.npy"
            self.joint_num = 22
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
        elif dataset_name == 'kit_ml':  
            #assert dataset_name == "kit_ml"
            mean_path = "data/datasets/kit_ml/mean.npy"
            std_path = "data/datasets/kit_ml/std.npy"
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
            self.joint_num = 21
        else:
            raise NotImplementedError

        self.joints = []
        train_entity = int(len(dataset)*train_ratio)
        if test:
            idx_range = range(len(dataset))
        elif train:
            idx_range = range(train_entity)
        else:
            idx_range = range(train_entity, len(dataset))
        a = 0
        for db_idx in tqdm(idx_range):
            a += 1
            # if a > 100: break ###########################################################################
            data = dataset[db_idx]
            motion_length = data['motion_length'].item()

            motion = data['motion'].numpy()[:motion_length]
            motion = motion * self.std + self.mean
            joint = recover_from_ric(torch.Tensor(motion), self.joint_num).cpu().numpy()
                
            self.joints.append(joint)

        self.joints = np.concatenate(self.joints, axis=0)


    def __len__(self):
        return len(self.joints)
    
    def __getitem__(self, idx):
        joint = self.joints[idx]
        stand_distance = self.stickman.jump_stand(joint)
        track, norm_joint = self.stickman(joint, return_array=True, point_len=64)
        # print(np.linalg.norm(norm_joint[16] - norm_joint[17])) ensure

        return {'track': track.astype(np.float32), # [6, 64, 3]
                'stand_distance': stand_distance, # [22, 3]
                'norm_pose': norm_joint.astype(np.float32)} # [22, 3]


    

if __name__ == "__main__":
    seed = 12
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # cfg = Config.fromfile("configs/remodiffuse/remodiffuse_kit.py")
    cfg = Config.fromfile("configs/remodiffuse/remodiffuse_t2m.py")
    joint_num = cfg.norm_pose_dim // 3
    train_dataset = StickmanDataset(cfg, train=True, train_ratio=0.8)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)
    a = 0
    distances = []
    poses = []
    for data in tqdm(train_loader):
        a += 1
        stand_distance = data['stand_distance']
        track = data['track']
        norm_pose = data['norm_pose']
        poses = norm_pose.cpu().numpy()
        track = track.cpu().numpy()
        from eval_with_eye import eval_vis
        eval_vis(pose=poses[:8], track=track[:8], joints_num=joint_num)
        # break
    ## show distance with stand pose
    # for data in tqdm(train_loader):
    #     a += 1
    #     stand_distance = data['stand_distance']
    #     track = data['track']
    #     norm_pose = data['norm_pose']
    #     for i in range(len(stand_distance)):
    #         distances.append(stand_distance[i].cpu().item())
    #         poses.append(norm_pose[i].cpu().numpy())
    #     if a > 10: break
    # from eval_with_eye import eval_vis
    # sorted_lists = sorted(zip(distances, poses), key=lambda x: x[0])
    # _distances, _poses = zip(*sorted_lists)
    # eval_vis(pose=_poses[::400], joints_num=21, title=[str(round(d, 2)) for d in _distances[::400]])
    
