import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict
import random
from typing import Any, List, Optional, Union

import mmcv
import copy
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from .base_dataset import BaseMotionDataset
from .builder import DATASETS
from .utils import random_select_stickman
from stickman.utils import StickLocus, StickMan
from mogen.utils.plot_utils import recover_from_ric

@DATASETS.register_module()
class Stickmant2mDataset(BaseMotionDataset):
    """TextMotion dataset.
    
    Args:
        text_dir (str): Path to the directory containing the text files.
    """
    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: Optional[Union[str, None]] = None,
                 fixed_length: Optional[Union[int, None]] = None,
                 crop_size: Optional[Union[int, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 motion_dir: Optional[Union[str, None]] = None,
                 text_dir: Optional[Union[str, None]] = None,
                 token_dir: Optional[Union[str, None]] = None,
                 clip_feat_dir: Optional[Union[str, None]] = None,
                 eval_cfg: Optional[Union[dict, None]] = None,
                 fine_mode: Optional[bool] = False,
                 test_mode: Optional[bool] = False):
        self.text_dir = os.path.join(data_prefix, 'datasets', dataset_name, text_dir)
        if token_dir is not None:
            self.token_dir = os.path.join(data_prefix, 'datasets', dataset_name, token_dir)
        else:
            self.token_dir = None
        if clip_feat_dir is not None:
            self.clip_feat_dir = os.path.join(data_prefix, 'datasets', dataset_name, clip_feat_dir)
        else:
            self.clip_feat_dir = None
        self.fine_mode = fine_mode
        # self.stickman = StickMan(part_scale=0,)
        self.sticklocus = StickLocus(dataset_name=dataset_name)
        self.stickman = StickMan(dataset_name=dataset_name)
        mean_path = os.path.join(data_prefix, 'datasets', dataset_name, 'mean.npy')
        std_path = os.path.join(data_prefix, 'datasets', dataset_name, 'std.npy')
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)      
        if dataset_name == 'human_ml3d':
            self.joint_num = 22
        elif dataset_name == 'kit_ml':  
            self.joint_num = 21
        else:
            raise NotImplementedError
        super(Stickmant2mDataset, self).__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            dataset_name=dataset_name,
            fixed_length=fixed_length,
            crop_size=crop_size,
            ann_file=ann_file,
            motion_dir=motion_dir,
            eval_cfg=eval_cfg,
            test_mode=test_mode)
        
    def load_anno(self, name):
        results = super().load_anno(name)
        text_path = os.path.join(self.text_dir, name + '.txt')
        text_data = []
        for line in open(text_path, 'r'):
            text_data.append(line.strip())
        results['text'] = text_data
        if self.token_dir is not None:
            token_path = os.path.join(self.token_dir, name + '.txt')
            token_data = []
            for line in open(token_path, 'r'):
                token_data.append(line.strip())
            results['token'] = token_data
        if self.clip_feat_dir is not None:
            clip_feat_path = os.path.join(self.clip_feat_dir, name + '.npy')
            clip_feat = torch.from_numpy(np.load(clip_feat_path))
            results['clip_feat'] = clip_feat
        return results
    


    def prepare_data(self, idx: int):
        """"Prepare raw data for the f'{idx'}-th data."""
        results = copy.deepcopy(self.data_infos[idx])
        text_list = results['text']
        sub_idx = np.random.randint(0, len(text_list))
        if self.fine_mode:
            results['text'] = json.loads(text_list[sub_idx])
        else:
            results['text'] = text_list[sub_idx]
        if 'clip_feat' in results.keys():
            results['clip_feat'] = results['clip_feat'][sub_idx]
        if 'token' in results.keys():
            results['token'] = results['token'][sub_idx]
        results['dataset_name'] = self.dataset_name
        results['text_idx'] = sub_idx
        results['sample_idx'] = int(idx)
        # stickman
        length = min(len(results['motion']), self.crop_size)
        if self.test_mode:
            results['specified_idx'] = [int(p*length) for p in [0.125,0.5,0.875]]
        else:
            results['specified_idx'] = random_select_stickman(length=length)
        ori_joints = recover_from_ric(torch.Tensor(results['motion']), self.joint_num).cpu().numpy()
        # locus = self.sticklocus(ori_joints)
        locus = ori_joints.copy()[:, 0, [0,2]] #*(-2) # [t, 2]
        joints = ori_joints[results['specified_idx'], :, :]
        tracks = []
        norm_joints = []
        for i in range(len(joints)):
            track, norm_joint = self.stickman(joints[i], return_array=True, point_len=64)
            tracks.append(track)
            norm_joints.append(norm_joint)
        tracks = np.array(tracks)
        results['stickman_tracks'] = tracks.astype(np.float32)
        results['norm_joints'] = np.array(norm_joints).astype(np.float32)
        results['locus'] = locus.astype(np.float32)
        return self.pipeline(results)
