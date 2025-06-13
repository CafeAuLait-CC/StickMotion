import my_tools
import argparse
import os
import os.path as osp
import mmcv
import numpy as np
import torch
from mogen.models import build_architecture
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mogen.utils.plot_utils import (
    recover_from_ric,
    plot_3d_motion,
    t2m_kinematic_chain,
    plot_3d_motion_kit,
    kit_kinematic_chain
)
from scipy.ndimage import gaussian_filter


def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)


def plot_t2m(data, result_path, npy_path, caption):
    joint = recover_from_ric(torch.from_numpy(data).float(), 22).numpy()
    joint = motion_temporal_filter(joint, sigma=2.5)
    plot_3d_motion(result_path, t2m_kinematic_chain, joint, title=caption, fps=20)
    if npy_path is not None:
        np.save(npy_path, joint)

def plot_kit(data, result_path, npy_path, caption): 
    joint = recover_from_ric(torch.from_numpy(data).float(), 22).numpy()
    joint = motion_temporal_filter(joint, sigma=2.5)
    plot_3d_motion_kit(result_path, kit_kinematic_chain, joint, title=caption, fps=20)
    if npy_path is not None:
        np.save(npy_path, joint)

def parse_args():
    parser = argparse.ArgumentParser(description='mogen evaluation')
    parser.add_argument('config', help='test config file path') # kit(configs/remodiffuse/remodiffuse_kit.py)
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--text', help='motion description')
    parser.add_argument('--motion_length', type=int, help='expected motion length')
    parser.add_argument('--out', help='output animation file')
    parser.add_argument('--pose_npy', help='output pose sequence file', default=None)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    assert args.motion_length >= 16 and args.motion_length <= 196
    
    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    dataset_name = cfg.data.test.dataset_name
    print("dataset_name",dataset_name)
    if dataset_name == 'human_ml3d':
        #assert dataset_name == "human_ml3d"
        mean_path = "data/datasets/human_ml3d/mean.npy"
        std_path = "data/datasets/human_ml3d/std.npy"
        mean = np.load(mean_path)
        std = np.load(std_path)      
    else:  
        #assert dataset_name == "kit_ml"
        mean_path = "data/datasets/kit_ml/mean.npy"
        std_path = "data/datasets/kit_ml/std.npy"
        mean = np.load(mean_path)
        std = np.load(std_path)
         
    device = args.device
    text = args.text
    motion_length = args.motion_length
    if dataset_name == 'human_ml3d':
        motion = torch.zeros(1, motion_length, 263).to(device)
    else:
        motion = torch.zeros(1, motion_length, 251).to(device)
    motion_mask = torch.ones(1, motion_length).to(device)
    motion_length = torch.Tensor([motion_length]).long().to(device)
    model = model.to(device)
    model.module.others_cuda()
    
    input = {
        'motion': motion,
        'motion_mask': motion_mask,
        'motion_length': motion_length,
        'motion_metas': [{'text': text}],
    }

    all_pred_motion = []
    with torch.no_grad():
        input['inference_kwargs'] = {}
        output_list = []
        output = model(**input)[0]['pred_motion']
        pred_motion = output.cpu().detach().numpy()
        pred_motion = pred_motion * std + mean
    
    if dataset_name == 'human_ml3d':
        plot_t2m(pred_motion, args.out, args.pose_npy, text)
    else:
        plot_kit(pred_motion, args.out, args.pose_npy, text)
    
import matplotlib.pyplot as plt
class VisMy():
    def __init__(self, dataset_name) -> None:    
        self.dataset_name = dataset_name
        self.title_size = 10
        if dataset_name == 'human_ml3d':
            #assert dataset_name == "human_ml3d"
            mean_path = "data/datasets/human_ml3d/mean.npy"
            std_path = "data/datasets/human_ml3d/std.npy"
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)      
            self.joint_num = 22
        else:  
            #assert dataset_name == "kit_ml"
            mean_path = "data/datasets/kit_ml/mean.npy"
            std_path = "data/datasets/kit_ml/std.npy"
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
            self.joint_num = 21

    def vis_seq(self, entity, out_path):
        stickman_tracks = entity['stick_tracks']
        pred_index = entity['pred_index'][-1]
        motion = entity['motion']
        text = entity['text']

        
        motion = motion * self.std + self.mean
        if self.dataset_name == 'human_ml3d':
            plot_t2m(motion, out_path, None, text)
        else:
            plot_kit(motion, out_path, None, text)

    def pose_vis(self, pose, fig, idx, title):
        ax = fig.add_subplot(idx, projection='3d', aspect='equal')
        colors = ['red', 'blue', 'black', 'red', 'blue',
                'yellow', 'yellow', 'darkblue', 'darkblue', 'darkblue',
                'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

        for i, (chain, color) in enumerate(zip(kit_kinematic_chain, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(pose[chain, 0], pose[chain, 1], pose[chain, 2], linewidth=linewidth,
                        color=color)
        for i, (x, y, z) in enumerate(pose):  
            ax.scatter(x, y, z, s=20, color=[float(i)/255 for i in [30, 30, 30]])  
            ax.text(x, y, z, f'{i}', size=14, zorder=1, color='green')  

        ax.set_xlabel('0')  
        ax.set_ylabel('1')  
        ax.set_zlabel('2') 
        ax.set_aspect('equal', adjustable='box')
        ax.view_init(elev=10, azim=10)
        ax.title.set_text(title)
        ax.title.set_fontsize(self.title_size)

    def track_vis(self, track, fig, idx):
        ax = fig.add_subplot(idx)
        for i in range(6):
            ax.plot(track[i, :, 0], track[i, :, 1], linewidth=4, color='black')
        
        ax.axis('equal')
        ax.title.set_text('stickman')
        ax.title.set_fontsize(self.title_size)
    
    def diff_vis(self, track_list, pred_motion_list, gt_motion_list, pred_index, gt_index):
        idx = 230
        fig = plt.figure(figsize=(20, 30))
        for i, track in enumerate(track_list):
            self.track_vis(track, fig, idx+i)
        for j, motion in enumerate(pred_motion_list):
            self.pose_vis(motion, fig, idx+2+j, f'pred index: {pred_index[j]}')
        for k, motion in enumerate(gt_motion_list):
            self.pose_vis(motion, fig, idx+4+k, f'gt_index: {gt_index[k]}')
        plt.savefig('diff.pdf')
        
    
    

if __name__ == '__main__':
    # main()
    vismy = VisMy('kit_ml')
    import pickle
    result = pickle.load(open('res.pkl', 'rb'))
    index = 0
    res = result[index]
    track_list = res['stickman_tracks']
    motion = res['pred_motion']
    pred_index = res['pred_index'][-1]

# export CUDA_VISIBLE_DEVICES=4; gopy tools/visualize.py     configs/remodiffuse/remodiffuse_t2m.py     logs/remodiffuse/remodiffuse_t2m/latest.pth     --text "a person is running quickly"     --motion_length 120     --out "test.gif"     --device cuda