# import my_tools
import argparse
import os
import sys

import cv2
from cv2.cuda import GLOBAL_ATOMICS
from tqdm import tqdm
from multiprocessing import Pool


workspace_path = os.path.abspath(os.path.join(__file__, *[".."] * 2))
os.chdir(workspace_path)
sys.path.insert(0, workspace_path)
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
    kit_kinematic_chain,
)
from scipy.ndimage import gaussian_filter

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_ckpt(model, ckpt_path, map_location="cpu"):
    """Load checkpoint for both .pth and lightning .ckpt files."""
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")
    if ckpt_path.endswith(".ckpt"):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        load_checkpoint(model, ckpt_path, map_location=map_location)


def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)


def plot_t2m(data, result_path, npy_path, caption):
    data = np.squeeze(data)
    joint = recover_from_ric(torch.from_numpy(data).float(), 22).numpy()
    joint = motion_temporal_filter(joint, sigma=2.5)
    plot_3d_motion(result_path, t2m_kinematic_chain, joint, title=caption, fps=20)
    if npy_path is not None:
        np.save(npy_path, joint)


def plot_kit(data, result_path, npy_path, caption):
    data = np.squeeze(data)
    joint = recover_from_ric(torch.from_numpy(data).float(), 21).numpy()

    # 1. Center at root joint
    # root_positions = joint[:, 0:1, :].copy()
    # joint = joint - root_positions

    # 2. Rotate to standing position
    # We'll rotate around X-axis to make Y the vertical axis
    rotation_angle = np.pi / 2  # 90 degrees rotation
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
            [0, np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    joint = np.dot(joint, rotation_matrix.T)

    # 3. Scale to reasonable size
    max_val = np.max(np.abs(joint))
    if max_val > 0:
        joint = joint / max_val * 5  # Scale to ±5 units

    face_camera_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    joint = np.dot(joint, face_camera_rotation)
    # 4. Debug plot
    # debug_plot_first_frame(joint[0])

    # 4. Apply temporal filter and plot
    joint = motion_temporal_filter(joint, sigma=2.5)
    # plot_3d_motion_kit(result_path, kit_kinematic_chain, joint,
    #                   title=caption, fps=20, radius=5)

    # 5. Compute global bounds for fixed coordinate system
    all_positions = joint.reshape(-1, 3)
    min_vals = np.min(all_positions, axis=0)
    max_vals = np.max(all_positions, axis=0)

    # Calculate global range with padding
    ranges = max_vals - min_vals
    max_range = np.max(ranges) * 1.2  # Add 20% padding
    # mid_point = (min_vals + max_vals) / 2

    # Create bounding box
    global_bounds = {
        "x": [min_vals[0] - max_range * 0.1, max_vals[0] + max_range * 0.1],
        "y": [min_vals[1] - max_range * 0.1, max_vals[1] + max_range * 0.1],
        "z": [min_vals[2] - max_range * 0.1, max_vals[2] + max_range * 0.1],
    }

    # 6. Create output directory for frames
    frame_dir = "animation_frames"
    os.makedirs(frame_dir, exist_ok=True)

    # 7. Generate frames
    if os.name == "nt":
        for i in tqdm(range(joint.shape[0]), desc="Generating frames"):
            frame_path = plot_single_frame(
                joint[i], i, frame_dir, global_bounds=global_bounds
            )

    else:
        frame_data = []
        for i in range(joint.shape[0]):
            frame_data.append(
                (joint[i], i, frame_dir, kit_kinematic_chain, global_bounds)
            )
        # Generate frames in parallel with progress bar
        with Pool(processes=os.cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.starmap(plot_single_frame, frame_data),
                    total=joint.shape[0],
                    desc="Generating frames",
                )
            )

    # 8. Compile video
    create_video_from_frames(frame_dir, result_path, fps=20)

    # 9. Clean up
    # shutil.rmtree(frame_dir)

    if npy_path is not None:
        np.save(npy_path, joint)


# def plot_single_frame_wrapper(args):
#     """Wrapper function for parallel processing"""
#     return plot_single_frame(*args)


def debug_plot_first_frame(joints):
    """Plot first frame separately for debugging"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=50)

    # Plot connections
    colors = ["red", "blue", "black", "red", "blue"]
    for i, (chain, color) in enumerate(zip(kit_kinematic_chain, colors)):
        linewidth = 4.0 if i < 5 else 2.0
        x = joints[chain, 0]
        y = joints[chain, 1]
        z = joints[chain, 2]
        ax.plot(x, y, z, linewidth=linewidth, color=color)

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("First Frame Debug")

    # Calculate proper aspect ratio
    max_range = (
        np.array(
            [
                joints[:, 0].max() - joints[:, 0].min(),
                joints[:, 1].max() - joints[:, 1].min(),
                joints[:, 2].max() - joints[:, 2].min(),
            ]
        ).max()
        * 0.5
    )

    mid_x = (joints[:, 0].max() + joints[:, 0].min()) * 0.5
    mid_y = (joints[:, 1].max() + joints[:, 1].min()) * 0.5
    mid_z = (joints[:, 2].max() + joints[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set view to standing position
    ax.view_init(elev=15, azim=45)

    plt.savefig("debug_first_frame.png")
    plt.close()


def plot_single_frame(
    joints, frame_idx, frame_dir, kinematic_tree=kit_kinematic_chain, global_bounds=None
):
    """Plot a single frame and save it to file"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=50)

    # Plot connections
    colors = ["red", "blue", "black", "red", "blue"]
    for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
        linewidth = 4.0 if i < 5 else 2.0
        x = joints[chain, 0]
        y = joints[chain, 1]
        z = joints[chain, 2]
        ax.plot(x, y, z, linewidth=linewidth, color=color)

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Frame {frame_idx}")

    # Set fixed global bounds if provided
    if global_bounds:
        ax.set_xlim(global_bounds["x"])
        ax.set_ylim(global_bounds["y"])
        ax.set_zlim(global_bounds["z"])

        # Add ground plane at actual ground level
        x_min, x_max = global_bounds["x"]
        z_min, z_max = global_bounds["z"]
        ground_y = global_bounds["y"][0]  # Use min Y as ground

        # Create grid for ground plane
        xx, zz = np.meshgrid(
            np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10)
        )
        yy = np.full_like(xx, ground_y)

        # Plot semi-transparent ground plane
        ax.plot_surface(xx, yy, zz, alpha=0.1, color="gray")

        # Add coordinate axes at origin
        axis_length = (
            np.max(
                [
                    x_max - x_min,
                    global_bounds["y"][1] - global_bounds["y"][0],
                    z_max - z_min,
                ]
            )
            * 0.1
        )

        # X-axis (red)
        ax.quiver(0, ground_y, 0, axis_length, 0, 0, color="r", arrow_length_ratio=0.1)
        # Y-axis (green)
        ax.quiver(0, ground_y, 0, 0, axis_length, 0, color="g", arrow_length_ratio=0.1)
        # Z-axis (blue)
        ax.quiver(0, ground_y, 0, 0, 0, axis_length, color="b", arrow_length_ratio=0.1)

    else:
        # Fallback to per-frame scaling
        # Calculate proper aspect ratio
        max_range = (
            np.array(
                [
                    joints[:, 0].max() - joints[:, 0].min(),
                    joints[:, 1].max() - joints[:, 1].min(),
                    joints[:, 2].max() - joints[:, 2].min(),
                ]
            ).max()
            * 0.5
        )

        mid_x = (joints[:, 0].max() + joints[:, 0].min()) * 0.5
        mid_y = (joints[:, 1].max() + joints[:, 1].min()) * 0.5
        mid_z = (joints[:, 2].max() + joints[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set consistent view
    ax.view_init(elev=15, azim=45)

    # Save frame
    frame_path = os.path.join(frame_dir, f"frame_{frame_idx:04d}.png")
    plt.savefig(frame_path)
    plt.close()
    return frame_path


def create_video_from_frames(frame_dir, output_path, fps=20):
    """Compile frames into a video file"""
    images = [img for img in sorted(os.listdir(frame_dir)) if img.endswith(".png")]
    if not images:
        raise RuntimeError(f"No frames found in {frame_dir}")

    # Determine frame size from first image
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, _ = frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Add frames to video
    for image in images:
        img_path = os.path.join(frame_dir, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video saved to {output_path}")


# def plot_kit(data, result_path, npy_path, caption):
#     data = np.squeeze(data)
#     joint = recover_from_ric(torch.from_numpy(data).float(), 21).numpy()
#     # joint = joint - joint[:, :1, :]
#     print("Recovered joint shape", joint.shape)
#     joint = motion_temporal_filter(joint, sigma=2.5)
#     print("Filtered joint shape", joint.shape)
#     plot_3d_motion_kit(result_path, kit_kinematic_chain, joint, title=caption, fps=20)
#     if npy_path is not None:
#         np.save(npy_path, joint)
#


def parse_args():
    parser = argparse.ArgumentParser(description="mogen evaluation")
    parser.add_argument(
        "config", help="test config file path"
    )  # kit(configs/remodiffuse/remodiffuse_kit.py)
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--text", default="", help="motion description")
    parser.add_argument("--motion_length", type=int, help="expected motion length")
    parser.add_argument("--out", help="output animation file")
    parser.add_argument("--pose_npy", help="output pose sequence file", default=None)
    parser.add_argument(
        "--stickman_path", help="path to stickman track npy file", default=None
    )
    parser.add_argument(
        "--specified_idx",
        type=int,
        nargs="+",
        default=None,
        help="indices of frames corresponding to each stickman track",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="device used for testing",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    assert args.motion_length >= 16 and args.motion_length <= 196

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # load_ckpt(model, args.checkpoint, map_location='cpu')
    from collections import OrderedDict

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Strip 'model.' prefix if it exists
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k[len("model.") :]
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)

    if args.device == "cpu":
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()

    dataset_name = cfg.data.test.dataset_name
    print("dataset_name", dataset_name)
    if dataset_name == "human_ml3d":
        # assert dataset_name == "human_ml3d"
        mean_path = "data/datasets/human_ml3d/mean.npy"
        std_path = "data/datasets/human_ml3d/std.npy"
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        # assert dataset_name == "kit_ml"
        mean_path = "data/datasets/kit_ml/mean.npy"
        std_path = "data/datasets/kit_ml/std.npy"
        mean = np.load(mean_path)
        std = np.load(std_path)

    device = args.device
    text = args.text if args.text is not None else ""
    motion_length = args.motion_length
    if dataset_name == "human_ml3d":
        motion = torch.zeros(1, motion_length, 263).to(device)
    else:
        motion = torch.zeros(1, motion_length, 251).to(device)
    motion_mask = torch.ones(1, motion_length).to(device)
    motion_length_tensor = torch.tensor(
        [[motion_length]], dtype=torch.long, device=device
    )
    model = model.to(device)
    model.module.others_cuda()

    if args.stickman_path is not None:
        tracks_np = np.load(args.stickman_path)
        stickman_tracks = torch.from_numpy(tracks_np).float().unsqueeze(0).to(device)
        if args.specified_idx is None:
            raise ValueError("--specified_idx must be provided with --stickman_path")
        specified_idx = (
            torch.tensor(args.specified_idx, dtype=torch.long).unsqueeze(0).to(device)
        )
    else:
        index_num = cfg.model.index_num
        stickman_tracks = torch.zeros(1, index_num, 6, 64, 2).to(device)
        mid = motion_length // 2
        specified_idx = (
            torch.tensor([0, mid, motion_length - 1], dtype=torch.long)
            .unsqueeze(0)
            .to(device)
        )

    input = {
        "motion": motion,
        "motion_mask": motion_mask,
        "motion_length": motion_length_tensor,
        "motion_metas": [{"text": [text]}],
        "text": [text],
        "stickman_tracks": stickman_tracks,
        "specified_idx": specified_idx,
    }

    all_pred_motion = []
    with torch.no_grad():
        input["inference_kwargs"] = {}
        output_list = []
        output = model(**input)[0]["pred_motion"]
        pred_motion = output.cpu().detach().numpy()
        pred_motion = pred_motion * std + mean

    np.save("pred_motion.npy", pred_motion)
    print("shape of output (in pred_motion.npy):", pred_motion.shape)

    if dataset_name == "human_ml3d":
        plot_t2m(pred_motion, args.out, args.pose_npy, text)
    else:
        plot_kit(pred_motion, args.out, args.pose_npy, text)


import matplotlib.pyplot as plt


class VisMy:
    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name
        self.title_size = 10
        if dataset_name == "human_ml3d":
            # assert dataset_name == "human_ml3d"
            mean_path = "data/datasets/human_ml3d/mean.npy"
            std_path = "data/datasets/human_ml3d/std.npy"
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
            self.joint_num = 22
        else:
            # assert dataset_name == "kit_ml"
            mean_path = "data/datasets/kit_ml/mean.npy"
            std_path = "data/datasets/kit_ml/std.npy"
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
            self.joint_num = 21

    def vis_seq(self, entity, out_path):
        stickman_tracks = entity["stick_tracks"]
        pred_index = entity["pred_index"][-1]
        motion = entity["motion"]
        text = entity["text"]

        motion = motion * self.std + self.mean
        if self.dataset_name == "human_ml3d":
            plot_t2m(motion, out_path, None, text)
        else:
            plot_kit(motion, out_path, None, text)

    def pose_vis(self, pose, fig, idx, title):
        ax = fig.add_subplot(idx, projection="3d", aspect="equal")
        colors = [
            "red",
            "blue",
            "black",
            "red",
            "blue",
            "yellow",
            "yellow",
            "darkblue",
            "darkblue",
            "darkblue",
            "darkred",
            "darkred",
            "darkred",
            "darkred",
            "darkred",
        ]

        for i, (chain, color) in enumerate(zip(kit_kinematic_chain, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(
                pose[chain, 0],
                pose[chain, 1],
                pose[chain, 2],
                linewidth=linewidth,
                color=color,
            )
        for i, (x, y, z) in enumerate(pose):
            ax.scatter(x, y, z, s=20, color=[float(i) / 255 for i in [30, 30, 30]])
            ax.text(x, y, z, f"{i}", size=14, zorder=1, color="green")

        ax.set_xlabel("0")
        ax.set_ylabel("1")
        ax.set_zlabel("2")
        ax.set_aspect("equal", adjustable="box")
        ax.view_init(elev=10, azim=10)
        ax.title.set_text(title)
        ax.title.set_fontsize(self.title_size)

    def track_vis(self, track, fig, idx):
        ax = fig.add_subplot(idx)
        for i in range(6):
            ax.plot(track[i, :, 0], track[i, :, 1], linewidth=4, color="black")

        ax.axis("equal")
        ax.title.set_text("stickman")
        ax.title.set_fontsize(self.title_size)

    def diff_vis(
        self, track_list, pred_motion_list, gt_motion_list, pred_index, gt_index
    ):
        idx = 230
        fig = plt.figure(figsize=(20, 30))
        for i, track in enumerate(track_list):
            self.track_vis(track, fig, idx + i)
        for j, motion in enumerate(pred_motion_list):
            self.pose_vis(motion, fig, idx + 2 + j, f"pred index: {pred_index[j]}")
        for k, motion in enumerate(gt_motion_list):
            self.pose_vis(motion, fig, idx + 4 + k, f"gt_index: {gt_index[k]}")
        plt.savefig("diff.pdf")


if __name__ == "__main__":
    main()
