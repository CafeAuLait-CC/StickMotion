import my_tools
import argparse
import os
import sys
workspace_path = os.path.abspath(os.path.join(__file__, *['..']*2))
os.chdir(workspace_path)
sys.path.insert(0, workspace_path)
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from mmcv import Config, ConfigDict, DictAction
from mmcv.runner import get_dist_info, init_dist

from mogen import __version__
from mogen.apis.lg_train import LgModel
from mogen.datasets import build_dataset
from mogen.utils import collect_env, get_root_logger

from  lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.fabric.fabric import Fabric
from lightning.pytorch import seed_everything
from lightning.pytorch import loggers
from mmcv.parallel import DataContainer
import hashlib

torch.set_float32_matmul_precision('high')


MODEL_CHOICES = ("diffusion", "flowmatching", "rectified", "meanflow")


def apply_model_override(cfg, model_name):
    choice = model_name.lower()
    if choice not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {MODEL_CHOICES}.")

    if choice == "diffusion":
        cfg.model.type = "MotionDiffusion"
        if hasattr(cfg.model, "pop"):
            cfg.model.pop("flow", None)
        elif "flow" in cfg.model:
            del cfg.model["flow"]
        return

    flow_kind_map = {
        "flowmatching": "linear",
        "rectified": "rectified",
        "meanflow": "meanflow",
    }

    cfg.model.type = "MotionFlowMatching"
    flow_cfg = cfg.model.get("flow")
    if flow_cfg is None:
        flow_cfg = ConfigDict()
        cfg.model.flow = flow_cfg
    elif not isinstance(flow_cfg, ConfigDict):
        flow_cfg = ConfigDict(flow_cfg)
        cfg.model.flow = flow_cfg

    flow_cfg.kind = flow_kind_map[choice]

    path_cfg = flow_cfg.get("path")
    if path_cfg is None:
        path_cfg = ConfigDict()
        flow_cfg.path = path_cfg
    elif not isinstance(path_cfg, ConfigDict):
        path_cfg = ConfigDict(path_cfg)
        flow_cfg.path = path_cfg

    if flow_cfg.kind == "linear":
        path_cfg.type = "linear"
    else:
        path_cfg.type = "rectified"


def parse_args():
    parser = argparse.ArgumentParser(description='test a ckpt')
    parser.add_argument('ckpt', help='checkpoint file path')
    parser.add_argument('gpu', help='-1: 0,1,2,3; -2: 4,5,6,7; else: set to gpu directly')
    parser.add_argument(
        '--config',
        default=None,
        help='Config file to load. Defaults to the config saved next to the checkpoint or the KIT config.',
    )
    parser.add_argument(
        '--model',
        choices=MODEL_CHOICES,
        default=None,
        help='Override the model family defined in the config.',
    )
    args = parser.parse_args()

    return args

def parse_gpu(gpu):
    if gpu == '-1':
        return [0, 1, 2, 3]
    elif gpu == '-2':
        return [4, 5, 6, 7]
    else:
        try:
            return eval(f'[{gpu}]')
        except:
            raise ValueError(f'gpu format error: {gpu}')

def collate_fn(batch):
    keys = batch[0].keys()
    final_batch = {}
    for key in keys:
        if isinstance(batch[0][key], DataContainer):
            # return DataContainer([item.data for item in batch])
            final_batch[key] = [i[key]._data for i in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            # return torch.stack(batch, 0)
            final_batch[key] = torch.stack([item[key] for item in batch], 0)
        else:
            raise NotImplementedError

    return final_batch

def main():
    seed_everything(123, workers=True)
    args = parse_args()
    args.work_dir = Path(args.ckpt).parent
    if args.config is not None:
        config_path = args.config
    else:
        # try to reuse the config dumped alongside the checkpoint
        py_files = sorted(p for p in args.work_dir.glob('*.py') if p.is_file())
        if py_files:
            config_path = str(py_files[0])
        else:
            config_path = 'configs/remodiffuse/remodiffuse_kit.py'

    cfg = Config.fromfile(config_path)
    if args.model is not None:
        apply_model_override(cfg, args.model)
    cfg.data.test.test_mode = True
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    dataset = build_dataset(cfg.data.test)
    # add an attribute for visualization convenience
    if   Fabric().global_rank == 0:
        pid_seed = os.getpid()
    else:
        pid_seed = os.getppid()
    model = LgModel(cfg, dataset, unit=hashlib.md5(str(pid_seed).encode()).hexdigest()[:8])
    test_loader = DataLoader(dataset, batch_size=cfg.data.samples_per_gpu, shuffle=False, num_workers=cfg.data.workers_per_gpu, collate_fn=collate_fn)
    trainer = Trainer(accelerator="gpu",
                      strategy=DDPStrategy(),
                    #   devices=[0],
                      devices=parse_gpu(args.gpu),
                      logger=False,
                    #   max_steps=3,
                      precision='16-mixed',
                      )
    trainer.validate(model, test_loader, ckpt_path=args.ckpt)
    if getattr(model, 'last_aits', None) is not None:
        print(f'\nAverage inference time per sample: {model.last_aits:.6f}s')


if __name__ == '__main__':
    main()

# python tools/lg_test.py logs/kit_ml//last.ckpt -1

# python tools/lg_test.py /mnt/new_disk2/wangtao/StickMotion/logs/kit_ml/init25_fix/last.ckpt 2