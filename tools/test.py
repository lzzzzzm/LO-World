# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys
import torch

real_local_rank = int(os.environ.get('LOCAL_RANK', 0))
cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if cvd is not None:
    visible_devices = [x.strip() for x in cvd.split(',') if x.strip()]
    if real_local_rank < len(visible_devices):
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices[real_local_rank]
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(real_local_rank)
os.environ['LOCAL_RANK'] = '0'

import torch.distributed as dist
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend
from tools.bevworld_generation.runtime import configure_checkpoint_inference

import warnings
# filter DeprecationWarning caused by some third-party libraries
warnings.filterwarnings('ignore', category=DeprecationWarning)
# filter UserWarning caused by some third-party libraries
warnings.filterwarnings('ignore', category=UserWarning)
# filter FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    configure_checkpoint_inference(cfg, args.checkpoint)

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
