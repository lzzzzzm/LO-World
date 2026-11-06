# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import sys

real_local_rank = int(os.environ.get('LOCAL_RANK', 0))
cvd = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if cvd is not None:
    visible_devices = [x.strip() for x in cvd.split(',') if x.strip()]
    if real_local_rank < len(visible_devices):
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices[real_local_rank]
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(real_local_rank)
os.environ['LOCAL_RANK'] = '0'

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch

torch.set_float32_matmul_precision('high')

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend

import warnings
# filter DeprecationWarning caused by some third-party libraries
warnings.filterwarnings('ignore', category=DeprecationWarning)
# filter UserWarning caused by some third-party libraries
warnings.filterwarnings('ignore', category=UserWarning)
# filter FutureWarning caused by some third-party libraries
warnings.filterwarnings('ignore', category=FutureWarning)


def _dataset_uses_sequence_group(dataset_cfg):
    if not isinstance(dataset_cfg, dict):
        return False
    if dataset_cfg.get('use_sequence_group_flag', False):
        return True
    if 'dataset' in dataset_cfg:
        return _dataset_uses_sequence_group(dataset_cfg['dataset'])
    if 'datasets' in dataset_cfg:
        return any(
            _dataset_uses_sequence_group(item)
            for item in dataset_cfg['datasets'])
    return False


def _enable_sequence_group_training(cfg):
    """Switch training to sequence-group iteration when requested by dataset."""
    train_dataloader = cfg.get('train_dataloader', None)
    if train_dataloader is None:
        return
    if not _dataset_uses_sequence_group(train_dataloader.get('dataset', None)):
        return

    drop_last = train_dataloader.pop('drop_last', False)
    train_dataloader.sampler = dict(type='DefaultSampler', shuffle=False)
    train_dataloader.batch_sampler = dict(
        type='SequenceGroupBatchSampler',
        shuffle=True,
        drop_last=drop_last)

    train_cfg = cfg.get('train_cfg', None)
    if train_cfg is None:
        return

    if train_cfg.get('type', None) == 'IterBasedTrainLoop' or \
            train_cfg.get('by_epoch', None) is False:
        return

    if 'type' in train_cfg and train_cfg.type != 'EpochBasedTrainLoop':
        return

    train_cfg.pop('type', None)
    train_cfg.pop('by_epoch', None)
    train_cfg.type = 'SequenceIterBasedTrainLoop'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--sync_bn',
        choices=['none', 'torch', 'mmcv'],
        default='none',
        help='convert all BatchNorm layers in the model to SyncBatchNorm '
        '(SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    _enable_sequence_group_training(cfg)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # convert BatchNorm layers
    if args.sync_bn != 'none':
        cfg.sync_bn = args.sync_bn

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
