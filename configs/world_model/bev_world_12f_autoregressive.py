"""Fine-tune the no-Layout 6f generator for two autoregressive chunks."""
_base_ = './bev_world_6f.py'

stage_6f_checkpoint = 'work_dirs/bev_world_6f/epoch_20_ema.pth'
chunk_future_number = 6
rollout_future_number = 12
train_autoregressive_chunks = 2
memo_number = 4
eval_future_number = rollout_future_number
eval_frame_number = eval_future_number + 1

model = dict(
    # Restore the complete model (including frozen VAE) through load_from.
    world_pretrained=None,
    vae_pretrained='ckpts/occ_lidar_vae.pth',
    futu_number=chunk_future_number,
    use_bev_layout=False,
    rollout_future_number=rollout_future_number,
    rollout_stride=chunk_future_number,
    train_autoregressive_chunks=train_autoregressive_chunks,
    train_autoregressive_loss_weight=1.0,
    # One 6f loss graph per iteration. The far chunk conditions on a
    # detached sampled near chunk; alternate near/far supervision.
    train_autoregressive_chunk_mode='round_robin',
    range_lidar_loss_frame_indices=[4, 5])

train_dataloader = dict(
    batch_size=1,
    dataset=dict(load_futu_number=rollout_future_number))
val_dataloader = dict(dataset=dict(load_futu_number=rollout_future_number))
test_dataloader = val_dataloader

occ3d_classes_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
val_evaluator = [
    dict(type='Occ4DMetric', eval_frame_number=eval_frame_number,
         warmup_frame_number=memo_number, num_classes=18, empty_cls=17,
         ignore_index=255, label2cat=occ3d_classes_names,
         prefix='Fore_Occ', vis_interval=10),
    dict(type='LiDAR4DMetric', eval_frame_number=eval_frame_number,
         warmup_frame_number=memo_number, prefix='Fore_LiDAR',
         vis_interval=10),
]
test_evaluator = val_evaluator

num_gpus = 8
batch_size = 1
max_epochs = 8
val_epoch_interval = 2
save_epoch_interval = 2
max_keep_ckpts = 5
base_lr = 1e-5
lr = base_lr
train_cfg = dict(type='SequenceIterBasedTrainLoop',
                 max_epochs=max_epochs, val_interval=val_epoch_interval)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=15, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False,
         begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
         milestones=[4, 6], gamma=0.1),
]
auto_scale_lr = dict(enable=True, base_batch_size=num_gpus * batch_size)
default_hooks = dict(checkpoint=dict(
    type='CheckpointHook', interval=save_epoch_interval,
    max_keep_ckpts=max_keep_ckpts))
custom_hooks = [
    # EMA is created before Runner loads load_from. Initialize its complete
    # state from the same EMA checkpoint, including its update counter.
    dict(type='MEGVIIEMAHook', resume=stage_6f_checkpoint,
         init_updates=80900, interval=save_epoch_interval,
         max_keep_ckpts=max_keep_ckpts),
]
randomness = dict(seed=20260710, deterministic=False)
load_from = stage_6f_checkpoint
resume = False
work_dir = 'work_dirs/bev_world_12f_autoregressive'

# +---------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric        | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | 3.5s   | 4s     | 4.5s   | 5s     | 5.5s   | 6s     | future_avg |
# +---------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
# | mIoU          | 0.8454 | 0.6002 | 0.5356 | 0.4889 | 0.4515 | 0.4200 | 0.3922 | 0.3596 | 0.3401 | 0.3230 | 0.3070 | 0.2912 | 0.2748 | 0.3987     |
# | Non-empty IoU | 0.7714 | 0.6458 | 0.6172 | 0.5931 | 0.5714 | 0.5502 | 0.5297 | 0.5028 | 0.4865 | 0.4715 | 0.4562 | 0.4399 | 0.4238 | 0.5240     |
# +---------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
# +------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric           | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | 3.5s   | 4s     | 4.5s   | 5s     | 5.5s   | 6s     | future_avg |
# +------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Chamfer_Distance | 0.1458 | 0.3164 | 0.4206 | 0.5440 | 0.6833 | 0.7906 | 0.9740 | 1.2734 | 1.4884 | 1.7019 | 1.9976 | 2.1157 | 2.4280 | 1.2278     |
# +------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
