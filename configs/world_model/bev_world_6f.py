_base_ = './bev_world_2f.py'

futu_number = 6
memo_number = 4
eval_future_number = futu_number
eval_frame_number = eval_future_number + 1
stage_2f_checkpoint = 'work_dirs/bev_world_2f/epoch_72_ema.pth'

model = dict(
    futu_number=futu_number,
    memo_number=memo_number,
    world_pretrained=stage_2f_checkpoint,
    use_bev_layout=False,
    # At 2 Hz, only the final two 6f outputs receive direct LiDAR loss.
    range_lidar_loss_frame_indices=[4, 5])

train_dataloader = dict(
    batch_size=1,
    dataset=dict(load_futu_number=futu_number))

val_dataloader = dict(
    dataset=dict(load_futu_number=eval_future_number))
test_dataloader = val_dataloader

occ3d_classes_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

val_evaluator = [
    dict(
        type='Occ4DMetric',
        eval_frame_number=eval_frame_number,
        warmup_frame_number=memo_number,
        num_classes=18,
        empty_cls=17,
        ignore_index=255,
        label2cat=occ3d_classes_names,
        prefix='Fore_Occ',
        vis_interval=10),
    dict(
        type='LiDAR4DMetric',
        eval_frame_number=eval_frame_number,
        warmup_frame_number=memo_number,
        prefix='Fore_LiDAR',
        detail=True,
        vis_interval=10),
]
test_evaluator = val_evaluator

# Keep the validated 6-frame optimization strategy unchanged.
num_gpus = 8
batch_size = 1
max_epochs = 20
val_epoch_interval = 2
save_epoch_interval = 2
max_keep_ckpts = 5
base_lr = 1e-4
lr = base_lr

train_cfg = dict(
    type='SequenceIterBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_epoch_interval)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=15, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=max_epochs,
        milestones=[10, 14],
        gamma=0.1),
]

auto_scale_lr = dict(enable=True, base_batch_size=num_gpus * batch_size)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_interval,
        max_keep_ckpts=max_keep_ckpts))

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        interval=save_epoch_interval,
        max_keep_ckpts=max_keep_ckpts)
]

randomness = dict(seed=20260710, deterministic=False)
load_from = None
resume = False

# --------------------------- Results on the validation set ---------------------------
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric        | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | future_avg |
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | mIoU          | 0.8449 | 0.6041 | 0.5391 | 0.4921 | 0.4550 | 0.4230 | 0.3943 | 0.4846     |
# | Non-empty IoU | 0.7706 | 0.6474 | 0.6195 | 0.5959 | 0.5746 | 0.5542 | 0.5334 | 0.5875     |
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric           | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | future_avg |
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Chamfer_Distance | 0.1451 | 0.3089 | 0.4153 | 0.5329 | 0.6759 | 0.7962 | 0.9815 | 0.6185     |
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+

# +-------------------------+-------------------------+
# | Metric                  | pooled_future (0.5s–3s) |
# +-------------------------+-------------------------+
# | Point_Feature_FID       | 5.16342080              |
# | Point_Squared_MMD       | 0.04555055              |
# | BEV_Feature_JSDx10^2    | 2.96490490              |
# | BEV_Feature_MMDx10^4    | 0.76141876              |
# | Future_PointCloud_Count | 27114.00000000          |
# | Valid_Sequence_Count    | 4519.00000000           |
# +-------------------------+-------------------------+