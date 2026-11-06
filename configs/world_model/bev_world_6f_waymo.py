"""Final Waymo World: six future frames with frozen 64×2048 tokenizer."""

_base_ = ['./bev_world_6f.py']

custom_imports = dict(
    imports=[
        'projects',
        'projects.models.tokenizer.occ_lidar_vae'
    ],
    allow_failed_imports=False)

# Model
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
range_resolution = (64, 2048)
fov_up = 2.274
fov_down = -17.734
min_depth = 1.45
max_depth = 80.0
futu_number = 6
memo_number = 4
eval_frame_number = futu_number + 1

model = dict(
    type='BEVWorld',
    decode_chunk_size=3,
    preserve_full_pred_points=True,
    use_projected_gt=True,
    vae_pretrained=None,
    world_pretrained=None,
    vae=dict(
        range_resolution=range_resolution,
        fov_up=fov_up,
        fov_down=fov_down,
        preserve_ray_geometry=True))

# Data
dataset_type = 'WaymoWorldDataset'
data_root = 'data/waymo'
train_ann_file = 'waymo_info/train_2hz.pkl'
val_ann_file = 'waymo_info/val_2hz.pkl'
train_lidar_root = 'points/train'
val_lidar_root = 'points/val'
train_extrinsics_file = 'waymo_info/train_conversion_manifest.json'
val_extrinsics_file = 'waymo_info/val_conversion_manifest.json'
train_latent_root = 'data/waymo/latents/train'
val_latent_root = 'data/waymo/latents/val'
occ_num_classes = 18
occ3d_classes_names = _base_.occ3d_classes_names
meta_keys = [
    'scene_name', 'token', 'total_token', 'curr2futu_ego',
    'futu_padding_mask', 'sample_index', 'sequence_group_idx',
    'sequence_frame_idx', 'start_of_sequence', 'lidar2ego',
    'ego2lidar', 'total_lidar2ego'
]
waymo_label_mapping = {
    0: 0, 1: 4, 2: 7, 3: 15, 4: 2, 5: 15, 6: 15, 7: 8,
    8: 2, 9: 6, 10: 15, 11: 16, 12: 16, 13: 11, 14: 13, 23: 17
}

range_projection = dict(
    type='LoadStreamRangeImageFromPoints',
    height=range_resolution[0],
    width=range_resolution[1],
    fov_up=fov_up,
    fov_down=fov_down,
    min_depth=min_depth,
    max_depth=max_depth,
    load_dim=5,
    use_dim=5,
    pc_range=point_cloud_range,
    pc_range_coord='ego',
    points_coord='lidar',
    filter_egocar_points=True,
    filter_egocar_coord='ego',
    filter_out_of_fov=True)

train_pipeline = [
    dict(
        type='LoadStreamDataFromFile',
        data_root=train_latent_root,
        input_keys='latent',
        output_keys='bev_latent',
        suffix='npz'),
    dict(**range_projection),
    dict(
        type='MMPack3DDetInputs',
        keys=['bev_latent', 'range_imgs'],
        meta_keys=meta_keys),
]
val_pipeline = [
    dict(
        type='LoadStreamDataFromFile',
        data_root=val_latent_root,
        input_keys='latent',
        output_keys='bev_latent',
        suffix='npz'),
    dict(type='LoadStreamOcc3DFromFile', to_long=False),
    dict(
        type='RemapOccLabels',
        mapping=waymo_label_mapping,
        unknown_label=255),
    dict(**range_projection),
    dict(**range_projection, output_key='fixed_range_imgs'),
    dict(
        type='MMPack3DDetInputs',
        keys=['voxel_semantics', 'bev_latent', 'range_imgs'],
        meta_keys=meta_keys + ['fixed_range_imgs']),
]

shared_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    load_prev_number=0,
    load_futu_number=futu_number,
    load_interval=1,
    use_sequence_group_flag=True,
    sequences_split_num=1,
    serialize_data=False)

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(
        type='SequenceGroupBatchSampler',
        seed=20261010,
        shuffle=True,
        drop_last=False),
    dataset=dict(
        **shared_dataset,
        modality=dict(use_camera=False, use_lidar=True, use_occ=False),
        ann_file=train_ann_file,
        lidar_root=train_lidar_root,
        lidar_extrinsics_file=train_extrinsics_file,
        pipeline=train_pipeline,
        test_mode=False))
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='SequenceGroupDistributedSampler', shuffle=False),
    dataset=dict(
        **shared_dataset,
        modality=dict(use_camera=False, use_lidar=True, use_occ=True),
        ann_file=val_ann_file,
        lidar_root=val_lidar_root,
        lidar_extrinsics_file=val_extrinsics_file,
        pipeline=val_pipeline,
        test_mode=True))

test_evaluator = [
    dict(
        type='Occ4DMetric',
        eval_frame_number=eval_frame_number,
        warmup_frame_number=memo_number,
        frame_interval=0.5,
        num_classes=occ_num_classes,
        empty_cls=17,
        ignore_index=255,
        label2cat=occ3d_classes_names,
        prefix='Fore_Occ',
        vis_interval=0),
    dict(
        type='LiDAR4DMetric',
        eval_frame_number=eval_frame_number,
        warmup_frame_number=memo_number,
        frame_interval=0.5,
        max_pred_points=None,
        prefix='Fore_LiDAR',
        vis_interval=0),
]

# Optimization and runtime
num_gpus = 8
batch_size = 1
max_epochs = 30
base_lr = 1e-5
min_lr = 1e-7
warmup_iters = 100
total_iters = 7500
train_cfg = dict(
    type='SequenceIterBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=999)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    clip_grad=dict(max_norm=15, norm_type=2))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=warmup_iters),
    dict(
        type='CosineAnnealingLR',
        T_max=total_iters - warmup_iters,
        eta_min=min_lr,
        by_epoch=False,
        begin=warmup_iters,
        end=total_iters),
]
auto_scale_lr = dict(enable=False, base_batch_size=num_gpus * batch_size)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=30))
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        decay=0.999,
        init_updates=0,
        interval=1,
        max_keep_ckpts=30,
        resume=None),
]
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='gloo'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
randomness = dict(seed=20261010, deterministic=False)
load_from = 'ckpts/bev_world_6f.pth'
resume = False
work_dir = 'work_dirs/bev_world_6f_waymo/runs/source_finetune'


# ---------------------------- Zero-shot evaluation on the Waymo validation set --------------------- #
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric        | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | future_avg |
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | mIoU          | 0.7851 | 0.5481 | 0.4848 | 0.4406 | 0.4076 | 0.3759 | 0.3470 | 0.4340     |
# | Non-empty IoU | 0.8205 | 0.6990 | 0.6538 | 0.6158 | 0.5822 | 0.5480 | 0.5131 | 0.6020     |
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric           | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | future_avg |
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Chamfer_Distance | 0.2034 | 0.6066 | 0.9139 | 1.1616 | 1.3848 | 1.4123 | 1.7729 | 1.2087     |
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+

# ----------------------------Fine-tuning results on the Waymo validation set --------------------- #
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric        | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | future_avg |
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | mIoU          | 0.7851 | 0.5979 | 0.5375 | 0.4917 | 0.4549 | 0.4180 | 0.3875 | 0.4813     |
# | Non-empty IoU | 0.8205 | 0.7170 | 0.6783 | 0.6438 | 0.6113 | 0.5772 | 0.5427 | 0.6284     |
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric           | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | future_avg |
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Chamfer_Distance | 0.2034 | 0.3725 | 0.5087 | 0.6515 | 0.8354 | 1.0192 | 1.2847 | 0.7787     |
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
