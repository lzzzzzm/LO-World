"""Joint tokenizer training/evaluation configuration shared by dataset variants."""

_base_ = ['../_base_/default_runtime.py']

custom_imports = dict(imports=['projects'], allow_failed_imports=False)

# Model
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size = [0.4, 0.4, 0.4]
range_resolution = (32, 1024)
occ_num_classes = 18
latent_channels = 16
stem_channels = 64
scene_feature_channels = 64
base_channel = 64

model = dict(
    type='OccLiDARVAE',
    data_preprocessor=dict(type='MMDataPreprocessor'),

    # Scene and ray geometry
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
    range_resolution=range_resolution,
    min_depth=1.45,
    max_depth=80.0,
    range_sample_depth_bins=192,
    ray_depth_sampling='log',

    # OCC branch
    occ_num_classes=occ_num_classes,
    free_class=17,
    occ_class_embeds_dim=16,
    occ_stem_channels=stem_channels,
    free_logit_bias=2.0,
    occupancy_prior_logit=0.833213344056216,

    # LiDAR branch
    range_voxel_channels=2,
    range_stem_channels=stem_channels,
    lidar_volume_channels=16,
    lidar_ray_channels=4,
    occ_ray_condition_channels=2,
    ray_hidden_channels=8,
    occ_support_log_weight=0.25,
    range_shared_grad_scale=0.25,

    # Shared latent representation
    scene_feature_channels=scene_feature_channels,
    latent_channels=latent_channels,

    # Training objectives
    occ_focal_loss=dict(type='CustomFocalLoss', loss_weight=10.0),
    occ_loss_weight=1.0,
    range_loss_weight=1.0,
    kl_loss_weight=1e-6,
    valid_pixel_loss_weight=1.0,
    empty_pixel_loss_weight=0.1,
    range_depth_loss_weight=2.0,
    range_edge_loss_weight=2.0,
    range_edge_threshold=0.08,

    # Runtime output
    save_latent_root=None,

    # Shared latent backbone
    shared_encoder=dict(
        type='Encoder2D',
        in_channels=stem_channels * 2,
        ch=base_channel,
        out_channels=latent_channels * 2,
        ch_mult=(1, 2, 4),
        num_res_blocks=(2, 2, 2),
        attn_level=(2,),
        downsample_level=(0, 1)),
    shared_decoder=dict(
        type='Decoder2D',
        in_channels=latent_channels,
        ch=base_channel,
        out_channels=scene_feature_channels,
        dual_out_channels=None,
        ch_mult=(1, 2, 4),
        num_res_blocks=(2, 2, 4),
        upsample_level=(1, 2),
        attn_level=(2,)))

# Data
dataset_type = 'NuScenesWorldDataset'
data_root = 'data/nuscenes/'
train_ann_file = 'data/nus_infos/nuscenes_infos_train.pkl'
val_ann_file = 'data/nus_infos/nuscenes_infos_val.pkl'
tokenizer_range_root = 'data/nus_tokenizer_lidar_range'
meta_keys = [
    'scene_name', 'token', 'total_token', 'sample_idx',
    'ego2lidar', 'lidar2ego'
]

occ3d_classes_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

train_pipeline = [
    dict(type='LoadOcc3DFromFile', to_long=False),
    dict(
        type='LoadDataFromFile',
        data_root=tokenizer_range_root,
        input_keys='range_imgs',
        output_keys='range_imgs',
        suffix='npz'),
    dict(
        type='MMPack3DDetInputs',
        keys=['voxel_semantics', 'range_imgs'],
        meta_keys=meta_keys),
]
val_pipeline = [
    dict(type='LoadOcc3DFromFile', to_long=False),
    dict(
        type='LoadDataFromFile',
        data_root=tokenizer_range_root,
        input_keys='range_imgs',
        output_keys='range_imgs',
        suffix='npz'),
    dict(
        type='MMPack3DDetInputs',
        keys=['voxel_semantics', 'range_imgs'],
        meta_keys=meta_keys),
]

shared_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    modality=dict(use_camera=False, use_lidar=True, use_occ=True))

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        **shared_dataset,
        pipeline=train_pipeline,
        ann_file=train_ann_file,
        test_mode=False))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **shared_dataset,
        pipeline=val_pipeline,
        ann_file=val_ann_file,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='Occ3DMetric',
        num_classes=occ_num_classes,
        empty_cls=17,
        ignore_index=255,
        label2cat=occ3d_classes_names,
        prefix='Rec_OccRange_Occ',
        vis_interval=10),
    dict(
        type='LiDAR3DMetric',
        prefix='Rec_OccRange_Range',
        vis_interval=10),
]
test_evaluator = val_evaluator

# Optimization and runtime
num_gpus = 8
batch_size = 2
max_epochs = 40
base_lr = 1e-4
min_lr = 1e-6
warmup_iters = 500

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))
auto_scale_lr = dict(enable=True, base_batch_size=num_gpus * batch_size)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-2,
        by_epoch=False,
        begin=0,
        end=warmup_iters),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=min_lr,
        by_epoch=True,
        begin=0,
        end=max_epochs),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=12))
randomness = dict(seed=437180218, diff_rank_seed=False)
load_from = None
resume = False

# -------------- Result ----------------
# Occupancy metrics:
# +----------------------+------------+
# | Class                | IoU        |
# +----------------------+------------+
# | others               | 0.7741     |
# | barrier              | 0.9259     |
# | bicycle              | 0.9026     |
# | bus                  | 0.8452     |
# | car                  | 0.8920     |
# | construction_vehicle | 0.7198     |
# | motorcycle           | 0.9225     |
# | pedestrian           | 0.9322     |
# | traffic_cone         | 0.9091     |
# | trailer              | 0.7789     |
# | truck                | 0.8423     |
# | driveable_surface    | 0.9190     |
# | other_flat           | 0.9562     |
# | sidewalk             | 0.8734     |
# | terrain              | 0.8585     |
# | manmade              | 0.7090     |
# | vegetation           | 0.5848     |
# | free                 | 0.9864     |
# | ----------           | ---------- |
# | mIoU                 | 0.8438     |
# | ----------           | ---------- |
# | Non-empty IoU        | 0.7697     |
# +----------------------+------------+
# LiDAR metrics:
# +------------------------+--------+
# | Metric                 | Value  |
# +------------------------+--------+
# | Chamfer_Distance       | 0.1428 |
# | Point_Feature_FID      | 0.7451 |
# | Point_Squared_MMD      | 0.0047 |
# | BEV_Feature_JSDx10^2   | 3.3527 |
# | BEV_Feature_MMDx10^4   | 0.9084 |
# | Intensity_Matched_MAE  | 0.0000 |
# | Intensity_Matched_RMSE | 0.0000 |
# +------------------------+--------+
