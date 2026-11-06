_base_ = ['../_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects'],
    allow_failed_imports=False)

# -------------- Model configuration -----------------
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size = [0.4, 0.4, 0.4]
range_resolution = (32, 1024)
occ_num_classes = 18
latent_channels = 16
stem_channels = 64
scene_feature_channels = 64
base_channel = 64

occ_range_latent_root = 'data/nus_occ_lidar_latent'
occ_range_target_root = 'data/nus_tokenizer_lidar_range'
occ_range_vae_pretrained = 'work_dirs/occ_lidar_vae/epoch_40.pth'

prev_number = 0
futu_number = 2
memo_number = 4
img_size = (200, 200)
vae_stride = 4
vae_embed_dim = 16
patch_size = 1
encoder_embed_dim = 512
encoder_depth = 8
encoder_num_heads = 8
flow_depth = 4
mlp_ratio = 4.0
proj_dropout = 0.1
attn_dropout = 0.1
flow_num_train_timesteps = 1000
flow_num_steps = 4
flow_shift = 2.0
flow_inference_schedule = 'linear_quadratic'
flow_inference_lq_threshold = 0.03
flow_inference_lq_linear_steps = 1
rollout_future_number = None
rollout_stride = None
eval_future_number = futu_number
future_padding_loss_weight = 0.1
range_lidar_loss_weight = 0.05
range_lidar_loss_frame_indices = [0, 1]

model = dict(
    type='BEVWorld',
    data_preprocessor=dict(type='MMDataPreprocessor'),
    prev_number=prev_number,
    futu_number=futu_number,
    memo_number=memo_number,
    img_size=img_size,
    vae_stride=vae_stride,
    encoder_embed_dim=encoder_embed_dim,
    encoder_depth=encoder_depth,
    encoder_num_heads=encoder_num_heads,
    flow_depth=flow_depth,
    mlp_ratio=mlp_ratio,
    vae_embed_dim=vae_embed_dim,
    patch_size=patch_size,
    proj_dropout=proj_dropout,
    attn_dropout=attn_dropout,
    flow_num_train_timesteps=flow_num_train_timesteps,
    flow_num_steps=flow_num_steps,
    flow_shift=flow_shift,
    flow_inference_schedule=flow_inference_schedule,
    flow_inference_lq_threshold=flow_inference_lq_threshold,
    flow_inference_lq_linear_steps=flow_inference_lq_linear_steps,
    rollout_future_number=rollout_future_number,
    rollout_stride=rollout_stride,
    future_padding_loss_weight=future_padding_loss_weight,
    range_lidar_loss_weight=range_lidar_loss_weight,
    range_lidar_loss_frame_indices=range_lidar_loss_frame_indices,
    vae_pretrained=occ_range_vae_pretrained,
    world_pretrained=None,
    use_bev_layout=False,
    vae=dict(
        type='OccLiDARVAE',
        data_preprocessor=dict(type='MMDataPreprocessor'),
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        range_resolution=range_resolution,
        occ_num_classes=occ_num_classes,
        free_class=17,
        occ_class_embeds_dim=16,
        occ_stem_channels=stem_channels,
        range_voxel_channels=2,
        range_stem_channels=stem_channels,
        scene_feature_channels=scene_feature_channels,
        latent_channels=latent_channels,
        range_sample_depth_bins=192,
        ray_depth_sampling='log',
        min_depth=1.45,
        max_depth=80.0,
        valid_pixel_loss_weight=1.0,
        empty_pixel_loss_weight=0.1,
        range_depth_loss_weight=2.0,
        range_edge_loss_weight=2.0,
        range_edge_threshold=0.08,
        occ_loss_weight=1.0,
        range_loss_weight=1.0,
        kl_loss_weight=1e-6,
        free_logit_bias=2.0,
        occupancy_prior_logit=0.833213344056216,
        lidar_volume_channels=16,
        lidar_ray_channels=4,
        occ_ray_condition_channels=2,
        ray_hidden_channels=8,
        occ_support_log_weight=0.25,
        range_shared_grad_scale=0.25,
        save_latent_root=None,
        occ_focal_loss=dict(type='CustomFocalLoss', loss_weight=10.0),
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
            attn_level=(2,))))

# ----------------- Dataset configuration -----------------
dataset_type = 'NuScenesWorldDataset'
data_root = 'data/nuscenes/'
train_ann_file = 'data/nus_infos/nuscenes_infos_train.pkl'
val_ann_file = 'data/nus_infos/nuscenes_infos_val.pkl'

occ3d_classes_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

meta_keys = [
    'scene_name',
    'token',
    'total_token',
    'curr2futu_ego',
    'futu_padding_mask',
    'sample_index',
    'sequence_group_idx',
    'sequence_frame_idx',
    'start_of_sequence',
    'lidar2ego',
    'ego2lidar',
    'total_lidar2ego',
]

train_pipeline = [
    dict(
        type='LoadStreamDataFromFile',
        data_root=occ_range_latent_root,
        input_keys='latent',
        output_keys='bev_latent',
        suffix='npz'),
    dict(
        type='LoadStreamDataFromFile',
        data_root=occ_range_target_root,
        input_keys='range_imgs',
        output_keys='range_imgs',
        suffix='npz'),
    dict(
        type='MMPack3DDetInputs',
        keys=['bev_latent', 'range_imgs'],
        meta_keys=meta_keys),
]

val_pipeline = [
    dict(
        type='LoadStreamDataFromFile',
        data_root=occ_range_latent_root,
        input_keys='latent',
        output_keys='bev_latent',
        suffix='npz'),
    dict(type='LoadStreamOcc3DFromFile', to_long=False),
    dict(
        type='LoadStreamDataFromFile',
        data_root=occ_range_target_root,
        input_keys='range_imgs',
        output_keys='range_imgs',
        suffix='npz'),
    dict(
        type='MMPack3DDetInputs',
        keys=['voxel_semantics', 'bev_latent', 'range_imgs'],
        meta_keys=meta_keys),
]

sequence_group_cfg = dict(
    use_sequence_group_flag=True,
    sequences_split_num=2)

shared_params = dict(
    type=dataset_type,
    data_root=data_root,
    modality=dict(use_camera=False, use_lidar=True, use_occ=True),
    load_prev_number=prev_number)

# ---------------- Training configuration --------------------------
num_gpus = 8
batch_size = 1
reference_batch_size = num_gpus * 2

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(
        type='SequenceGroupBatchSampler',
        shuffle=True,
        drop_last=False),
    dataset=dict(
        **shared_params,
        **sequence_group_cfg,
        load_futu_number=futu_number,
        pipeline=train_pipeline,
        ann_file=train_ann_file,
        test_mode=False))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='SequenceGroupDistributedSampler', shuffle=False),
    dataset=dict(
        **shared_params,
        use_sequence_group_flag=True,
        sequences_split_num=1,
        load_futu_number=eval_future_number,
        pipeline=val_pipeline,
        ann_file=val_ann_file,
        test_mode=True))

test_dataloader = val_dataloader

max_epochs = 72
val_epoch_interval = 2
save_epoch_interval = 2
max_keep_ckpts = 5
base_lr = 1e-3
lr = base_lr

train_cfg = dict(
    type='SequenceIterBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_epoch_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=15, norm_type=2))
auto_scale_lr = dict(enable=True, base_batch_size=reference_batch_size)

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
        milestones=[40, 56],
        gamma=0.1),
]

val_evaluator = [
    dict(
        type='Occ4DMetric',
        eval_frame_number=eval_future_number + 1,
        warmup_frame_number=memo_number,
        num_classes=18,
        empty_cls=17,
        ignore_index=255,
        label2cat=occ3d_classes_names,
        prefix='Fore_Occ',
        vis_interval=10),
    dict(
        type='LiDAR4DMetric',
        eval_frame_number=eval_future_number + 1,
        warmup_frame_number=memo_number,
        prefix='Fore_LiDAR',
        vis_interval=10),
]
test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_interval,
        max_keep_ckpts=max_keep_ckpts),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

ema_save_interval = save_epoch_interval
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        interval=ema_save_interval,
        max_keep_ckpts=max_keep_ckpts)
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=20260710, deterministic=False)



# --------------------------- Results on the validation set ---------------------------
# +---------------+--------+--------+--------+------------+
# | Metric        | 0s     | 0.5s   | 1s     | future_avg |
# +---------------+--------+--------+--------+------------+
# | mIoU          | 0.8444 | 0.6100 | 0.5424 | 0.5762     |
# | Non-empty IoU | 0.7701 | 0.6493 | 0.6202 | 0.6348     |
# +---------------+--------+--------+--------+------------+
# +------------------+--------+--------+--------+------------+
# | Metric           | 0s     | 0.5s   | 1s     | future_avg |
# +------------------+--------+--------+--------+------------+
# | Chamfer_Distance | 0.1441 | 0.2889 | 0.3963 | 0.3426     |
# +------------------+--------+--------+--------+------------+