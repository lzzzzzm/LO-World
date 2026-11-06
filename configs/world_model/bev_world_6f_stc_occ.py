"""Fine-tune and evaluate 6f BEVWorld on STCOcc-conditioned latents."""

_base_ = './bev_world_6f.py'

stc_latent_root = 'data/stc_occ_lidar_latent'
stc_train_latent_root = 'data/stc_occ_lidar_latent_train'
occ_range_target_root = 'data/nus_tokenizer_lidar_range'

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

stc_test_pipeline = [
    dict(
        type='LoadStreamDataFromFile',
        data_root=stc_latent_root,
        input_keys='latent',
        output_keys='bev_latent',
        suffix='npz'),
    # STCOcc latent is the model input. Without data_root, this loader follows
    # each sample's official Occ3D occ_path for the evaluation target.
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

stc_train_pipeline = [
    dict(
        type='LoadStreamDataFromFile',
        data_root=stc_train_latent_root,
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

train_dataloader = dict(
    dataset=dict(pipeline=stc_train_pipeline))
val_dataloader = dict(
    dataset=dict(pipeline=stc_test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=stc_test_pipeline))

# Restore the complete six-frame checkpoint through MMEngine. Disable the
# inherited two-frame/world and standalone VAE initializers so they cannot
# overwrite parts of the requested starting checkpoint.
model = dict(
    world_pretrained=None,
    vae_pretrained=None)
load_from = 'ckpts/bev_world_6f.pth'
resume = False

# Low-rate domain adaptation; validation remains the existing 4519-sample
# sequence-warmup protocol and is never included in the training dataloader.
max_epochs = 8
val_epoch_interval = 2
save_epoch_interval = 2
max_keep_ckpts = 5
base_lr = 1e-5
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
        milestones=[4, 6],
        gamma=0.1),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_interval,
        max_keep_ckpts=max_keep_ckpts))

randomness = dict(seed=20261006, deterministic=False)

# ------------------------ --- Results on the validation set ---------------------------
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric        | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | future_avg |
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | mIoU          | 0.2539 | 0.2432 | 0.2301 | 0.2177 | 0.2057 | 0.1952 | 0.1846 | 0.2128     |
# | Non-empty IoU | 0.3343 | 0.3346 | 0.3297 | 0.3239 | 0.3177 | 0.3112 | 0.3032 | 0.3200     |
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric           | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | future_avg |
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Chamfer_Distance | 0.2378 | 0.6281 | 0.9241 | 1.2354 | 1.5511 | 1.6991 | 2.0556 | 1.3489     |
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+