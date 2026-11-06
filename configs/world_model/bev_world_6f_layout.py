"""Fine-tune the six-frame World Model with BEVLayout conditioning."""

_base_ = './bev_world_6f.py'

stage_6f_checkpoint = 'work_dirs/bev_world_6f/epoch_20_ema.pth'

model = dict(
    world_pretrained=stage_6f_checkpoint,
    use_bev_layout=True,
    bev_layout_in_channels=18)

bev_layout_root = 'data/bev_layout'
occ_range_latent_root = 'data/nus_occ_lidar_latent'
occ_range_target_root = 'data/nus_tokenizer_lidar_range'

meta_keys = _base_.meta_keys

train_pipeline = [
    dict(
        type='LoadStreamDataFromFile',
        data_root=occ_range_latent_root,
        input_keys='latent', output_keys='bev_latent', suffix='npz'),
    dict(
        type='LoadStreamDataFromFile',
        data_root=bev_layout_root,
        input_keys='latent', output_keys='bev_layout', suffix='npy'),
    dict(
        type='LoadStreamDataFromFile',
        data_root=occ_range_target_root,
        input_keys='range_imgs', output_keys='range_imgs', suffix='npz'),
    dict(
        type='MMPack3DDetInputs',
        keys=['bev_latent', 'bev_layout', 'range_imgs'],
        meta_keys=meta_keys),
]

val_pipeline = [
    dict(
        type='LoadStreamDataFromFile',
        data_root=occ_range_latent_root,
        input_keys='latent', output_keys='bev_latent', suffix='npz'),
    dict(
        type='LoadStreamDataFromFile',
        data_root=bev_layout_root,
        input_keys='latent', output_keys='bev_layout', suffix='npy'),
    dict(type='LoadStreamOcc3DFromFile', to_long=False),
    dict(
        type='LoadStreamDataFromFile',
        data_root=occ_range_target_root,
        input_keys='range_imgs', output_keys='range_imgs', suffix='npz'),
    dict(
        type='MMPack3DDetInputs',
        keys=['voxel_semantics', 'bev_latent', 'bev_layout', 'range_imgs'],
        meta_keys=meta_keys),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader

max_epochs = 8
base_lr = 1e-5
lr = base_lr

train_cfg = dict(
    type='SequenceIterBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=_base_.val_epoch_interval)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            bev_layout_embed=dict(lr_mult=10.0),
            bev_layout_adapter=dict(lr_mult=10.0))),
    clip_grad=dict(max_norm=15, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
        milestones=[4, 6], gamma=0.1),
]

# -------------------------- Results of the BEVLayout-conditioned six-frame World Model --------------------------
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric        | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | future_avg |
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | mIoU          | 0.8449 | 0.6309 | 0.5950 | 0.5694 | 0.5478 | 0.5293 | 0.5116 | 0.5640     |
# | Non-empty IoU | 0.7706 | 0.6490 | 0.6252 | 0.6056 | 0.5876 | 0.5700 | 0.5519 | 0.5982     |
# +---------------+--------+--------+--------+--------+--------+--------+--------+------------+
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric           | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | future_avg |
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Chamfer_Distance | 0.1451 | 0.3060 | 0.3888 | 0.4776 | 0.5792 | 0.6434 | 0.7781 | 0.5288     |
# +------------------+--------+--------+--------+--------+--------+--------+--------+------------+
# +-------------------------+-------------------------+
# | Metric                  | pooled_future (0.5s–3s) |
# +-------------------------+-------------------------+
# | Point_Feature_FID       | 4.68303963              |
# | Point_Squared_MMD       | 0.04071129              |
# | BEV_Feature_JSDx10^2    | 2.93595009              |
# | BEV_Feature_MMDx10^4    | 0.76809439              |
# | Future_PointCloud_Count | 27114.00000000          |
# | Valid_Sequence_Count    | 4519.00000000           |
# +-------------------------+-------------------------+