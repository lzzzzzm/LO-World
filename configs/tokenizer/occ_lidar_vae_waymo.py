"""Frozen Waymo tokenizer for 64×2048 latent extraction and evaluation."""

_base_ = ['./occ_lidar_vae.py']

custom_imports = dict(
    imports=['projects'],
    allow_failed_imports=False)

# Model
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
range_resolution = (64, 2048)
fov_up = 2.274
fov_down = -17.734
min_depth = 1.45
max_depth = 80.0
val_latent_root = 'data/waymo/latents/val'

model = dict(
    range_resolution=range_resolution,
    fov_up=fov_up,
    fov_down=fov_down,
    preserve_ray_geometry=True,
    use_projected_gt=True,
    save_latent_root=val_latent_root)

# Data
dataset_type = 'WaymoWorldDataset'
data_root = 'data/waymo'
train_data_root = data_root
train_ann_file = 'waymo_info/train_2hz.pkl'
val_ann_file = 'waymo_info/val_2hz.pkl'
train_lidar_root = 'points/train'
val_lidar_root = 'points/val'
train_extrinsics_file = 'waymo_info/train_conversion_manifest.json'
val_extrinsics_file = 'waymo_info/val_conversion_manifest.json'
meta_keys = [
    'scene_name', 'token', 'sample_idx',
    'ego2lidar', 'lidar2ego', 'fixed_range_imgs'
]
occ_num_classes = 18
occ3d_classes_names = _base_.occ3d_classes_names
waymo_label_mapping = {
    0: 0, 1: 4, 2: 7, 3: 15, 4: 2, 5: 15, 6: 15, 7: 8,
    8: 2, 9: 6, 10: 15, 11: 16, 12: 16, 13: 11, 14: 13, 23: 17
}

range_projection = dict(
    type='PointsToRangeImage',
    height=range_resolution[0],
    width=range_resolution[1],
    fov_up=fov_up,
    fov_down=fov_down,
    min_depth=min_depth,
    max_depth=max_depth,
    pc_range=point_cloud_range,
    pc_range_coord='ego',
    points_coord='lidar',
    filter_out_of_fov=True)

val_pipeline = [
    dict(type='LoadOcc3DFromFile', to_long=False),
    dict(
        type='RemapOccLabels',
        mapping=waymo_label_mapping,
        unknown_label=255),
    dict(
        type='CustomLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        pc_range=None,
        filter_egocar_points=True,
        filter_egocar_coord='ego'),
    dict(**range_projection),
    dict(**range_projection, output_key='fixed_range_imgs'),
    dict(
        type='MMPack3DDetInputs',
        keys=['voxel_semantics', 'range_imgs'],
        meta_keys=meta_keys),
]
train_pipeline = val_pipeline

shared_dataset = dict(
    type=dataset_type,
    modality=dict(use_camera=False, use_lidar=True, use_occ=True),
    load_prev_number=0,
    load_futu_number=0,
    load_interval=1,
    use_sequence_group_flag=False,
    serialize_data=False,
    test_mode=True)

# Both splits use inference; train_extract_dataset selects the training split
# for latent generation, without fitting the tokenizer.
train_extract_dataset = dict(
    **shared_dataset,
    data_root=train_data_root,
    occ_root='training',
    ann_file=train_ann_file,
    lidar_root=train_lidar_root,
    lidar_extrinsics_file=train_extrinsics_file,
    pipeline=train_pipeline)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **shared_dataset,
        data_root=data_root,
        ann_file=val_ann_file,
        lidar_root=val_lidar_root,
        lidar_extrinsics_file=val_extrinsics_file,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='Occ3DMetric',
        num_classes=occ_num_classes,
        empty_cls=17,
        ignore_index=255,
        label2cat=occ3d_classes_names,
        prefix='Rec_Occ',
        vis_interval=0),
    dict(
        type='LiDAR3DMetric',
        max_pred_points=None,
        cd_only=True,
        prefix='Rec_LiDAR',
        vis_interval=0),
]
test_evaluator = val_evaluator

# Runtime: use the source tokenizer; no tokenizer optimization.
train_cfg = None
train_dataloader = None
optim_wrapper = None
param_scheduler = None
auto_scale_lr = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
env_cfg = dict(dist_cfg=dict(backend='gloo'))
randomness = dict(seed=20261010, deterministic=False)
load_from = 'ckpts/occ_lidar_vae.pth'
resume = False

# ----------- Results -----------
# Occupancy Metric
# +----------------------+------------+
# | Class                | IoU        |
# +----------------------+------------+
# | others               | 0.6581     |
# | barrier              | nan        |
# | bicycle              | 0.8793     |
# | bus                  | nan        |
# | car                  | 0.8698     |
# | construction_vehicle | nan        |
# | motorcycle           | 0.5074     |
# | pedestrian           | 0.9034     |
# | traffic_cone         | 0.8481     |
# | trailer              | nan        |
# | truck                | nan        |
# | driveable_surface    | 0.9257     |
# | other_flat           | nan        |
# | sidewalk             | 0.8924     |
# | terrain              | nan        |
# | manmade              | 0.7060     |
# | vegetation           | 0.6547     |
# | free                 | 0.9849     |
# | ----------           | ---------- |
# | mIoU                 | 0.7845     |
# | ----------           | ---------- |
# | Non-empty IoU        | 0.8215     |
# +----------------------+------------+
# LiDAR Metric
# +------------------+--------+
# | Metric           | Value  |
# +------------------+--------+
# | Chamfer_Distance | 0.1992 |
# +------------------+--------+
