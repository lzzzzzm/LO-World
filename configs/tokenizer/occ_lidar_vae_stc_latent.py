"""Extract joint latents using STC image-predicted OCC on nuScenes val."""

_base_ = './occ_lidar_vae.py'

stc_occ_root = 'data/nuscenes/stc-results'
stc_latent_root = 'data/stc_occ_lidar_latent'
tokenizer_range_root = 'data/nus_tokenizer_lidar_range'

model = dict(save_latent_root=stc_latent_root)

meta_keys = [
    'scene_name', 'token', 'total_token', 'sample_idx',
    'ego2lidar', 'lidar2ego'
]

stc_val_pipeline = [
    # Preserve the dataset-provided Occ3D labels as the evaluation target.
    dict(
        type='LoadOcc3DFromFile',
        to_long=False,
        output_key='gt_voxel_semantics'),
    # STCOcc is the tokenizer input, and therefore overwrites the standard key.
    dict(
        type='LoadOcc3DFromFile',
        data_root=stc_occ_root,
        to_long=False),
    dict(
        type='LoadDataFromFile',
        data_root=tokenizer_range_root,
        input_keys='range_imgs',
        output_keys='range_imgs',
        suffix='npz'),
    dict(
        type='MMPack3DDetInputs',
        keys=['voxel_semantics', 'range_imgs'],
        meta_keys=meta_keys + ['gt_voxel_semantics']),
]

test_dataloader = dict(dataset=dict(pipeline=stc_val_pipeline))

# Inherit the Occ3D and LiDAR reconstruction evaluators from the base config.
