"""BEVLayout-conditioned World Model with twelve-frame autoregressive rollout."""

_base_ = './bev_world_6f_layout.py'

layout_6f_checkpoint = 'work_dirs/bev_world_6f_layout/epoch_8_ema.pth'

futu_number = 6
memo_number = 4
rollout_future_number = 12
train_autoregressive_chunks = 2
eval_future_number = rollout_future_number
eval_frame_number = eval_future_number + 1

model = dict(
    # Restore the complete six-frame Layout model, including its frozen VAE.
    vae_pretrained=None,
    world_pretrained=None,
    use_bev_layout=True,
    bev_layout_in_channels=18,
    futu_number=futu_number,
    rollout_future_number=rollout_future_number,
    rollout_stride=futu_number,
    train_autoregressive_chunks=train_autoregressive_chunks,
    train_autoregressive_loss_weight=1.0,
    train_autoregressive_chunk_mode='round_robin',
    range_lidar_loss_frame_indices=[4, 5])

train_dataloader = dict(
    batch_size=1,
    dataset=dict(load_futu_number=rollout_future_number))
val_dataloader = dict(
    dataset=dict(load_futu_number=eval_future_number))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='Occ4DMetric',
        eval_frame_number=eval_frame_number,
        warmup_frame_number=memo_number,
        num_classes=18,
        empty_cls=17,
        ignore_index=255,
        label2cat=_base_.occ3d_classes_names,
        prefix='Fore_Occ',
        vis_interval=10),
    dict(
        type='LiDAR4DMetric',
        eval_frame_number=eval_frame_number,
        warmup_frame_number=memo_number,
        prefix='Fore_LiDAR',
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

train_cfg = dict(
    type='SequenceIterBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_epoch_interval)

auto_scale_lr = dict(enable=True, base_batch_size=num_gpus * batch_size)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=save_epoch_interval,
        max_keep_ckpts=max_keep_ckpts))

custom_hooks = [
    # Runner creates EMA before loading load_from; restore both from the
    # same six-frame EMA checkpoint, including the saved update counter.
    dict(
        type='MEGVIIEMAHook',
        resume=layout_6f_checkpoint,
        interval=save_epoch_interval,
        max_keep_ckpts=max_keep_ckpts),
]
randomness = dict(seed=20260710, deterministic=False)
load_from = layout_6f_checkpoint
resume = False

# +---------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric        | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | 3.5s   | 4s     | 4.5s   | 5s     | 5.5s   | 6s     | future_avg |
# +---------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
# | mIoU          | 0.8454 | 0.6332 | 0.5965 | 0.5701 | 0.5486 | 0.5293 | 0.5110 | 0.4941 | 0.4816 | 0.4699 | 0.4574 | 0.4466 | 0.4352 | 0.5145     |
# | Non-empty IoU | 0.7714 | 0.6495 | 0.6249 | 0.6046 | 0.5860 | 0.5677 | 0.5491 | 0.5247 | 0.5101 | 0.4959 | 0.4821 | 0.4676 | 0.4528 | 0.5429     |
# +---------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
# +------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Metric           | 0s     | 0.5s   | 1s     | 1.5s   | 2s     | 2.5s   | 3s     | 3.5s   | 4s     | 4.5s   | 5s     | 5.5s   | 6s     | future_avg |
# +------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
# | Chamfer_Distance | 0.1459 | 0.3066 | 0.3896 | 0.4829 | 0.5817 | 0.6526 | 0.7628 | 0.9560 | 1.0986 | 1.2609 | 1.4621 | 1.5186 | 1.7239 | 0.9330     |
# +------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+------------+
