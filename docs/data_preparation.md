# Data preparation

Run commands from the repository root. Activate the environment required by
each step.

## Choose a path

- **Tokenizer training:** steps 1–4.
- **World Model training/inference:** steps 1–4; add step 5 for Layout configs.
- **STCOcc (optional):** steps 1–3, then [STCOcc latents](#stcocc-latents-optional).
- **GenieDrive conditions:** steps 1–5, then step 6.
- **Waymo:** follow step 7; it is separate from nuScenes.

## 1. Download released files

Download only the files needed for your task:

| File | Command |
| --- | --- |
| Six checkpoints + checksums | `hf download JimmyLiaoo/LO-World bev_world_6f.pth bev_world_12f_autoregressive.pth bev_world_12f_layout_autoregressive.pth bev_world_6f_stc.pth bev_world_6f_waymo.pth occ_lidar_vae.pth SHA256SUMS --local-dir ckpts` |
| nuScenes train/val infos | `hf download JimmyLiaoo/LO-World nus_infos/nuscenes_infos_train.pkl nus_infos/nuscenes_infos_val.pkl --local-dir data` |
| STCOcc predictions (`r50_704_256_16f_occ3d`) | [Google Drive](https://drive.google.com/file/d/1dXB9mtROLWChycBZlhYIf_JBLshXogBs/view?usp=drive_link) |
| GenieDrive 12 Hz train/val infos | `hf download JimmyLiaoo/LO-World data/nus_infos_12hz/nuscenes_interp_12Hz_infos_train.pkl data/nus_infos_12hz/nuscenes_interp_12Hz_infos_val.pkl --local-dir .` |
| 2 Hz train/val BEVLayout | `hf download JimmyLiaoo/LO-World data/bev_layout/2hz/train.tar.zst data/bev_layout/2hz/val.tar.zst --local-dir .` |
| Published 12 Hz train700/val150 layouts (optional) | `hf download JimmyLiaoo/LO-World --include "data/bev_layout/12hz_*_81f/**" --local-dir .` |
| GenieDrive 448×800 transformer | `hf download JimmyLiaoo/LO-World geniedrive_448x800.safetensors --local-dir ckpts/geniedrive` |
| GenieDrive fine-tuning transformer | `hf download ANIYA673/GenieDrive diffusion_pytorch_model.safetensors --local-dir ckpts/geniedrive/GenieDrive` |
| Wan 1.3B base model | `hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control --local-dir ckpts/geniedrive/Wan2.1-Fun-V1.1-1.3B-Control` |

The release provides the nuScenes six-frame World Model, 12-frame autoregressive
World Model, BEVLayout-conditioned World Model, STCOcc-conditioned six-frame
World Model, Waymo World Model, and joint occupancy-LiDAR tokenizer.
`bev_world_6f.pth` is the shared initialization for STCOcc and Waymo fine-tuning.
Verify the downloaded files from `ckpts/` with `sha256sum -c SHA256SUMS`.

For STCOcc inputs, follow [STCOcc latent preparation](#stcocc-latents-optional).

## 2. Prepare nuScenes

1. Download nuScenes **v1.0 trainval** to `data/nuscenes/` (`samples/`,
   `sweeps/`, `maps/`, `v1.0-trainval/`).
2. Download Occ3D-nuScenes labels from the
   [official folder](https://drive.google.com/drive/folders/1wZ-8OI1IJkrXo6BudFSGmaKXBUYQ3ts_)
   and extract `gts/` into `data/nuscenes/`. See the
   [Occ3D instructions](https://github.com/Tsinghua-MARS-Lab/Occ3D#occ3d-nuscenes).
3. Download train/val infos in step 1, or generate them:

   ```bash
   python tools/create_nus_data.py nuscenes \
     --root-path data/nuscenes --version v1.0-trainval \
     --out-dir data/nus_infos --extra-tag nuscenes
   ```

4. For GenieDrive or BEVFormer evaluation, download the interpolated 12 Hz
   infos in step 1 to `data/nus_infos_12hz/`.

## 3. Generate LiDAR range-image caches

```bash
bash tools/precompute_nuscenes_lidar_range.sh
```

Output: `data/nus_tokenizer_lidar_range/<sample_token>.npz`. Required for
tokenizer training/export and World Model training/evaluation.

## 4. Export tokenizer latents

Use the tokenizer checkpoint that matches the World Model. Export both splits
for training; export val for inference:

```bash
for split in train val; do
  bash tools/dist_test.sh configs/tokenizer/occ_lidar_vae.py \
    ckpts/occ_lidar_vae.pth 8 \
    --cfg-options \
    "test_dataloader.dataset.ann_file=data/nus_infos/nuscenes_infos_${split}.pkl" \
    model.save_latent_root=data/nus_occ_lidar_latent
done
```

Output: `data/nus_occ_lidar_latent/<sample_token>.npz`. This step needs
nuScenes, Occ3D labels and the range-image caches from steps 2–3.

### STCOcc latents (optional)

Required only for STCOcc-conditioned World Model training or evaluation.

Complete steps 2–3 and download the
[STCOcc prediction results](https://drive.google.com/file/d/1dXB9mtROLWChycBZlhYIf_JBLshXogBs/view?usp=drive_link),
generated with the `r50_704_256_16f_occ3d` checkpoint from
[STCOcc](https://github.com/lzzzzzm/STCOcc#-model-zoo). Arrange validation
predictions as `data/nuscenes/stc-results/<scene_name>/<sample_token>/labels.npz`.

Encode them with the released tokenizer in the `loworld` environment:

```bash
bash tools/dist_test.sh configs/tokenizer/occ_lidar_vae_stc_latent.py \
  ckpts/occ_lidar_vae.pth 8
```

Output: validation latents in `data/stc_occ_lidar_latent/`.

For fine-tuning, also prepare predictions for the official training split using
the same STCOcc checkpoint. Generate any missing predictions in the STCOcc
repository and place them at
`data/nuscenes/stc-results-train/<scene_name>/<sample_token>/labels.npz`.
Export training latents separately:

```bash
bash tools/dist_test.sh configs/tokenizer/occ_lidar_vae_stc_latent.py \
  ckpts/occ_lidar_vae.pth 8 \
  --cfg-options \
  test_dataloader.dataset.ann_file=data/nus_infos/nuscenes_infos_train.pkl \
  test_dataloader.dataset.pipeline.1.data_root=data/nuscenes/stc-results-train \
  model.save_latent_root=data/stc_occ_lidar_latent_train
```

Output: training latents in `data/stc_occ_lidar_latent_train/`. Continue with
[STCOcc fine-tuning](training.md#35-stcocc-fine-tuning-6f) or
[evaluation](inference_evaluation.md#stcocc-conditioned-world-model).

## 5. Prepare BEVLayout (Layout configs only)

The 2 Hz archives cover all nuScenes train/val keyframes. The optional
12 Hz packages in step 1 cover selected train700/val150 clips of 81 frames,
rather than every frame of each full scene. A 12-frame World Model rollout
at 2 Hz spans six seconds; it is separate from the 12 Hz video conditions.

Download the 2 Hz archives from step 1 and extract them:

```bash
tar --zstd -xf data/bev_layout/2hz/train.tar.zst -C data/bev_layout
tar --zstd -xf data/bev_layout/2hz/val.tar.zst -C data/bev_layout
```

Output: `data/bev_layout/`, ready for BEVLayout-conditioned training and inference.

Alternatively, generate the 2 Hz train/val layouts yourself. Prepare the nuScenes
metadata, train/val infos and map expansion files (`data/nuscenes/maps/expansion/`)
from step 2. Start with an empty `data/bev_layout/` directory and run in the
`loworld` environment:

```bash
for split in train val; do
  python tools/trajectory_generation/bev_layout.py \
    --ann-file "data/nus_infos/nuscenes_infos_${split}.pkl" \
    --data-root data/nuscenes \
    --version v1.0-trainval \
    --box-convention native \
    --save-root data/bev_layout
done
```

Output: train/val layouts in `data/bev_layout/`, ready for training and inference.

## 6. Prepare GenieDrive video conditions

Prepare 81-frame clips at 12 FPS for GenieDrive inference or training.
The pipeline generates occupancy predictions, renders six-camera semantic
conditions, and exports the corresponding RGB reference videos.

1. Complete steps 2–5 and download the 12-frame BEVLayout checkpoint from step 1.
   nuScenes metadata, maps and camera images are required. You can also download
   the train700/val150 Layout packages from step 1; the script uses them when
   available.
2. Prepare val150 conditions:

   ```bash
   SPLIT=val NUM_CLIPS=150 GPUS=0,1,2,3,4,5,6,7 \
   bash tools/genie/prepare_conditions.sh
   ```

3. Prepare train700 conditions:

   ```bash
   SPLIT=train NUM_CLIPS=700 GPUS=0,1,2,3,4,5,6,7 \
   bash tools/genie/prepare_conditions.sh
   ```

Both examples use eight GPUs. The defaults are the `loworld` environment,
81 frames per clip, 450×800 conditions and 12 FPS. Set `GPUS=0` to use one GPU.

Use val150 for inference and train700 for video-model training. Each output
directory contains RGB reference videos, semantic condition videos, and
`meta.json`:

| Split | Output directory |
| --- | --- |
| val150 | `work_dirs/genie_occ_val150_81f/genie_dataset/` |
| train700 | `work_dirs/genie_occ_train700_81f/genie_dataset/` |

To resume an interrupted run, repeat the same command.

## 7. Prepare Waymo (optional)

1. Download [Waymo Perception TFRecords](https://waymo.com/open/download/)
   to `data/waymo/raw/{train,val}/` and
   [Occ3D-Waymo 0.4 m labels](https://github.com/Tsinghua-MARS-Lab/Occ3D#occ3d-waymo)
   (`*_04.npz`) to `data/waymo/{training,validation}/<scene>/<frame>_04.npz`.
   The [fixed split manifest](../tools/dataset_converters/waymo/splits.json)
   defines TFRecord names, sample IDs and timestamps: 50 training scenes /
   2,000 frames and 202 validation scenes / 8,069 frames. Preserve its scene
   numbering when generating the source infos.
2. Generate the source info PKLs with [MMDetection3D](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html).
   Prepare its Waymo inputs as described in the official guide, then run from
   the MMDetection3D repository root:

   ```bash
   TF_CPP_MIN_LOG_LEVEL=3 python tools/create_data.py waymo \
     --root-path ./data/waymo --out-dir ./data/waymo \
     --workers 8 --extra-tag waymo --version v1.4
   ```

   Copy the generated infos into this repository. `train_10hz.pkl` is the
   MMDetection3D `waymo_infos_train.pkl` renamed for this project's converter:

   ```bash
   cp /path/to/mmdetection3d/data/waymo/kitti_format/waymo_infos_train.pkl \
     data/waymo/waymo_info/train_10hz.pkl
   cp /path/to/mmdetection3d/data/waymo/kitti_format/waymo_infos_val.pkl \
     data/waymo/waymo_info/waymo_infos_val.pkl
   ```

3. Convert the selected splits in an environment with TensorFlow and the
   Waymo SDK:

   ```bash
   python tools/dataset_converters/waymo_converter.py --split all
   ```

   Output: 2 Hz infos in `data/waymo/waymo_info/` and point clouds in
   `data/waymo/points/{train,val}/`.

4. Export val latents for inference and train latents for training:

   ```bash
   # Validation
   bash tools/dist_test.sh configs/tokenizer/occ_lidar_vae_waymo.py \
     ckpts/occ_lidar_vae.pth 8

   # Training
   bash tools/dist_test.sh configs/tokenizer/occ_lidar_vae_waymo.py \
     ckpts/occ_lidar_vae.pth 8 \
     --cfg-options \
     test_dataloader.dataset.ann_file=waymo_info/train_2hz.pkl \
     test_dataloader.dataset.occ_root=training \
     test_dataloader.dataset.lidar_root=points/train \
     test_dataloader.dataset.lidar_extrinsics_file=waymo_info/train_conversion_manifest.json \
     model.save_latent_root=data/waymo/latents/train
   ```

   Output: `data/waymo/latents/{train,val}/`. See
   [Waymo evaluation](inference_evaluation.md#waymo) for the evaluation command.
