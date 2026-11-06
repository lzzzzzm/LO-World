# Inference and evaluation

Run commands from the repository root. Prepare the required data as described
in the [data preparation guide](data_preparation.md).

## World Model

1. Choose a matching config and checkpoint:

   - nuScenes, 6 frames: `configs/world_model/bev_world_6f.py` + `ckpts/bev_world_6f.pth`
   - nuScenes autoregressive, 12 frames: `configs/world_model/bev_world_12f_autoregressive.py` + `ckpts/bev_world_12f_autoregressive.pth`
   - nuScenes BEVLayout-conditioned, 12 frames: `configs/world_model/bev_world_12f_layout_autoregressive.py` + `ckpts/bev_world_12f_layout_autoregressive.pth`
   - nuScenes STCOcc-conditioned, 6 frames: `configs/world_model/bev_world_6f_stc_occ.py` + `ckpts/bev_world_6f_stc.pth`
   - Waymo, 6 frames: `configs/world_model/bev_world_6f_waymo.py` + `ckpts/bev_world_6f_waymo.pth` (end-to-end inference not validated)

   `ckpts/occ_lidar_vae.pth` is the joint occupancy-LiDAR tokenizer used for
   reconstruction and latent export.

2. Evaluate the model. Example: nuScenes, 12-frame autoregressive checkpoint:

   ```bash
   bash tools/dist_test.sh configs/world_model/bev_world_12f_autoregressive.py \
     ckpts/bev_world_12f_autoregressive.pth 8
   ```

   Metrics: occupancy mIoU, non-empty IoU and LiDAR Chamfer Distance.

   To evaluate `ckpts/bev_world_6f.pth`, use `configs/world_model/bev_world_6f.py`
   with the same command format. This six-frame checkpoint is also
   the initialization for STCOcc and Waymo fine-tuning.

### BEVLayout-conditioned World Model

The BEVLayout-conditioned World Model uses a six-frame generator internally and
rolls out two chunks: 12 future frames at 2 Hz, covering 0.5–6 seconds.
Train/val/test pipelines read the Layout arrays from `data/bev_layout`.
The evaluators cover the current frame plus 12 futures, with the four-frame
sequence warmup. Scene and batch generation commands use
`ckpts/bev_world_12f_layout_autoregressive.pth` by default.

```bash
conda activate loworld
bash tools/dist_test.sh configs/world_model/bev_world_12f_layout_autoregressive.py \
  ckpts/bev_world_12f_layout_autoregressive.pth 8 \
  --work-dir work_dirs/bev_world_12f_layout_eval
```

Outputs go to `work_dirs/bev_world_12f_layout_eval/`.
To evaluate another full Layout checkpoint, replace the checkpoint argument.
Compare `future_avg` only with results using the same horizon; a 12-frame
average covers a different horizon from a six-frame average.

### STCOcc-conditioned World Model

The released STCOcc model is fine-tuned from `ckpts/bev_world_6f.pth` and predicts
six future frames. See [STCOcc fine-tuning](training.md#35-stcocc-fine-tuning-6f)
to train this variant.

Prepare the [STCOcc validation latents](data_preparation.md#stcocc-latents-optional),
then evaluate in the `loworld` environment:

```bash
bash tools/dist_test.sh configs/world_model/bev_world_6f_stc_occ.py \
  ckpts/bev_world_6f_stc.pth 8 \
  --work-dir work_dirs/bev_world_6f_stc_eval
```

Output: occupancy and LiDAR metrics and logs in `work_dirs/bev_world_6f_stc_eval/`.
The model predicts six future frames at 2 Hz, covering 0.5–3 seconds.
STCOcc latents are model inputs; evaluation targets remain the original Occ3D
labels and LiDAR range images.

### Waymo

Prepare the fixed splits and latents using
[data preparation step 7](data_preparation.md#7-prepare-waymo-optional), then run:

```bash
bash tools/dist_test.sh configs/world_model/bev_world_6f_waymo.py \
  ckpts/bev_world_6f_waymo.pth 8
```

The protocol predicts six future frames at 2 Hz (0.5–3 seconds), with four
frames of streaming memory and 6,049 valid samples after warmup and future
padding exclusions. End-to-end inference with the released checkpoint has
not been validated.

Points are filtered in ego coordinates to `[-40, -40, -1, 40, 40, 5.4]` m
and projected to 64×2048 with vertical FOV `[-17.734°, 2.274°]` and depth
range `(1.45, 80)` m; each pixel keeps the nearest depth. LiDAR GT is
reconstructed from projected depth, and Chamfer Distance uses complete point
sets. Occupancy uses 18 mapped classes, free class 17 and ignore index 255.

## GenieDrive video generation

Generate six-camera RGB videos from the prepared semantic conditions using
the LO-World GenieDrive transformer and Wan base model.

1. Prepare val150 video conditions using [data preparation step 6](data_preparation.md#6-prepare-geniedrive-video-conditions):

   ```bash
   bash tools/genie/prepare_conditions.sh
   ```

   Output: `work_dirs/genie_occ_val150_81f/genie_dataset/`.

2. Install `loworld-genie` using the [installation instructions](../README.md#geniedrive-downstream-video-generation).
   Download the [Wan base model and LO-World transformer](data_preparation.md#1-download-released-files).

3. Generate all 150 clips across eight GPUs:

   ```bash
   CONDA_ENV=loworld-genie GPU_IDS=0,1,2,3,4,5,6,7 \
   DATA_DIR=work_dirs/genie_occ_val150_81f/genie_dataset \
   NUM_FRAMES=81 MAX_SAMPLES=150 SAVE_DIR=work_dirs/genie_inference_val150_81f \
   bash tools/genie/inference/infer.sh
   ```

   Outputs: `work_dirs/genie_inference_val150_81f/`.
   Set `NUM_FRAMES=17 MAX_SAMPLES=1` to generate a single short clip.

## GenieDrive evaluation

### Evaluation sources

FVD and BEVFormer evaluation require the
[W-CODA 2024 Track 2 repository](https://github.com/pixeli99/W-CODA2024-Track2).
Download the sources from the LO-World repository root:

```bash
git clone https://github.com/pixeli99/W-CODA2024-Track2.git externals/W-CODA2024-Track2
git -C externals/W-CODA2024-Track2 checkout e194e95253c34441fa6daa90e3f273d2e67cd6e1
```

The evaluation commands below use this checkout automatically. To use an
existing checkout, set `WCODA_ROOT` for FVD or `BEVFORMER_ROOT` to its
`BEVFormer_segmentation_detection/` directory for BEVFormer.
Downloaded sources are excluded from Git and retain their upstream terms.

### W-CODA FVD

Use the `loworld` environment with [requirements.txt](../requirements.txt)
installed. The evaluator reads W-CODA sources from
`externals/W-CODA2024-Track2/` and expects `i3d_pretrained_400.pt` in
`ckpts/fvd/` ([weight download](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI)).
Use the val150 conditions prepared in the data preparation guide.

Evaluate all 81 frames and the first 37 frames using eight GPUs:

```bash
WCODA_DATASET_DIR=work_dirs/genie_occ_val150_81f \
bash tools/genie/evaluation/run_wcoda_fvd.sh \
  --gt_path data/nuscenes \
  --pred_path work_dirs/genie_inference_val150_81f
```

Results: `fvd_81f.json` and `fvd_37f.json` under
`work_dirs/genie_inference_val150_81f/eval/wcoda_fp64_batch50/`.

### BEVFormer mAP/mIoU

Install the [worldbench environment](../README.md#worldbench-bevformer-evaluation)
and BEVFormer checkpoint. Set `DATASET_ROOT` to the condition dataset containing
`selection_manifest.json` and `conditions/`. The GT infos must contain the same
frame tokens as those conditions:

```bash
CONDA_ENV=worldbench GPUS=0,1,2,3,4,5,6,7 \
DATASET_ROOT=/path/to/condition_dataset \
bash tools/genie/evaluation/worldbench/evaluate.sh \
  --gt_path /path/to/matching_infos.pkl \
  --pred_path /path/to/generated_videos
```
