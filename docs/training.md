# Training

The nuScenes training workflow is:

**Train tokenizer → export train/val latents → train World Model.**

Run commands from the repository root in the `loworld` Conda environment.
The examples use eight GPUs:

```bash
conda activate loworld
```

## 1. Train the tokenizer

Prepare nuScenes, Occ3D labels and LiDAR range-image caches using
[data preparation steps 2–3](data_preparation.md#2-prepare-nuscenes).
Train the joint occupancy/LiDAR tokenizer for 40 epochs:

```bash
bash tools/dist_train.sh configs/tokenizer/occ_lidar_vae.py 8 \
  --work-dir work_dirs/occ_lidar_vae
```

The following steps use `work_dirs/occ_lidar_vae/epoch_40.pth`.

## 2. Export train/val latents

After tokenizer training finishes, use that checkpoint to encode **both**
training and validation frames. This runs the tokenizer in inference mode:

```bash
for split in train val; do
  bash tools/dist_test.sh configs/tokenizer/occ_lidar_vae.py \
    work_dirs/occ_lidar_vae/epoch_40.pth 8 \
    --cfg-options \
    "test_dataloader.dataset.ann_file=data/nus_infos/nuscenes_infos_${split}.pkl" \
    model.save_latent_root=data/nus_occ_lidar_latent
done
```

Output: `data/nus_occ_lidar_latent/<sample_token>.npz`. World Model training
reads these cached latents and keeps the tokenizer frozen. Use the same
tokenizer checkpoint for latent export and all World Model stages; exporting
with a different tokenizer requires regenerating the latents.

## 3. Train the World Model

The configurations support 2f, 6f, 12f autoregressive, and BEVLayout-conditioned
World Models. Frame counts refer to future predictions at 2 Hz; the 12f models
roll out two 6-frame chunks. The tokenizer remains frozen during World Model
training.

| Stage | Initialization | Epochs | Prediction horizon |
| --- | --- | --- | --- |
| 2f | Trained tokenizer; randomly initialized World Model | 72 | 1 second |
| 6f | 2f EMA checkpoint | 20 | 3 seconds |
| 12f | 6f EMA checkpoint | 8 | 6 seconds |
| 12f with BEVLayout | Released BEVLayout-conditioned checkpoint | 8 | 6 seconds |
| 6f with STCOcc | Released `ckpts/bev_world_6f.pth` | 8 | 3 seconds |

The released `ckpts/bev_world_6f.pth` is also the initialization for Waymo
fine-tuning in `configs/world_model/bev_world_6f_waymo.py`. Both adaptations
use the six-frame World Model.

The commands below use output directories matching the configs' initialization
paths. If you change them, update the next stage's checkpoint path too. Keep
`model.vae_pretrained` consistent with the tokenizer used in step 2.

### 3.1. 2f

Train with the tokenizer checkpoint from step 1 and latents from step 2:

```bash
bash tools/dist_train.sh configs/world_model/bev_world_2f.py 8 \
  --work-dir work_dirs/bev_world_2f
```

Output checkpoint: `work_dirs/bev_world_2f/epoch_72_ema.pth`.

### 3.2. 6f

Initialize from the 2f EMA checkpoint (`model.world_pretrained` in the config):

```bash
bash tools/dist_train.sh configs/world_model/bev_world_6f.py 8 \
  --work-dir work_dirs/bev_world_6f
```

Output checkpoint:
`work_dirs/bev_world_6f/epoch_20_ema.pth`.

To start STCOcc or Waymo fine-tuning directly, download the released six-frame
checkpoint from [data preparation step 1](data_preparation.md#1-download-released-files).

### 3.3. Autoregressive World Model

Initialize from the 6f EMA checkpoint produced in step 3.2. In
`configs/world_model/bev_world_12f_autoregressive.py`, set
`stage_6f_checkpoint` to
`work_dirs/bev_world_6f/epoch_20_ema.pth` and
`model.vae_pretrained` to `work_dirs/occ_lidar_vae/epoch_40.pth`, then run:

```bash
bash tools/dist_train.sh configs/world_model/bev_world_12f_autoregressive.py 8 \
  --work-dir work_dirs/bev_world_12f_autoregressive
```

### 3.4. BEVLayout-conditioned World Model

Prepare the train/val Layout arrays, tokenizer latents, and LiDAR range-image
caches using [data preparation steps 2–5](data_preparation.md#2-prepare-nuscenes).
The models read Layout arrays from `data/bev_layout`.

For six future frames at 2 Hz (0.5–3 seconds), initialize from the 6f EMA
checkpoint produced in step 3.2 and the tokenizer checkpoint from step 1:

```bash
bash tools/dist_train.sh configs/world_model/bev_world_6f_layout.py 8 \
  --work-dir work_dirs/bev_world_6f_layout
```

Output: `work_dirs/bev_world_6f_layout/epoch_8_ema.pth`.

For 12 future frames at 2 Hz (0.5–6 seconds), use two six-frame autoregressive
chunks with the following configuration.

Initialize from `work_dirs/bev_world_6f_layout/epoch_8_ema.pth` produced above.
Set `layout_6f_checkpoint` in the configuration if the six-frame Layout
checkpoint is stored elsewhere. This restores the model, frozen tokenizer,
and EMA weights, then starts a new eight-epoch fine-tuning run:

```bash
bash tools/dist_train.sh configs/world_model/bev_world_12f_layout_autoregressive.py 8 \
  --work-dir work_dirs/bev_world_12f_layout
```

The configuration uses 8 epochs, 8 GPUs × batch 1, AdamW with learning rate
1e-5, and a 10× learning-rate multiplier for the Layout embedding and adapter.
Validation and checkpoint saving run every 2 epochs. Checkpoints and logs are
written to `work_dirs/bev_world_12f_layout/`. See
[BEVLayout evaluation](inference_evaluation.md#bevlayout-conditioned-world-model)
for the matching inference command.

Output: `work_dirs/bev_world_12f_layout/epoch_8_ema.pth`.

### 3.5. STCOcc fine-tuning (6f)

The released STCOcc adaptation fine-tunes the six-frame World Model from
`ckpts/bev_world_6f.pth`. It predicts six future frames at 2 Hz (0.5–3 seconds)
and keeps the joint occupancy-LiDAR tokenizer frozen.

Download `bev_world_6f.pth` and `occ_lidar_vae.pth` from
[data preparation step 1](data_preparation.md#1-download-released-files), then
prepare the [STCOcc train/val latents](data_preparation.md#stcocc-latents-optional).

Start fine-tuning:

```bash
bash tools/dist_train.sh configs/world_model/bev_world_6f_stc_occ.py 8 \
  --work-dir work_dirs/bev_world_6f_stc_occ_finetune
```

Output: `work_dirs/bev_world_6f_stc_occ_finetune/epoch_8_ema.pth`.
See [STCOcc evaluation](inference_evaluation.md#stcocc-conditioned-world-model).

## 4. Downstream video generation

After preparing the train700 RGB references and **12 Hz semantic conditions**
using [data preparation step 6](data_preparation.md#6-prepare-geniedrive-video-conditions),
fine-tune GenieDrive in two stages. Both use eight GPUs at **448×800**:

| Stage | Initialization | Frames | Training steps |
| --- | --- | --- | --- |
| 37f | Released GenieDrive transformer checkpoint | 37 | 5,000 |
| 81f | The 37f checkpoint from step 5,000 | 81 | 1,000 additional steps |

The 81f stage loads the trained 37f transformer weights and starts a new
1,000-step fine-tuning run. Wan2.1-Fun supplies the base model components.
Install the [loworld-genie environment](../README.md#geniedrive-downstream-video-generation)
and download the GenieDrive transformer and Wan base model from
[data preparation step 1](data_preparation.md#1-download-released-files).

Set the shared options from the repository root. Training uses bf16 and
DeepSpeed ZeRO-2. Both stages read the prepared 81-frame clips; the first
stage samples 37-frame sequences:

```bash
export CONDA_ENV=loworld-genie GPUS=0,1,2,3,4,5,6,7 NUM_PROCESSES=8
export DATASET_DIR=work_dirs/genie_occ_train700_81f/genie_dataset
export HEIGHT=448 WIDTH=800
export IMAGE_SAMPLE_SIZE=800 VIDEO_SAMPLE_SIZE=800 TOKEN_SAMPLE_SIZE=800
export LOW_VRAM=1 CHECKPOINTING_STEPS=500
```

### 4.1. 37f: 5,000 steps

Initialize from the released GenieDrive transformer:

```bash
NUM_FRAMES=37 MAX_TRAIN_STEPS=5000 RESUME_FROM_CHECKPOINT= \
INIT_CHECKPOINT=ckpts/geniedrive/GenieDrive/diffusion_pytorch_model.safetensors \
OUTPUT_DIR=work_dirs/genie_occ_train700_37f/finetune_5000steps \
bash tools/genie/training/train.sh
```

### 4.2. 81f: 1,000 additional steps

Initialize from the 37f checkpoint produced above:

```bash
NUM_FRAMES=81 MAX_TRAIN_STEPS=1000 RESUME_FROM_CHECKPOINT= \
INIT_CHECKPOINT=work_dirs/genie_occ_train700_37f/finetune_5000steps/checkpoint-5000/transformer/diffusion_pytorch_model.safetensors \
OUTPUT_DIR=work_dirs/genie_occ_train700_81f/finetune_1000steps \
bash tools/genie/training/train.sh
```

Final transformer:
`work_dirs/genie_occ_train700_81f/finetune_1000steps/checkpoint-1000/transformer/`.

For validation and metrics, see
[inference and evaluation](inference_evaluation.md#geniedrive-video-generation).
