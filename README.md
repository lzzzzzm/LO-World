<h1 align="center">LO-World: Unified BEV Latent Modeling for Joint 4D Occupancy and LiDAR Forecasting</h1>

LO-World predicts future 3D occupancy and LiDAR observations in a shared BEV
latent space. At 2 Hz, its 6-frame and 12-frame models predict 3 and 6 seconds.

> 💬 Questions about installation or usage? Please open an issue in this repository.

<!-- demo:start -->
## Demo

https://github.com/user-attachments/assets/6d579523-b6f2-42f4-98c5-ddffa46fc550
<!-- demo:end -->

## Installation

Choose the Conda environment for your task. Run commands from the repository
root.

| Task | Environment |
| --- | --- |
| Tokenizer, occupancy World Model and semantic rendering | `loworld` |
| GenieDrive video generation and fine-tuning | `loworld-genie` |
| W-CODA FVD evaluation | `loworld` |
| BEVFormer mAP / mIoU / NDS evaluation | `worldbench` |

### LO-World: tokenizer and occupancy World Model

Use Python 3.10, PyTorch 2.4.0 with CUDA 11.8, MMCV 2.1.0, MMDetection 3.3.0,
and MMDetection3D 1.4.0. A matching CUDA Toolkit (`nvcc`) is required to build
the custom operators.

```bash
conda create -n loworld python=3.10 -y
conda activate loworld
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install "setuptools<81" wheel ninja mmengine==0.10.7
MAX_JOBS=8 pip install --no-build-isolation mmcv==2.1.0
pip install mmdet==3.3.0 mmdet3d==1.4.0
pip install -r requirements.txt
MAX_JOBS=8 pip install --no-build-isolation ./externals/chamferdist
MAX_JOBS=8 pip install --no-build-isolation \
  ./third_party/geniedrive_occ_rasterizer/diff-gaussian-rasterization
```

The CUDA rasterizer provides `diff_gaussian_rasterization` for GenieDrive
semantic condition rendering in the `loworld` environment.

### GenieDrive: downstream video generation

Use a separate Python 3.10 environment with PyTorch 2.4.0 and CUDA 11.8. Install
the dependencies for the task you plan to run:

```bash
conda create -n loworld-genie python=3.10 -y
conda activate loworld-genie
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-genie.txt
```

### WorldBench: BEVFormer evaluation

Use a separate Python 3.9 environment with PyTorch 2.1.0 and CUDA 11.8.
The evaluator uses MMCV-full 1.6.0, MMDetection 2.28.2, MMSegmentation 0.30.0
and the BEVFusion-based MMDetection3D from WorldLens. CUDA Toolkit 11.8
(`nvcc`) is required to build the operators.

```bash
conda create -n worldbench python=3.9.20 -y
conda activate worldbench
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install "setuptools<81" wheel ninja addict pyyaml "yapf<0.40.2" \
  "numpy==1.23.5" scipy==1.13.1 scikit-image==0.19.3 scikit-learn==1.6.1 \
  shapely==1.8.5.post1 \
  "opencv-python<4.12" mmdet==2.28.2 mmcls==0.25.0 \
  mmsegmentation==0.30.0 nuscenes-devkit==1.1.11 \
  torchmetrics==0.7.3 numba==0.53.0 einops

# WorldLens supplies the compatible MMCV and BEVFusion source trees.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/worldbench/WorldLens.git externals/WorldLens
GIT_LFS_SKIP_SMUDGE=1 git -C externals/WorldLens checkout fa6c5ff177e99d0d5b126f238ca42a430982b19a
MMCV_WITH_OPS=1 MAX_JOBS=8 pip install --no-build-isolation --no-deps \
  -e ./externals/WorldLens/worldbench/third_party/mmcv-1.6.0
MAX_JOBS=8 pip install --no-build-isolation --no-deps \
  -e ./externals/WorldLens/worldbench/third_party/bevfusion
```

Download the [evaluation sources](docs/inference_evaluation.md#evaluation-sources).
Place the [BEVFormer checkpoint](https://drive.google.com/file/d/1UM9Ifduu-IMF7qAW9nUlwXiXPZ17RkqQ/view)
at `ckpts/bevformer/bevformer_base_seg_det_150.pth`. Run evaluation using
`CONDA_ENV=worldbench`; see [evaluation commands](docs/inference_evaluation.md#bevformer-mapmiou).

## Data Preparation

* Follow the [Data Preparation guide](docs/data_preparation.md) to download and prepare the nuScenes and Waymo dataset.


### Expected folder structure after preparation

```text
data/
├── nuscenes/                                      # download nuScenes v1.0 trainval
│   ├── samples/
│   ├── sweeps/
│   ├── maps/
│   ├── v1.0-trainval/
│   ├── gts/                                        # download Occ3D-nuScenes labels
│   │   └── <scene_name>/<sample_token>/labels.npz
│   ├── stc-results/                                # optional: STCOcc val predictions
│   │   └── <scene_name>/<sample_token>/labels.npz
│   └── stc-results-train/                          # optional: STCOcc train predictions
│       └── <scene_name>/<sample_token>/labels.npz
├── nus_infos/                                      # download from Hub or generate
│   ├── nuscenes_infos_train.pkl
│   └── nuscenes_infos_val.pkl
├── nus_infos_12hz/                                 # download from Hub; GenieDrive
│   ├── nuscenes_interp_12Hz_infos_train.pkl
│   └── nuscenes_interp_12Hz_infos_val.pkl
├── nus_tokenizer_lidar_range/                      # generate with preprocessing script
│   └── <sample_token>.npz
├── nus_occ_lidar_latent/                           # generate with tokenizer checkpoint
│   └── <sample_token>.npz
├── stc_occ_lidar_latent/                           # optional: STCOcc val latents
│   └── <sample_token>.npz
├── stc_occ_lidar_latent_train/                     # optional: STCOcc train latents
│   └── <sample_token>.npz
└── bev_layout/                                     # download archives and extract, or generate
    ├── <sample_token>.npy                          # native 2 Hz model inputs
    ├── 12hz_train700_81f/                          # optional download: prepared train conditions
    └── 12hz_val150_81f/                            # optional download: prepared val conditions

work_dirs/
├── genie_occ_train700_81f/                         # generate with prepare_conditions.sh
│   ├── selection_manifest.json                     # generated
│   ├── selected_12hz_frames.pkl                    # generated
│   ├── conditions/<clip_id>.npz                    # generated
│   ├── predictions/<clip_id>.npz                   # generated
│   └── genie_dataset/                              # generated; training input
│       ├── RGB_<clip_id>_<camera>.mp4
│       ├── SEMANTIC_<clip_id>_<camera>.mp4
│       ├── clip_manifests/<clip_id>.json
│       └── meta.json
└── genie_occ_val150_81f/                            # same outputs for inference
    └── genie_dataset/
```

The `data/bev_layout/12hz_*_81f/` directories are optional downloaded packages.
When generating conditions from raw data, `prepare_conditions.sh` writes them
directly into the corresponding `work_dirs/genie_occ_*/` directory.

For the optional Waymo pipeline, the processed data uses a separate tree:

```text
data/waymo/
├── raw/                                           # download Waymo TFRecords
│   ├── train/
│   └── val/
├── waymo_info/
│   ├── train_10hz.pkl                             # MMDetection3D train infos; see docs/data_preparation.md
│   ├── waymo_infos_val.pkl                        # MMDetection3D val infos; see docs/data_preparation.md
│   ├── train_2hz.pkl                              # generated by converter
│   ├── val_2hz.pkl                                # generated by converter
│   ├── train_conversion_manifest.json             # generated by converter
│   └── val_conversion_manifest.json               # generated by converter
├── points/                                        # generated from TFRecords
│   ├── train/<sample_id:07d>.bin
│   └── val/<sample_id:07d>.bin
├── training/<scene>/<frame>_04.npz                # download Occ3D-Waymo labels
├── validation/<scene>/<frame>_04.npz              # download Occ3D-Waymo labels
└── latents/                                       # generated with tokenizer
    ├── train/<scene>_<frame>.npz
    └── val/<scene>_<frame>.npz
```

## Project guides

- [Installation](#installation)
- [Data preparation](docs/data_preparation.md): pretrained downloads, nuScenes, BEVLayout, and Waymo inputs.
- [Inference and evaluation](docs/inference_evaluation.md): World Model prediction, GenieDrive video generation, and metrics.
- [Visualization quick start](#quick-start-visualization): predicted OCC/LiDAR, semantic rendering, and video generation.
- [Training](docs/training.md): tokenizer, World Model, STCOcc fine-tuning, and downstream GenieDrive video generation training.

## Quick start: visualization

Complete [data preparation and checkpoint downloads](docs/data_preparation.md)
and install the `loworld` and `loworld-genie` [environments](#installation). The example generates
81 frames (current reconstruction + 80 future frames) at 12 FPS.

The pipeline forecasts occupancy and LiDAR, renders six-camera semantic
conditions, and uses GenieDrive to generate camera videos.

For BEVLayout-conditioned generation, use
`configs/world_model/bev_world_12f_layout_autoregressive.py` with
`ckpts/bev_world_12f_layout_autoregressive.pth` and the Layout arrays in
`data/bev_layout`. See the
[evaluation commands](docs/inference_evaluation.md#bevlayout-conditioned-world-model).

1. Generate OCC and LiDAR predictions:

   ```bash
   conda run --no-capture-output -n loworld python \
     tools/genie/occ_generation/generate_scene.py \
     --scene 0001 \
     --out-dir work_dirs/visualization/scene_0001
   ```

2. Render six-camera semantic conditions from the latest successful prediction:

   ```bash
   conda run --no-capture-output -n loworld python \
     tools/genie/semantic_export/renderer.py \
     --scene-dir work_dirs/visualization/scene_0001
   ```

   When moving results, copy the complete scene directory.

3. Generate six-camera videos and the overview GIF:

   ```bash
   conda run --no-capture-output -n loworld-genie python \
     tools/genie/inference/generate_scene.py \
     --scene-dir work_dirs/visualization/scene_0001
   ```

Outputs under `work_dirs/visualization/scene_0001/`:

- `overview.gif`: combined six-camera and OCC/LiDAR preview.
- `video/000000_grid.mp4`: six-camera grid video.
- `video/000000_CAM_*.mp4`: individual camera videos.

## Acknowledgements

We thank the authors of the following projects for their open-source contributions:

- OpenMMLab: [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [GenieDrive](https://github.com/Huster-YZY/GenieDrive)
- [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [WorldLens](https://github.com/worldbench/WorldLens)
- [W-CODA 2024 Track 2](https://github.com/pixeli99/W-CODA2024-Track2)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)
- [STCOcc](https://github.com/lzzzzzm/STCOcc)
- [ChamferDistance](https://github.com/krrish94/chamferdist) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- [Differentiable Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

## License

The LO-World project code is released under the [MIT License](LICENSE).
Third-party code retains its original licenses and notices; see
[`externals/`](externals/) and [`third_party/`](third_party/).
