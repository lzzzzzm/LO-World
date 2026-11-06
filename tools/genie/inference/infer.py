#!/usr/bin/env python3
"""Generate six-camera videos from GenieDrive RGB and semantic conditions."""

import argparse
from contextlib import nullcontext, redirect_stdout
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.genie.inference.progress import InferenceProgress

progress = InferenceProgress()

# Match only the known dependency notices before importing videox_fun. Keep
# other FutureWarnings and actual model-loading diagnostics visible.
warnings.filterwarnings(
    'ignore', category=FutureWarning,
    message=r'^`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\.',
    module=r'^videox_fun\.(dist|models)(\.|$)')
warnings.filterwarnings(
    'ignore', category=FutureWarning,
    message=r'^You are using `torch\.load` with `weights_only=False`',
    module=r'^videox_fun\.models\.wan_vae$')

with progress.stage('Import PyTorch, Diffusers and dependencies'):
    import cv2
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from PIL import Image


CAM_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    default_project = Path(
        os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[3])
    )
    default_genie = os.environ.get(
        "GENIE_DIR", str(default_project / "third_party/geniedrive_occ_render")
    )
    default_base_model = os.environ.get(
        "WAN21_FUN_MODEL",
        str(
            default_project
            / "ckpts/geniedrive/Wan2.1-Fun-V1.1-1.3B-Control"
        ),
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--genie-dir", default=default_genie)
    parser.add_argument("--base-model", default=default_base_model)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--meta-path", default=None)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--video-length", type=int, default=17)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--depth-control-latent-weight", type=float, default=0.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--gpu-memory-mode",
        choices=[
            "model_full_load",
            "model_cpu_offload",
            "sequential_cpu_offload",
            "model_full_load_and_qfloat8",
            "model_cpu_offload_and_qfloat8",
        ],
        default="model_cpu_offload",
    )
    parser.add_argument("--disable-teacache", action="store_true")
    parser.add_argument("--teacache-threshold", type=float, default=0.10)
    parser.add_argument("--teacache-skip-start-steps", type=int, default=5)
    parser.add_argument("--cfg-skip-ratio", type=float, default=0.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-grid", action="store_true")
    return parser.parse_args()


def setup_imports(genie_dir):
    genie_dir = str(Path(genie_dir).resolve())
    if genie_dir not in sys.path:
        sys.path.insert(0, genie_dir)
    return genie_dir


def resolve_transformer_path(checkpoint):
    checkpoint = Path(checkpoint)
    if (checkpoint / "transformer").is_dir():
        return checkpoint / "transformer"
    if (
        checkpoint.is_dir()
        and not (checkpoint / "config.json").exists()
        and (checkpoint / "diffusion_pytorch_model.safetensors").exists()
    ):
        return checkpoint / "diffusion_pytorch_model.safetensors"
    return checkpoint


def load_transformer_state_dict(transformer, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(
        f"loaded transformer state from {checkpoint_path}; "
        f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"missing keys sample: {missing[:20]}", flush=True)
    if unexpected:
        print(f"unexpected keys sample: {unexpected[:20]}", flush=True)


def read_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def shard_records(records, start_index, max_samples, rank, world_size):
    indexed = list(enumerate(records))
    indexed = indexed[start_index:]
    if max_samples is not None:
        indexed = indexed[:max_samples]
    return indexed[rank::world_size]


def all_view_outputs_exist(save_dir, sample_idx):
    return all((save_dir / f"{sample_idx:06d}_{cam}.mp4").exists() for cam in CAM_NAMES)


def run_inference(args):
    # Fail before importing the full video pipeline or loading any model files.
    with progress.stage('Check CUDA devices'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                'GenieDrive video inference requires an exposed CUDA GPU; '
                'check the GPU device mapping and CUDA_VISIBLE_DEVICES.')
    genie_dir = setup_imports(args.genie_dir)

    with progress.stage('Import GenieDrive video model components'):
        from transformers import AutoTokenizer
        from videox_fun.models import (
            AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel,
        )
        from videox_fun.models.cache_utils import get_teacache_coefficients
        from videox_fun.pipeline import WanFunControlPipeline
        from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
        from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from videox_fun.utils.fp8_optimization import (
            convert_model_weight_to_float8, convert_weight_dtype_wrapper,
            replace_parameters_by_name,
        )
        from videox_fun.utils.utils import (
            filter_kwargs, get_image_latent, get_video_to_video_latent, save_videos_grid,
        )

    rank = args.rank
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    world_size = args.world_size
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GenieDrive video inference requires a CUDA GPU; install a CUDA-enabled "
            "PyTorch build and make an A100 visible with CUDA_VISIBLE_DEVICES."
        )
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"local_rank={local_rank}, but only {torch.cuda.device_count()} visible GPU(s)."
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    weight_dtype = dtype_map[args.dtype]

    data_dir = Path(args.data_dir)
    meta_path = Path(args.meta_path) if args.meta_path else data_dir / "meta.json"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    progress.set_output(save_dir / 'logs' / f'progress_rank{rank}.json', rank=rank, world_size=world_size)
    with progress.stage('Read condition metadata'):
        with meta_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        local_records = shard_records(records, args.start_index, args.max_samples, rank, world_size)
        progress.info('GPU: %s; rank=%d/%d; local samples: %d; condition metadata: %s',
                      torch.cuda.get_device_name(device), rank, world_size, len(local_records), meta_path)

    config_path = Path(args.config_path) if args.config_path else Path(genie_dir) / "config/wan2.1/wan_civitai.yaml"
    config = OmegaConf.load(config_path)
    transformer_path = resolve_transformer_path(args.checkpoint)
    transformer_init_path = (
        Path(args.base_model) if Path(transformer_path).is_file() else transformer_path
    )

    print(
        f"rank={rank}/{world_size} local_rank={local_rank} device={device} "
        f"samples={len(local_records)} transformer={transformer_path}",
        flush=True,
    )

    init_log_path = save_dir / 'logs' / f'model_load_rank{rank}.log'
    init_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress.info('Base model loading details: %s', init_log_path)
    with progress.stage('Load base Transformer'), init_log_path.open('w', buffering=1) as init_log:
        # A plain fine-tuned state dict is loaded after base initialization.
        # Preserve that intermediate missing-key list in a file; the final
        # checkpoint's actual missing/unexpected keys are still printed below.
        output_context = (redirect_stdout(init_log) if Path(transformer_path).is_file()
                          else nullcontext())
        with output_context:
            transformer = WanTransformer3DModel.from_pretrained(
                str(transformer_init_path),
                transformer_additional_kwargs=OmegaConf.to_container(
                    config["transformer_additional_kwargs"]
                ),
                low_cpu_mem_usage=False,
                torch_dtype=weight_dtype,
            )
    if Path(transformer_path).is_file():
        with progress.stage('Load fine-tuned Transformer weights'):
            load_transformer_state_dict(transformer, transformer_path)

    with progress.stage('Load VAE'):
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.base_model, config["vae_kwargs"].get("vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).to(weight_dtype)
    with progress.stage('Load tokenizer'):
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.base_model, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
        )
    with progress.stage('Load T5 text encoder'):
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.base_model, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
            low_cpu_mem_usage=False,
            torch_dtype=weight_dtype,
        ).eval()
    with progress.stage('Load CLIP image encoder'):
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(args.base_model, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))
        ).to(weight_dtype).eval()

    scheduler_name = "Flow"
    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[scheduler_name]
    scheduler = scheduler_cls(
        **filter_kwargs(scheduler_cls, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    with progress.stage('Build video generation pipeline'):
        pipeline = WanFunControlPipeline(
            transformer=transformer, vae=vae, tokenizer=tokenizer,
            text_encoder=text_encoder, scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        # The worker is piped through Conda/the scene controller. A carriage-
        # return progress bar collides with timestamped logs on that stream.
        # Use the step callback below as the single denoising progress display.
        pipeline.set_progress_bar_config(disable=True)

    with progress.stage(f'Configure GPU memory mode: {args.gpu_memory_mode}'):
        if args.gpu_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation"], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            transformer.mv_freqs = transformer.mv_freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif args.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif args.gpu_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif args.gpu_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)

    if not args.disable_teacache:
        coefficients = get_teacache_coefficients(args.base_model)
        if coefficients is not None:
            teacache_skip_start_steps = min(
                args.teacache_skip_start_steps, args.num_inference_steps
            )
            pipeline.transformer.enable_teacache(
                coefficients,
                args.num_inference_steps,
                args.teacache_threshold,
                num_skip_start_steps=teacache_skip_start_steps,
                offload=False,
            )

    if args.cfg_skip_ratio is not None:
        pipeline.transformer.enable_cfg_skip(args.cfg_skip_ratio, args.num_inference_steps)

    video_length = (
        int(
            (args.video_length - 1)
            // vae.config.temporal_compression_ratio
            * vae.config.temporal_compression_ratio
        )
        + 1
        if args.video_length != 1
        else 1
    )
    sample_size = [args.height, args.width]
    progress.info('Generation settings: %d frames, %d FPS, %dx%d, %d denoising steps, CFG=%.2f, TeaCache=%s',
                  video_length, args.fps, args.width, args.height,
                  args.num_inference_steps, args.guidance_scale,
                  'disabled' if args.disable_teacache else 'configured if supported')

    for sample_order, (sample_idx, item) in enumerate(local_records, 1):
        progress.info('Sample %d/%d: %06d; seed=%d', sample_order, len(local_records),
                      sample_idx, args.seed + sample_idx)
        if not args.overwrite and all_view_outputs_exist(save_dir, sample_idx):
            print(f"skip existing sample {sample_idx:06d}", flush=True)
            continue

        control_paths = []
        depth_control_paths = []
        has_depth_control = False
        ref_images = []
        with progress.stage(f'Sample {sample_idx:06d}: read six reference images'):
            for view_idx, cam in enumerate(CAM_NAMES, 1):
                progress.detail(f'Reference image {view_idx}/6: {cam}')
                control_rel = item[f"control_file_path_{cam}"]
                depth_control_rel = item.get(f"depth_control_file_path_{cam}")
                rgb_rel = item[f"file_path_{cam}"]
                control_path = data_dir / control_rel
                rgb_path = data_dir / rgb_rel
                if not control_path.exists():
                    raise FileNotFoundError(control_path)
                if not rgb_path.exists():
                    raise FileNotFoundError(rgb_path)
                control_paths.append(str(control_path))
                if depth_control_rel:
                    has_depth_control = True
                    depth_control_path = data_dir / depth_control_rel
                    if not depth_control_path.exists():
                        raise FileNotFoundError(depth_control_path)
                    depth_control_paths.append(str(depth_control_path))
                ref_images.append(read_first_frame(rgb_path))

        prompt = item.get("text", "") if args.prompt is None else args.prompt
        generator = torch.Generator(device=device).manual_seed(args.seed + sample_idx)
        with progress.stage(f'Sample {sample_idx:06d}: prepare reference image tensors'):
            clip_image = [img.convert("RGB") for img in ref_images]
            ref_image = torch.cat(
                [get_image_latent(img, sample_size=sample_size) for img in ref_images], dim=0
            )
            start_image = ref_image.clone()
        with progress.stage(f'Sample {sample_idx:06d}: read six semantic condition videos'):
            control_tensors = []
            for view_idx, path in enumerate(control_paths):
                progress.detail(f'Semantic video {view_idx + 1}/6: {CAM_NAMES[view_idx]}')
                control_tensors.append(get_video_to_video_latent(
                    path, video_length=video_length, sample_size=sample_size,
                    fps=args.fps, ref_image=None)[0])
            input_video = torch.cat(control_tensors, dim=0)
            del control_tensors
        depth_input_video = None
        if args.depth_control_latent_weight > 0 and has_depth_control:
            with progress.stage(f'Sample {sample_idx:06d}: read depth condition videos'):
                if len(depth_control_paths) != len(CAM_NAMES):
                    raise ValueError(
                        "depth_control_file_path must be provided for all camera views "
                        f"when --depth-control-latent-weight > 0; got {len(depth_control_paths)}."
                    )
                depth_tensors = []
                for view_idx, path in enumerate(depth_control_paths):
                    progress.detail(f'Depth video {view_idx + 1}/6: {CAM_NAMES[view_idx]}')
                    depth_tensors.append(get_video_to_video_latent(
                        path, video_length=video_length, sample_size=sample_size,
                        fps=args.fps, ref_image=None)[0])
                depth_input_video = torch.cat(depth_tensors, dim=0)
                del depth_tensors

        def report_step(pipe, step_index, timestep, callback_kwargs):
            progress.denoising_step(step_index + 1, pipe.num_timesteps)
            # No tensor inputs requested, and no tensors changed by this callback.
            return {}

        with progress.stage(f'Sample {sample_idx:06d}: generate six-camera videos'), torch.no_grad():
            output = pipeline(
                prompt,
                num_frames=video_length,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_videos_per_prompt=len(CAM_NAMES),
                control_video=input_video,
                depth_control_video=depth_input_video,
                depth_control_latent_weight=args.depth_control_latent_weight,
                control_camera_video=None,
                ref_image=ref_image,
                start_image=start_image,
                clip_image=clip_image,
                shift=args.shift,
                stage_callback=progress.detail,
                callback_on_step_end=report_step,
                callback_on_step_end_tensor_inputs=[],
            )

        with progress.stage(f'Sample {sample_idx:06d}: save videos'):
            progress.detail('Transfer generated videos to CPU')
            sample = output.videos.cpu()
            for view_idx, cam in enumerate(CAM_NAMES):
                progress.detail(f'Write camera {view_idx + 1}/6: {cam}')
                view_path = save_dir / f"{sample_idx:06d}_{cam}.mp4"
                save_videos_grid(sample[view_idx : view_idx + 1], str(view_path), fps=args.fps, n_rows=1)
                progress.info('Saved: %s', view_path)
            if args.save_grid:
                progress.detail('Write six-camera grid video')
                grid_path = save_dir / f"{sample_idx:06d}_grid.mp4"
                save_videos_grid(sample, str(grid_path), fps=args.fps, n_rows=3)
                progress.info('Saved: %s', grid_path)
        progress.info('Sample %d/%d completed: %06d', sample_order, len(local_records), sample_idx)


def main():
    args = parse_args()
    try:
        run_inference(args)
    except BaseException as error:
        progress.finish(error)
        raise
    else:
        progress.finish()


if __name__ == "__main__":
    main()
