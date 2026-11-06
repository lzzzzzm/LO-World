"""W-CODA FVD adapter for LO-World clips and flat generated-video paths.

Imports W-CODA's original transform and VideoGPT implementation. GT is read
from original images; prediction videos are decoded with MoviePy as in W-CODA.
Only clip selection, temporal-prefix selection, batching and paths are adapted.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .fvd_progress import EvaluationProgress
else:
    from fvd_progress import EvaluationProgress

CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
           "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "work_dirs/genie_inference_val150_81f/eval/wcoda_fp64_batch50/assets"
PREDICTIONS = ROOT / "work_dirs/genie_inference_val150_81f"


def prepare(out, dataset_dir, predictions, nuscenes_root, meta_path, progress):
    with (dataset_dir / "selected_12hz_frames.pkl").open("rb") as stream:
        clips = pickle.load(stream)["clips"]
    metadata = json.loads(meta_path.read_text())
    metadata_indices = [int(row["_clip_index"]) for row in metadata]
    expected_indices = sorted(int(clip["record"]["clip_index"]) for clip in clips)
    if metadata_indices != expected_indices:
        raise RuntimeError("meta.json order/index does not match selected_12hz_frames.pkl")
    rows = []
    for clip in sorted(clips, key=lambda c: int(c["record"]["clip_index"])):
        index = int(clip["record"]["clip_index"])
        assert len(clip["frames"]) == 81
        for camera in CAMERAS:
            paths = []
            for frame in clip["frames"]:
                info = frame["images"][camera]
                raw = Path(info.get("img_path", info.get("data_path")))
                candidates = [raw, ROOT / raw, nuscenes_root / raw]
                if raw.parts[:2] == ("data", "nuscenes"):
                    candidates.append(nuscenes_root / Path(*raw.parts[2:]))
                path = next((p for p in candidates if p.is_file()), None)
                if path is None:
                    raise FileNotFoundError(raw)
                paths.append(str(path))
            pred = predictions / f"{index:06d}_{camera}.mp4"
            assert pred.is_file(), pred
            try:
                prediction_path = str(pred.relative_to(ROOT))
            except ValueError:
                prediction_path = str(pred)
            rows.append({"index": index, "camera": camera,
                         "gt_images": paths, "prediction": prediction_path})
    assert len(rows) == 900
    manifest = out / "inputs.json"
    has_cache = any(out.glob("features_*f/*.npy"))
    if has_cache:
        if not manifest.exists() or json.loads(manifest.read_text()) != rows:
            raise RuntimeError("Input data differs from the cached run; use a new --out-dir")
        for result_file in out.glob("fvd_*f.json"):
            result = json.loads(result_file.read_text())
            if result.get("extraction_batch_size") != 50:
                raise RuntimeError("Cache was not recorded with batch size 50; use a new --out-dir")
    manifest.write_text(json.dumps(rows))
    progress.info("Validated %d videos and all original GT image paths; cached features: %s",
                  len(rows), "present" if has_cache else "none")


class Clips(Dataset):
    def __init__(self, rows, kind, frames, transform):
        self.rows, self.kind, self.frames, self.transform = rows, kind, frames, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = []
        if self.kind == "gt":
            for path in row["gt_images"][:self.frames]:
                with Image.open(path) as image:
                    images.append(self.transform(image.convert("RGB")))
        else:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(row["prediction"], audio=False) as clip:
                for frame in clip.iter_frames():
                    images.append(self.transform(Image.fromarray(frame).convert("RGB")))
                    if len(images) == self.frames:
                        break
        if len(images) != self.frames:
            raise ValueError(f"Expected {self.frames} frames: {row['prediction']}")
        return torch.stack(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wcoda-root", type=Path,
                        default=ROOT / "externals/W-CODA2024-Track2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--pred-dir", type=Path, default=PREDICTIONS)
    parser.add_argument("--meta-path", type=Path)
    parser.add_argument("--nuscenes-root", type=Path, default=ROOT / "data/nuscenes")
    parser.add_argument("--i3d-weights-dir", type=Path, default=ROOT / "ckpts/fvd")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--frames", type=int, choices=[37, 81], default=81)
    parser.add_argument("--views", nargs="+", default=CAMERAS)
    parser.add_argument("--batch-size", type=int, choices=[50], default=50,
                        help="Fixed to the upstream W-CODA batch size")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-label", help="Label for concurrent worker log lines")
    args = parser.parse_args()
    progress = EvaluationProgress(args.log_label or (
        "prepare" if args.prepare else
        f"{args.frames}f aggregate" if args.aggregate else
        f"{args.frames}f GPU={os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}"))
    args.dataset_dir = args.dataset_dir.resolve()
    args.pred_dir = args.pred_dir.resolve()
    # Keep the repository-local data symlink visible in cache manifests; resolving
    # it would serialize the machine-specific external mount path.
    if not args.nuscenes_root.is_absolute():
        args.nuscenes_root = ROOT / args.nuscenes_root
    args.i3d_weights_dir = args.i3d_weights_dir.resolve()
    if args.meta_path is None:
        args.meta_path = args.dataset_dir / "genie_dataset/meta.json"
    args.meta_path = args.meta_path.resolve()
    if not args.meta_path.is_file():
        raise FileNotFoundError(args.meta_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        with progress.stage("validate videos, GT image paths and cache inputs"):
            prepare(args.out_dir, args.dataset_dir, args.pred_dir,
                    args.nuscenes_root, args.meta_path, progress)
        return
    sys.path.insert(0, str(args.wcoda_root))
    with progress.stage("load W-CODA evaluation dependencies"):
        from fvd.videogpt.fvd import frechet_distance, get_fvd_logits, load_i3d_pretrained
    cache = args.out_dir / f"features_{args.frames}f"
    cache.mkdir(exist_ok=True)
    rows = json.loads((args.out_dir / "inputs.json").read_text())
    if any(Path(row["prediction"]).resolve() != args.pred_dir /
           f"{row['index']:06d}_{row['camera']}.mp4" for row in rows):
        raise RuntimeError("Prediction directory differs from inputs.json; use a new --out-dir")
    if args.aggregate:
        features = []
        with progress.stage("load and combine six-camera feature caches"):
            for kind in ("gt", "pred"):
                arrays = [np.load(cache / f"{kind}_{cam}.npy") for cam in CAMERAS]
                assert all(a.shape == (150, 400) and a.dtype == np.float64
                           and np.isfinite(a).all() for a in arrays)
                combined = np.concatenate(arrays)
                np.save(args.out_dir / f"{kind}_{args.frames}f.npy", combined)
                features.append(torch.from_numpy(combined))
                progress.info("Combined %s features: shape=%s", kind, combined.shape)
        with progress.stage("compute Frechet distance"):
            score = frechet_distance(*features)
        result = {"metric": "W-CODA VideoGPT I3D FVD", "FVD": score,
                  "frames": args.frames, "clips": 150, "cameras": 6,
                  "features_per_distribution": [900, 400], "feature_dtype": "float64",
                  "i3d_inference_dtype": "float32", "distance_dtype": "float64",
                  "extraction_batch_size": args.batch_size,
                  "generation_resolution_hw": [448, 800], "resize_hw": [225, 400],
                  "crop_hw": [224, 400], "gt_source": "original nuScenes image files",
                  "prediction_decoder": "MoviePy VideoFileClip.iter_frames",
                  "aggregation": "pool all six cameras then compute one Frechet distance",
                  "protocol": "wcoda_videogpt_fp64_v1",
                  "wcoda_source": str(args.wcoda_root),
                  "prediction_dir": str(args.pred_dir.relative_to(ROOT)
                                        if args.pred_dir.is_relative_to(ROOT)
                                        else args.pred_dir)}
        result_path = args.out_dir / f"fvd_{args.frames}f.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        progress.info("FVD (%df): %.6f; result: %s", args.frames, score, result_path)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for W-CODA feature extraction")
    # Same top-center crop used by W-CODA's get_fvd_features_gen.py.
    def top_center_crop(image, target_size):
        target_h, target_w = target_size
        width, height = image.size
        crop_top = height - target_h
        crop_left = int(max(0, width - target_w) / 2)
        return image.crop((crop_left, crop_top, crop_left + target_w, crop_top + target_h))

    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((225, 400), interpolation=transforms.InterpolationMode.BICUBIC),
        lambda image: top_center_crop(image, (224, 400)),
        transforms.ToTensor(),
    ])
    with progress.stage("load I3D model on CUDA"):
        model = load_i3d_pretrained(device=torch.device("cuda"), load_from=str(args.i3d_weights_dir))
    assert next(model.parameters()).dtype == torch.float32
    cached_count = 0
    extracted_count = 0
    for camera in args.views:
        selected = sorted((r for r in rows if r["camera"] == camera), key=lambda r: r["index"])
        assert len(selected) == 150
        for kind in ("gt", "pred"):
            path = cache / f"{kind}_{camera}.npy"
            if path.exists():
                cached_count += 1
                progress.info("Cache hit: %s %s; skipping extraction (%s)", camera, kind, path)
                continue
            loader = DataLoader(Clips(selected, kind, args.frames, transform),
                                batch_size=args.batch_size, num_workers=args.workers,
                                shuffle=False, drop_last=False)
            features = np.empty((150, 400), dtype=np.float64)
            start = 0
            with progress.stage(f"extract {camera} {kind} features (150 clips)"):
                started = time.monotonic()
                for batch_index, batch in enumerate(loader, 1):
                    with torch.no_grad():
                        values = get_fvd_logits(batch.permute(0, 2, 1, 3, 4), model, "cuda").cpu().numpy()
                    features[start:start + len(values)] = values
                    start += len(values)
                    elapsed = time.monotonic() - started
                    progress.info(
                        "%s %s: batch %d/%d; clips %d/150; elapsed %.1fs; ETA %.1fs",
                        camera, kind, batch_index, len(loader), start,
                        elapsed, elapsed / start * (150 - start))
                assert start == 150 and np.isfinite(features).all()
                np.save(path, features)
                extracted_count += 1
                progress.info("Saved features: %s", path)
    progress.info("Extraction finished: %d saved, %d reused from cache",
                  extracted_count, cached_count)


if __name__ == "__main__":
    main()
