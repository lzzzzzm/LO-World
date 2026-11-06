#!/usr/bin/env python3
"""Keep only the WorldBench metrics used by the LO-World protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--detection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_last_json(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No metrics found in {path}")
    return json.loads(lines[-1])


def main():
    args = parse_args()
    segmentation = load_last_json(args.segmentation)
    detection = json.loads(args.detection.read_text())
    result = {
        "mIoU": float(segmentation["mIoU"]),
        "mAP": float(detection["mean_ap"]),
        "NDS": float(detection["nd_score"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Evaluation produced non-finite metrics")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    rows = [(name, f"{result[name]:.6f}") for name in ("mAP", "mIoU", "NDS")]
    summary_path = args.output.with_suffix(".md")
    summary_path.write_text(
        "## WorldBench results (higher is better)\n\n"
        "| Metric | Score |\n| --- | ---: |\n"
        + "".join(f"| {name} | {value} |\n" for name, value in rows),
        encoding="utf-8")
    print(f"Metrics saved to {args.output}\nSummary saved to {summary_path}\n", flush=True)
    name_width = max(len("Metric"), *(len(name) for name, _ in rows))
    score_width = max(len("Score"), *(len(value) for _, value in rows))
    border = f"+-{'-' * name_width}-+-{'-' * score_width}-+"
    lines = ["WorldBench results (higher is better)", border,
             f"| {'Metric':<{name_width}} | {'Score':>{score_width}} |", border]
    lines.extend(f"| {name:<{name_width}} | {value:>{score_width}} |" for name, value in rows)
    lines.append(border)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
