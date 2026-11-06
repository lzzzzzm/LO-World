"""Print aligned terminal FVD scores and save a Markdown summary."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for frames, label in ((81, "81 frames"), (37, "First 37 frames")):
        path = args.out_dir / f"fvd_{frames}f.json"
        result = json.loads(path.read_text())
        score = float(result["FVD"])
        if result.get("frames") != frames or not math.isfinite(score):
            raise ValueError(f"Invalid FVD result: {path}")
        rows.append((label, f"{score:.6f}"))

    lines = [
        "## FVD summary (lower is better)",
        "",
        "| Video prefix | FVD |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {label} | {score} |" for label, score in rows)

    table = "\n".join(lines) + "\n"
    summary_path = args.out_dir / "fvd_summary.md"
    summary_path.write_text(table)
    print(f"Summary saved to {summary_path}\n", flush=True)

    label_width = max(len("Video prefix"), *(len(label) for label, _ in rows))
    score_width = max(len("FVD"), *(len(score) for _, score in rows))
    border = f"+-{'-' * label_width}-+-{'-' * score_width}-+"
    terminal_lines = [
        "FVD summary (lower is better)",
        border,
        f"| {'Video prefix':<{label_width}} | {'FVD':>{score_width}} |",
        border,
    ]
    terminal_lines.extend(
        f"| {label:<{label_width}} | {score:>{score_width}} |"
        for label, score in rows)
    terminal_lines.append(border)
    print("\n".join(terminal_lines), flush=True)


if __name__ == "__main__":
    main()
