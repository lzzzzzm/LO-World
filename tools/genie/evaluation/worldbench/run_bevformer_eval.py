"""Add stage logs to the upstream BEVFormer evaluation entrypoint."""

import argparse
import functools
import importlib.util
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fvd_progress import EvaluationProgress


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--test-script", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    rank = int(os.environ.get("RANK", "0"))
    progress = EvaluationProgress("BEVFormer")

    def stage(name):
        return progress.stage(name) if rank == 0 else nullcontext()

    test_script = args.test_script.resolve()
    sys.path.insert(0, str(test_script.parents[1]))
    sys.argv = [str(test_script), *forwarded]
    with stage("load BEVFormer dependencies"):
        spec = importlib.util.spec_from_file_location("_loworld_bevformer_eval", test_script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    active_stage = ["evaluation"]

    def logged_call(name, function):
        @functools.wraps(function)
        def wrapped(*call_args, **kwargs):
            previous = active_stage[0]
            active_stage[0] = name
            try:
                with stage(name):
                    return function(*call_args, **kwargs)
            finally:
                active_stage[0] = previous
        return wrapped

    original_build_dataset = module.build_dataset

    def build_dataset(*call_args, **kwargs):
        with stage("build evaluation dataset"):
            dataset = original_build_dataset(*call_args, **kwargs)
        if rank == 0:
            progress.info("Evaluation dataset: %d frames", len(dataset))
            dataset.evaluate = logged_call("compute detection metrics", dataset.evaluate)
        return dataset

    module.build_dataset = build_dataset
    for name, label in (
        ("init_dist", "initialize distributed evaluation"),
        ("build_dataloader", "build evaluation data loader"),
        ("build_model", "build BEVFormer model"),
        ("load_checkpoint", "load BEVFormer checkpoint"),
        ("multi_gpu_test", "run inference, segmentation metrics and result collection"),
        ("single_gpu_test", "run inference and segmentation metrics"),
    ):
        setattr(module, name, logged_call(label, getattr(module, name)))

    import mmcv
    original_progress_bar = mmcv.ProgressBar
    progress_module = sys.modules[original_progress_bar.__module__]

    class LineProgress(original_progress_bar):
        def __init__(self, task_num=0, bar_width=50, start=True, file=None, **kwargs):
            self.task_num = task_num
            self.completed = 0
            self.bar_width = bar_width
            self.file = file if file is not None else sys.stdout
            self.label = active_stage[0]
            self._reported = 0
            self._last_report = time.monotonic()
            self.started = self._last_report
            if start:
                self.start()

        def start(self):
            self.started = time.monotonic()
            self.timer = mmcv.Timer()

        def update(self, num_tasks=1):
            self.completed += num_tasks
            completed = min(self.completed, self.task_num) if self.task_num else self.completed
            if rank != 0 or (self.task_num and self._reported >= self.task_num):
                return
            now = time.monotonic()
            if (self._reported == 0 or completed - self._reported >= 50
                    or now - self._last_report >= 30
                    or (self.task_num and completed == self.task_num)):
                elapsed = now - self.started
                remaining = elapsed / completed * max(0, self.task_num - completed) if completed else 0
                progress.info("%s: %d/%d frames; elapsed %.1fs; ETA %.1fs",
                              self.label, completed, self.task_num, elapsed, remaining)
                self._reported = completed
                self._last_report = now

    # Replace display only; retain the upstream evaluation and distributed flow.
    mmcv.ProgressBar = LineProgress
    mmcv.utils.ProgressBar = LineProgress
    progress_module.ProgressBar = LineProgress
    import tqdm
    original_tqdm = tqdm.tqdm

    class QuietTqdm(original_tqdm):
        def __init__(self, *call_args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*call_args, **kwargs)

    tqdm.tqdm = QuietTqdm
    module.torch.multiprocessing.set_start_method("fork")
    module.main()


if __name__ == "__main__":
    from torch.distributed.elastic.multiprocessing.errors import record
    record(main)()
