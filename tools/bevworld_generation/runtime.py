"""Model configuration and runner construction for BEVWorld inference."""

from __future__ import annotations

import os.path as osp
import logging

from mmengine.config import Config
from mmengine.runner import Runner


class _RunnerStartupFilter(logging.Filter):
    """Keep verbose environment/config records in files, not on the console."""

    def filter(self, record):
        return record.funcName != '_log_env' or record.levelno >= logging.WARNING


class _QuietStartupRunner(Runner):
    def _log_env(self, env_cfg):
        # FileHandler inherits StreamHandler: explicitly exclude it so the
        # original MMEngine log still contains the full environment and config.
        handlers = [handler for handler in self.logger.handlers
                    if isinstance(handler, logging.StreamHandler)
                    and not isinstance(handler, logging.FileHandler)]
        startup_filter = _RunnerStartupFilter()
        for handler in handlers:
            handler.addFilter(startup_filter)
        try:
            super()._log_env(env_cfg)
        finally:
            for handler in handlers:
                handler.removeFilter(startup_filter)


def load_config(args):
    cfg = Config.fromfile(str(args.config))
    if getattr(args, "cfg_options", None) is not None:
        cfg.merge_from_dict(args.cfg_options)
    return cfg


def configure_checkpoint_inference(cfg, checkpoint):
    """Use a complete inference checkpoint without training initializers."""
    # Keep paths serializable by MMEngine's Config.pretty_text.
    cfg.load_from = str(checkpoint) if checkpoint is not None else None
    cfg.resume = False
    for key in ('vae_pretrained', 'world_pretrained'):
        if key in cfg.model:
            cfg.model[key] = None

    # This training hook creates a second model and can load a previous
    # stage's EMA checkpoint in before_run, even during Runner.test().
    if cfg.get('custom_hooks') is not None:
        hooks = []
        for hook in cfg.custom_hooks:
            hook_type = hook.get('type') if isinstance(hook, dict) else type(hook)
            name = (hook_type if isinstance(hook_type, str)
                    else getattr(hook_type, '__name__', ''))
            if name.rsplit('.', 1)[-1] != 'MEGVIIEMAHook':
                hooks.append(hook)
        cfg.custom_hooks = hooks
    return cfg


def build_runner(args, *, quiet_startup=False):
    """Build and load an inference-only MMEngine runner."""
    cfg = load_config(args)
    cfg.launcher = "none"
    configure_checkpoint_inference(cfg, args.checkpoint)
    cfg.work_dir = getattr(args, "runner_work_dir", None) or cfg.get(
        "work_dir", osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0]))

    runner_type = _QuietStartupRunner if quiet_startup else Runner
    runner = runner_type.from_cfg(cfg)
    runner.load_or_resume()
    runner.model.eval()
    return runner
