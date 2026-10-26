"""Minimal experiment loggers (scalars, audio, images, hparams) with one interface.

Stands in for the Lightning loggers of Matcha-TTS. Everything is also written to the hydra job
log by the trainer, so a logger is optional (`logger=none`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class TensorBoardLogger:
    def __init__(self, save_dir: str):
        from torch.utils.tensorboard import SummaryWriter  # pylint: disable=import-outside-toplevel

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(save_dir)

    def log_scalars(self, scalars: dict[str, float], step: int):
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, step)

    def log_audio(self, key: str, audio: np.ndarray, step: int, sample_rate: int):
        self.writer.add_audio(key, audio, step, sample_rate=sample_rate)

    def log_image(self, key: str, image: np.ndarray, step: int):
        self.writer.add_image(key, image, step, dataformats="HWC")

    def log_hparams(self, hparams: dict):
        self.writer.add_text("hparams", "```\n" + _to_yaml(hparams) + "\n```", 0)

    def close(self):
        self.writer.close()


class WandbLogger:
    def __init__(self, save_dir: str, project: str, id: str | None = None,
                 offline: bool = False, tags=None, entity: str | None = None):  # pylint: disable=redefined-builtin
        import os  # pylint: disable=import-outside-toplevel

        import wandb  # pylint: disable=import-outside-toplevel

        if not offline and not os.environ.get("WANDB_API_KEY") and not wandb.api.api_key:
            raise RuntimeError("no wandb API key: put WANDB_API_KEY=... into .env in the repo root, or use logger=tensorboard")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.wandb = wandb
        self.run = wandb.init(
            project=project, entity=entity, id=id, dir=save_dir, tags=list(tags or []),
            mode="offline" if offline else "online", resume="allow" if id else None,
        )
        log.info("wandb run: %s", self.run.url)

    def log_scalars(self, scalars: dict[str, float], step: int):
        self.run.log(scalars, step=step)

    def log_audio(self, key: str, audio: np.ndarray, step: int, sample_rate: int):
        self.run.log({key: self.wandb.Audio(audio, sample_rate=sample_rate)}, step=step)

    def log_image(self, key: str, image: np.ndarray, step: int):
        self.run.log({key: self.wandb.Image(image)}, step=step)

    def log_hparams(self, hparams: dict):
        self.run.config.update(hparams, allow_val_change=True)

    def close(self):
        self.run.finish()


class MultiLogger:
    """Fans every call out to a list of loggers; an empty list is a valid no-op logger."""

    def __init__(self, loggers):
        self.loggers = list(loggers)

    def __bool__(self):
        return bool(self.loggers)

    def log_scalars(self, scalars, step):
        for lg in self.loggers:
            lg.log_scalars(scalars, step)

    def log_audio(self, key, audio, step, sample_rate):
        for lg in self.loggers:
            lg.log_audio(key, audio, step, sample_rate)

    def log_image(self, key, image, step):
        for lg in self.loggers:
            lg.log_image(key, image, step)

    def log_hparams(self, hparams):
        for lg in self.loggers:
            lg.log_hparams(hparams)

    def close(self):
        for lg in self.loggers:
            lg.close()


def _to_yaml(obj) -> str:
    from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel

    return OmegaConf.to_yaml(OmegaConf.create(obj))


def build_logger(cfg) -> MultiLogger:
    """Instantiate every logger under `cfg.logger` (none, one or several) and log the config."""
    import hydra  # pylint: disable=import-outside-toplevel
    from omegaconf import DictConfig, OmegaConf  # pylint: disable=import-outside-toplevel

    loggers = []
    for _, lg_cfg in (cfg.get("logger") or {}).items():
        if isinstance(lg_cfg, DictConfig) and "_target_" in lg_cfg:
            log.info("Instantiating logger <%s>", lg_cfg._target_)  # pylint: disable=protected-access
            loggers.append(hydra.utils.instantiate(lg_cfg))
    logger = MultiLogger(loggers)
    logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))
    return logger
