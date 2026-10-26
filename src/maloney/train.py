"""M3: fine-tune a Whisper encoder on grid cells, then predict with cell prototypes.

1. Classes are the centroid cells of the training municipalities; every training unit (clip, or
   segment with data.segment_duration) is labelled with its municipality's centroid cell.
2. Training: cross-entropy over the classes from a linear head on the mean-pooled encoder output,
   AdamW with linear warmup and decay, mixed precision on CUDA. After every epoch the head is
   evaluated on valid; checkpoints/last.pt (full state, for resuming via ckpt_path) and
   checkpoints/best.pt (weights, by `monitor`) are written.
3. Inference as in the paper: with the best weights, the train, valid and test units are re-embedded,
   cell prototypes are built in every mapping mode and each unit gets its nearest prototype (B2 on
   fine-tuned embeddings). The head's own predictions are reported as well. Metrics go to
   metrics.json, predictions to predictions_<split>_<level>_<mode>.csv.
"""

import json
import logging
import math
import time
from pathlib import Path

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from transformers import get_linear_schedule_with_warmup  # noqa: E402

from maloney.baselines.evaluate import GeoContext, evaluate  # noqa: E402
from maloney.baselines.methods import NearestPrototype  # noqa: E402
from maloney.baselines.run import evaluate_method, log_results  # noqa: E402
from maloney.baselines.units import unit_table  # noqa: E402
from maloney.geo import Geolects  # noqa: E402
from maloney.models.whisper_geo import WhisperCollate, WhisperGeoClassifier, class_index  # noqa: E402
from maloney.training.loggers import build_logger  # noqa: E402

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")
log = logging.getLogger(__name__)


class HeadMethod:
    """The fine-tuned classification head as a method for evaluate_method: its input is the head's
    logits, its scores the class probabilities."""

    class_regions = None

    def __init__(self, classes: np.ndarray):
        self.class_cells = classes

    def scores(self, logits: np.ndarray) -> np.ndarray:
        return torch.softmax(torch.as_tensor(logits), dim=1).numpy()


def autocast_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    if precision == "auto":
        if device.type != "cuda":
            return None
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


@torch.no_grad()
def run_model(model, loader, device, dtype, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pooled embeddings and head logits of every item of an (ordered) loader."""
    model.eval()
    embeddings, logits = [], []
    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            break
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            emb = model.embed(batch)
        embeddings.append(emb.float().cpu())
        logits.append(model.head(emb.float()).cpu())
    return torch.cat(embeddings).numpy(), torch.cat(logits).numpy()


def target_classes(batch: dict, grid, classes: np.ndarray) -> torch.Tensor:
    """Class of every item: its municipality's centroid cell (-1 if that cell is no class)."""
    return torch.as_tensor(class_index(grid.cell_of(batch["lv95"].numpy()), classes))


def train(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.paths.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tc = cfg.trainer

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    if datamodule.geo_table is None:
        raise ValueError("M3 needs speaker locations: set data.geo_table")
    unit = "segment" if datamodule.segment_duration else "clip"
    grid = hydra.utils.instantiate(cfg.geo.grid)
    geolects = Geolects.from_meta(datamodule.datasets["train"].meta)
    geo = GeoContext(grid, hydra.utils.instantiate(cfg.geo.mapping), geolects)
    classes = np.unique(grid.cell_of(geolects.xy))
    log.info("%s; %d training municipalities -> %d classes (centroid cells); unit %s", grid, len(geolects),
             len(classes), unit)

    device = torch.device(tc.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = autocast_dtype(device, tc.precision)
    mc = cfg.model
    model = WhisperGeoClassifier(mc.name, len(classes), mc.pool_layer, mc.freeze_layers, mc.dropout,
                                 mc.gradient_checkpointing and device.type == "cuda").to(device)
    total, trainable = model.n_parameters()
    log.info("%s pooled at layer %d: %.1fM parameters, %.1fM trainable; %s, autocast %s", mc.name,
             model.pool_layer, total / 1e6, trainable / 1e6, device, dtype)

    collate = WhisperCollate(mc.name)
    train_loader = datamodule.dataloader("train", collate_fn=collate, batch_size=tc.batch_size)
    eval_loader = lambda split: datamodule.dataloader(  # noqa: E731
        split, shuffle=False, collate_fn=collate, batch_size=tc.eval_batch_size or tc.batch_size)
    valid_units = unit_table(datamodule, "valid", unit, shuffle=False)[0]

    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.optim.weight_decay},
              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=cfg.optim.lr)
    batches_per_epoch = min(len(train_loader), tc.limit_train_batches or math.inf)
    steps_per_epoch = max(1, batches_per_epoch // tc.accumulate)
    total_steps = steps_per_epoch * tc.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.optim.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)

    sign = 1 if cfg.monitor_mode == "max" else -1
    start_epoch, step, best = 0, 0, -math.inf
    if cfg.ckpt_path:
        state = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        if not np.array_equal(state["classes"], classes):
            raise ValueError(f"{cfg.ckpt_path} was trained on other classes (grid or split changed)")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step, best = state["epoch"] + 1, state["step"], state["best"]
        if hasattr(train_loader.sampler, "epoch"):  # segment grouping continues where it stopped
            train_loader.sampler.epoch = start_epoch
        log.info("resumed from %s after epoch %d (step %d)", cfg.ckpt_path, state["epoch"] + 1, step)

    logger = build_logger(cfg)
    for epoch in range(start_epoch, tc.epochs):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            if i >= batches_per_epoch:
                break
            targets = target_classes(batch, grid, classes).to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits.float(), targets, ignore_index=-1)
            scaler.scale(loss / tc.accumulate).backward()
            running, seen = running + loss.item() * len(targets), seen + len(targets)
            if (i + 1) % tc.accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1
                if step % tc.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    log.info("epoch %d step %d/%d: loss %.4f, lr %.2e, %.1f items/s", epoch + 1, step, total_steps,
                             running / seen, lr, seen / (time.time() - start))
                    logger.log_scalars({"train/loss": running / seen, "train/lr": lr}, step)

        _, logits = run_model(model, eval_loader("valid"), device, dtype, tc.limit_eval_batches)
        units = valid_units.iloc[: len(logits)]
        targets = target_classes({"lv95": torch.as_tensor(units[["muni_E", "muni_N"]].to_numpy())}, grid, classes)
        metrics, _ = evaluate(classes[logits.argmax(1)], units, geo)
        # Valid speakers mostly live in municipalities that are no class (speaker-disjoint split): the
        # loss only covers those that are, and is left out if there are none.
        known = targets >= 0
        metrics["known_class"] = float(known.float().mean())
        if known.any():
            metrics["loss"] = torch.nn.functional.cross_entropy(torch.as_tensor(logits)[known], targets[known]).item()
        log.info("epoch %d done in %.0f min: train loss %.4f, valid %s", epoch + 1, (time.time() - start) / 60,
                 running / max(seen, 1), "  ".join(f"{k} {v:.3f}" for k, v in metrics.items() if k != "n"))
        logger.log_scalars({f"valid/{k}": v for k, v in metrics.items()} | {"epoch": epoch + 1}, step)

        value = sign * metrics[cfg.monitor]
        if value > best:
            best = value
            torch.save({"model": model.state_dict(), "classes": classes, "epoch": epoch,
                        "config": OmegaConf.to_container(cfg, resolve=True)}, ckpt_dir / "best.pt")
            log.info("new best valid %s %.4f -> %s", cfg.monitor, metrics[cfg.monitor], ckpt_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "classes": classes,
                    "epoch": epoch, "step": step, "best": best}, ckpt_dir / "last.pt")

    results = prototype_inference(cfg, datamodule, model, grid, geolects, classes, unit, device, dtype, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    logger.log_scalars({f"final/{key}/{k}": v for key, metrics in results.items() for k, v in metrics.items()}, step)
    logger.close()
    return results


def prototype_inference(cfg, datamodule, model, grid, geolects, classes, unit, device, dtype, out_dir) -> dict:
    """Re-embed all units with the best weights and evaluate nearest prototypes in every mode + the head."""
    best = out_dir / "checkpoints" / "best.pt"
    if best.is_file():
        model.load_state_dict(torch.load(best, map_location=device, weights_only=False)["model"])
        log.info("prototype inference with %s", best)
    tc = cfg.trainer
    collate = WhisperCollate(cfg.model.name)
    data, logits = {}, {}
    for split in ["train", *cfg.eval_splits]:
        loader = datamodule.dataloader(split, shuffle=False, collate_fn=collate,
                                       batch_size=tc.eval_batch_size or tc.batch_size)
        x, split_logits = run_model(model, loader, device, dtype, tc.limit_eval_batches)
        units = unit_table(datamodule, split, unit, shuffle=False)[0].iloc[: len(x)]
        data[split], logits[split] = (x, units), (split_logits, units)
        log.info("embedded %d %s units of %s", len(x), unit, split)

    x_train, train_units = data.pop("train")
    results = {}
    for mode in cfg.prototype_modes:
        mapping = hydra.utils.instantiate(cfg.geo.mapping, mode=mode)
        geo = GeoContext(grid, mapping, geolects)
        method = NearestPrototype(center=cfg.center_embeddings)
        method.fit(x_train, train_units, geo)
        for key, metrics in evaluate_method(method, geo, data, unit, out_dir, tag=f"_{mode}").items():
            results[f"prototype_{mode}/{key}"] = metrics
    geo = GeoContext(grid, hydra.utils.instantiate(cfg.geo.mapping), geolects)
    logits.pop("train")
    for key, metrics in evaluate_method(HeadMethod(classes), geo, logits, unit, out_dir, tag="_head").items():
        results[f"head/{key}"] = metrics
    log_results(results)
    return results


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="train.yaml")
def main(cfg: DictConfig):
    return train(cfg)


if __name__ == "__main__":
    main()
