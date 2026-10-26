"""Run one baseline (B1, B2, M1, M2) on precomputed embeddings and evaluate it.

    uv run baseline method=b2 geo.mapping.mode=uniform data.split_dir=<split> embeddings.layer=16

Fits the method on the training units, then predicts valid and test units and reports the metrics
of maloney.baselines.evaluate, per unit and per speaker (unit scores averaged over the speaker).
Writes metrics.json and predictions_<split>.csv to the run directory.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from maloney.baselines.evaluate import GeoContext, evaluate  # noqa: E402
from maloney.baselines.units import build_units, speaker_scores  # noqa: E402
from maloney.embeddings import EmbeddingStore  # noqa: E402
from maloney.geo import Geolects  # noqa: E402
from maloney.training.loggers import build_logger  # noqa: E402

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")
log = logging.getLogger(__name__)


def predict(method, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    best = scores.argmax(1)
    regions = method.class_regions[best] if method.class_regions is not None else None
    return method.class_cells[best], regions


def evaluate_method(method, geo: GeoContext, units_by_split: dict[str, tuple[np.ndarray, "pd.DataFrame"]],
                    unit: str, out_dir: Path, tag: str = "") -> dict[str, dict]:
    """Score and evaluate every split's units with a fitted method, per unit and per speaker.

    Writes predictions_<split>_<level><tag>.csv to `out_dir`; returns {"<split>/<level>": metrics}.
    """
    results = {}
    for split, (x, units) in units_by_split.items():
        scores = method.scores(x)
        levels = [(unit, scores, units)]
        if unit != "speaker":
            levels.append(("speaker", *speaker_scores(scores, units)))
        for level, level_scores, level_units in levels:
            cells, regions = predict(method, level_scores)
            results[f"{split}/{level}"], predictions = evaluate(cells, level_units, geo, regions)
            predictions.to_csv(out_dir / f"predictions_{split}_{level}{tag}.csv", index=False)
    return results


def log_results(results: dict[str, dict]):
    for key, metrics in results.items():
        log.info("%-24s %s", key, "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                            for k, v in metrics.items()))


def run(cfg: DictConfig) -> dict:
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    if datamodule.geo_table is None:
        raise ValueError("baselines need speaker locations: set data.geo_table")
    store = EmbeddingStore(cfg.embeddings.path)
    log.info("%s, using layer %d", store, cfg.embeddings.layer)

    grid = hydra.utils.instantiate(cfg.geo.grid)
    mapping = hydra.utils.instantiate(cfg.geo.mapping)
    geo = GeoContext(grid, mapping, Geolects.from_meta(datamodule.datasets["train"].meta))
    log.info("%s, %s, %d training municipalities on %d cells", grid, mapping, len(geo.geolects),
             int((geo.weights > 0).any(0).sum()))

    x, units = build_units(datamodule, "train", store, cfg.embeddings.layer, cfg.unit)
    method = hydra.utils.instantiate(cfg.method)
    start = time.time()
    log.info("fitting %s on %d %s units", type(method).__name__, len(units), cfg.unit)
    method.fit(x, units, geo)
    log.info("fitted in %.0f s: %d classes", time.time() - start, len(method.class_cells))

    out_dir = Path(cfg.paths.output_dir)
    units_by_split = {split: build_units(datamodule, split, store, cfg.embeddings.layer, cfg.unit)
                      for split in cfg.eval_splits}
    results = evaluate_method(method, geo, units_by_split, cfg.unit, out_dir)
    log_results(results)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    logger = build_logger(cfg)
    logger.log_scalars({f"{key}/{k}": v for key, metrics in results.items() for k, v in metrics.items()}, step=0)
    logger.close()
    return results


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="baseline.yaml")
def main(cfg: DictConfig):
    log.debug(OmegaConf.to_yaml(cfg))
    return run(cfg)


if __name__ == "__main__":
    main()
