import logging
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")
log = logging.getLogger(__name__)


def train(cfg: DictConfig):
    log.debug(cfg)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    log.info("dialect regions: %s", datamodule.vocabs["dialect_region"].itos)
    grid = hydra.utils.instantiate(cfg.geo.grid)
    mapping = hydra.utils.instantiate(cfg.geo.mapping)
    log.info("%s, mapping %s", grid, mapping)

    batch = next(iter(datamodule.train_dataloader()))
    log.info("train batch: audio %s, dialect %s", tuple(batch["audio"].shape), batch["dialect"].tolist())
    if "lv95" in batch:
        log.info("train batch: cells %s", grid.cell_of(batch["lv95"]).tolist())
    log.info('Done')


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="train.yaml")
def main(cfg: DictConfig):
    return train(cfg)


if __name__ == "__main__":
    main()
