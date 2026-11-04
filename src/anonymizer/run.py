"""Extend a split of STT4SG-350 with synthetic speakers made by the globi anonymizer.

    uv run anonymize                                         # every test speaker of data.split_dir
    uv run anonymize synth.max_speakers=2 synth.clips_per_speaker=5          # a quick look
    uv run anonymize stages=[pack]                           # only rebuild the dataset folder

Reads the source split with the `data` config (tsv files and the packed audio_16k.h5), and writes
`synth.out_dir` (default data/stt4sg-350_synth), which replaces the source data folder: train with
`data=stt4sg-350_synth` (configs/data/stt4sg-350_synth.yaml) and the test split is the synthetic one.
See anonymizer.synth for how speakers and donors are picked and anonymizer.pack for the layout.

Stages (`stages`): plan writes plan.tsv, generate calls globi for every planned clip not done yet
(a rerun resumes and retries failed clips), pack writes the tsv files and audio_16k.h5.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from anonymizer.globi import GlobiClient  # noqa: E402
from anonymizer.pack import pack  # noqa: E402
from anonymizer.synth import ClipReader, generate, plan_clips, plan_speakers, read_results, speaker_table  # noqa: E402
from maloney.data.stt4sg import read_h5_index, read_split  # noqa: E402

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")
STAGES = ("plan", "generate", "pack")
log = logging.getLogger(__name__)


def make_plan(cfg: DictConfig, meta: pd.DataFrame, donor_meta: pd.DataFrame) -> pd.DataFrame:
    sc = cfg.synth
    sources = speaker_table(meta)
    if sc.max_speakers is not None:
        # an even share of every dialect region, so a small run still covers all of them
        per_region = -(-sc.max_speakers // sources["dialect_region"].nunique())
        sources = (sources.sort_values("client_id").groupby("dialect_region").head(per_region)
                   .head(sc.max_speakers))
    speakers = plan_speakers(sources, speaker_table(donor_meta), sc.n_donors, cfg.seed)
    plan = plan_clips(meta, speakers, sc.clips_per_speaker, sc.max_target_seconds, cfg.seed)
    log.info("plan: %d source speakers x %d donors -> %d synthetic speakers, %d clips (%.1f h)", len(sources),
             sc.n_donors, plan["client_id"].nunique(), len(plan), plan["duration"].sum() / 3600)
    return plan


def packed_clips(meta: pd.DataFrame, packed: set[str], split: str, audio_h5: str) -> pd.DataFrame:
    """The clips of `meta` that are in the audio file; warns about the ones that are not."""
    in_file = meta["path"].isin(packed)
    if not in_file.all():
        lost = sorted(set(meta["client_id"]) - set(meta.loc[in_file, "client_id"]))
        log.warning("%s: %d clips are not in %s and are skipped; speakers without any audio there: %s",
                    split, (~in_file).sum(), audio_h5, lost or "none")
    return meta[in_file]


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="anonymize.yaml")
def main(cfg: DictConfig):
    if unknown := set(cfg.stages) - set(STAGES):
        raise ValueError(f"unknown stages {sorted(unknown)}, pick from {STAGES}")
    sc = cfg.synth
    out_dir = Path(sc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = cfg.data.split_dir
    meta = read_split(cfg.data.data_dir, sc.split, cfg.data.train_split, split_dir)
    donor_meta = meta if sc.donor_split == sc.split else read_split(
        cfg.data.data_dir, sc.donor_split, cfg.data.train_split, split_dir)
    # Only clips in the audio file can be re-spoken or lend their voice (e.g. a local file packed from
    # part of the release); speakers without any are left out of the plan.
    packed = set(read_h5_index(cfg.data.audio_h5)[0].index)
    meta, donor_meta = (packed_clips(m, packed, name, cfg.data.audio_h5)
                        for m, name in [(meta, sc.split), (donor_meta, sc.donor_split)])
    log.info("source: %s split of %s, %d clips of %d speakers", sc.split, split_dir or cfg.data.data_dir,
             len(meta), meta["client_id"].nunique())

    plan_file = out_dir / "plan.tsv"
    if "plan" in cfg.stages:
        plan = make_plan(cfg, meta, donor_meta)
        plan.to_csv(plan_file, sep="\t", index=False, quoting=3)
    elif not plan_file.is_file():
        raise FileNotFoundError(f"{plan_file} does not exist: run the plan stage first")
    plan = pd.read_csv(plan_file, sep="\t", quoting=3, dtype=str).astype({"duration": float})

    if "generate" in cfg.stages:
        gc = cfg.globi
        client = GlobiClient(gc.url, timeout=gc.timeout, retries=gc.retries, poll_interval=gc.poll_interval,
                             job_timeout=gc.job_timeout)
        generate(plan, donor_meta, ClipReader(cfg.data.audio_h5), client, out_dir, sc.reference_seconds,
                 gc.concurrency, cfg.seed, gc.max_consecutive_failures, gc.status_interval)
    if "pack" in cfg.stages:
        pack(cfg.data, plan, read_results(out_dir / "results.jsonl"), out_dir, sc.split, sc.include_original)


if __name__ == "__main__":
    main()
