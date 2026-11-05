"""Fill up every zip code (PLZ) of STT4SG-350 with synthetic speakers made by the globi anonymizer.

    uv run anonymize-zip                                          # plan, generate and pack
    uv run anonymize-zip stages=[plan]                            # only look at the plan
    uv run anonymize-zip stages=[pack]                            # only rebuild the dataset folder

Most PLZs have a single speaker, so a PLZ centroid is mostly a speaker embedding. Here every PLZ gets
at least `min_per_gender` female and as many male speakers, and the smaller gender is filled up to the
larger one, so both are equally represented. A new speaker of a PLZ re-speaks `clips_per_speaker` clips
of one of the PLZ's own speakers (its words and dialect, round robin over them) in the voice of a donor
of the needed gender: from another dialect region, from the same split of `data.split_dir` as the
source (so the splits stay speaker-disjoint in voice too), not yet a voice of that PLZ, and the least
used donor first so the voices spread over the PLZs.

The original speakers keep `clips_per_speaker` random clips each (all of them if they have fewer), so
every speaker of the dataset weighs the same. `synth.out_dir` (default data/stt4sg-350_synth_zip) holds
the release's layout (train_all.tsv, valid.tsv, test.tsv, by the release split of the source speaker)
and the split of `data.split_dir` (splits/<name>/), see write_dataset; audio_16k.h5 links to the
source's audio as in anonymizer.pack.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path, PurePosixPath

import hydra
import numpy as np
import pandas as pd
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from anonymizer.globi import GlobiClient  # noqa: E402
from anonymizer.pack import PROVENANCE, pack_h5, synthetic_meta, write_tsv  # noqa: E402
from anonymizer.synth import LANGUAGES, NAMESPACE, ClipReader, generate, read_results, speaker_rng, speaker_table  # noqa: E402
from maloney.data.meta import SPLITS, read_h5_index, read_split, read_tsv  # noqa: E402

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")
STAGES = ("plan", "generate", "pack")
RELEASE_FILES = ("train_all", "valid", "test")
GENDERS = ("female", "male")
log = logging.getLogger(__name__)


def read_sources(data_cfg: DictConfig, packed: set[str]) -> pd.DataFrame:
    """All clips of the release with the speaker's release file (`release`) and split of split_dir (`split`)."""
    release = pd.concat([read_tsv(Path(data_cfg.data_dir) / f"{name}.tsv").assign(release=name)
                         for name in RELEASE_FILES], ignore_index=True).drop_duplicates("path")
    split_of = pd.concat([read_split(data_cfg.data_dir, s, data_cfg.train_split, data_cfg.split_dir)
                          .drop_duplicates("client_id").set_index("client_id").assign(split=s)["split"]
                          for s in SPLITS])
    meta = release.assign(split=release["client_id"].map(split_of))
    if lost := sorted(meta.loc[meta["split"].isna(), "client_id"].unique()):
        log.warning("%d speakers are in no split of %s and are left out: %s", len(lost), data_cfg.split_dir, lost)
    meta = meta[meta["split"].notna() & meta["path"].isin(packed)]
    log.info("source: %d clips of %d speakers in %d zip codes", len(meta), meta["client_id"].nunique(),
             meta["zipcode"].nunique())
    return meta.reset_index(drop=True)


def plan_zip_speakers(speakers: pd.DataFrame, eligible: pd.Series, min_per_gender: int,
                      clips_per_speaker: int, seed: int) -> pd.DataFrame:
    """The synthetic speakers that fill up every zip code (module doc).

    `speakers` is a speaker table (speaker_table) with `split`; `eligible` the number of clips per
    speaker that globi can re-speak.
    """
    if unknown := set(speakers["dialect_region"]) - set(LANGUAGES):
        raise ValueError(f"no globi language for the dialect regions {sorted(unknown)}")
    speakers = speakers.assign(eligible=speakers["client_id"].map(eligible).fillna(0).astype(int))
    used = pd.Series(0, index=speakers["client_id"])  # times a speaker lent its voice so far
    rows = []
    for zipcode, local in speakers.sort_values("client_id").groupby("zipcode", sort=True):
        counts = local["gender"].value_counts().reindex(GENDERS, fill_value=0)
        target = max(min_per_gender, counts.max())
        needed = [g for g in GENDERS for _ in range(target - counts[g])]
        # sources with enough clips first, then round robin
        sources = (local[local["eligible"] > 0].assign(rank=lambda d: -d["eligible"].clip(upper=clips_per_speaker))
                   .sort_values(["rank", "client_id"]))
        if sources.empty:
            log.warning("zip %s: no speaker with clips globi can re-speak", zipcode)
            continue
        rng = speaker_rng(seed, f"zip/{zipcode}")
        voices = set(local["client_id"])
        for i, gender in enumerate(rng.permutation(needed).tolist()):
            x = sources.iloc[i % len(sources)]
            pool = speakers[(speakers["gender"] == gender) & (speakers["split"] == x["split"])
                            & (speakers["dialect_region"] != x["dialect_region"]) & ~speakers["client_id"].isin(voices)]
            if pool.empty:
                log.warning("zip %s: no %s donor left in the %s split", zipcode, gender, x["split"])
                continue
            # least used donor first, ties broken at random
            order = np.lexsort((rng.random(len(pool)), used.loc[pool["client_id"]].to_numpy()))
            donor = pool.iloc[order[0]]
            voices.add(donor["client_id"])
            used[donor["client_id"]] += 1
            rows.append({
                "client_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{NAMESPACE}_zip/{x['client_id']}/{donor['client_id']}")),
                "split": x["split"], "release": x["release"],
                "source_client_id": x["client_id"], "donor_client_id": donor["client_id"],
                "dialect_region": x["dialect_region"], "canton": x["canton"], "zipcode": zipcode,
                "age": donor["age"], "gender": gender,
                "donor_dialect_region": donor["dialect_region"], "donor_canton": donor["canton"],
                "donor_zipcode": donor["zipcode"], "language": LANGUAGES[x["dialect_region"]],
            })
    log.info("%d synthetic speakers for %d zip codes; %d donors, each lent its voice %d to %d times",
             len(rows), speakers["zipcode"].nunique(), (used > 0).sum(), used[used > 0].min(), used.max())
    return pd.DataFrame(rows)


def sample_clips(clips: pd.DataFrame, n: int, key: str, seed: int) -> pd.DataFrame:
    """`n` random clips (all if fewer), drawn with the rng of `key`."""
    clips = clips.sort_values("path")
    if len(clips) > n:
        clips = clips.iloc[np.sort(speaker_rng(seed, key).choice(len(clips), n, replace=False))]
    return clips


def plan_zip_clips(meta: pd.DataFrame, speakers: pd.DataFrame, clips_per_speaker: int,
                   max_target_seconds: float, seed: int) -> pd.DataFrame:
    """One row per synthetic clip: every synthetic speaker draws its own clips of its source speaker."""
    eligible = meta[meta["duration"] <= max_target_seconds].groupby("client_id")
    columns = ["path", "duration", "sentence", "sentence_source"]
    parts = []
    for s in speakers.itertuples():
        clips = sample_clips(eligible.get_group(s.source_client_id), clips_per_speaker, s.client_id, seed)[columns]
        parts.append(clips.rename(columns={"path": "source_path"}).assign(client_id=s.client_id))
    plan = speakers.merge(pd.concat(parts, ignore_index=True), on="client_id")
    plan.insert(0, "path", [f"{cid}/{PurePosixPath(p).stem}.wav" for cid, p in zip(plan["client_id"], plan["source_path"])])
    return plan.sort_values(["zipcode", "client_id", "source_path"]).reset_index(drop=True)


def write_dataset(data_cfg: DictConfig, meta: pd.DataFrame, plan: pd.DataFrame, out_dir: Path,
                  clips_per_speaker: int, seed: int):
    """Write the dataset folder: tsvs of the original (subsampled) and synthetic clips, and audio_16k.h5.

        {train_all,valid,test}.tsv            by the release file of the (source) speaker
        splits/<split_dir name>/{train,valid,test}.tsv   by its split of data.split_dir
        speakers.csv  zip_municipality.csv  audio_16k.h5
    """
    generated = synthetic_meta(plan, read_results(out_dir / "results.jsonl"), out_dir / "clips")
    if generated.empty:
        raise ValueError(f"no generated clip in {out_dir / 'clips'}: run the generate stage first")
    original = pd.concat([sample_clips(clips, clips_per_speaker, f"original/{cid}", seed)
                          for cid, clips in meta.groupby("client_id", sort=True)]).assign(synthetic=False)
    columns = [c for c in meta.columns if c not in ("release", "split")]
    clips = pd.concat([original, generated], ignore_index=True)
    for name in RELEASE_FILES:
        write_tsv(clips.loc[clips["release"] == name, [*columns, "synthetic", *PROVENANCE]], out_dir / f"{name}.tsv")
    if data_cfg.split_dir:
        for split in SPLITS:
            write_tsv(clips.loc[clips["split"] == split, [*columns, "synthetic", *PROVENANCE]],
                      out_dir / "splits" / Path(data_cfg.split_dir).name / f"{split}.tsv")
    if data_cfg.geo_table and Path(data_cfg.geo_table).is_file():
        (out_dir / "zip_municipality.csv").write_bytes(Path(data_cfg.geo_table).read_bytes())

    per_speaker = clips.groupby("client_id").agg(
        synthetic=("synthetic", "first"), split=("split", "first"), release=("release", "first"),
        zipcode=("zipcode", "first"), gender=("gender", "first"), dialect_region=("dialect_region", "first"),
        n_clips=("path", "size"), hours=("duration", lambda d: d.sum() / 3600)).reset_index()
    provenance = plan.drop_duplicates("client_id")[["client_id", "source_client_id", "donor_client_id",
                                                    "donor_dialect_region", "donor_zipcode", "language"]]
    per_speaker.merge(provenance, on="client_id", how="left").round({"hours": 3}).to_csv(
        out_dir / "speakers.csv", index=False)
    per_zip = per_speaker.groupby(["zipcode", "gender"]).size().unstack(fill_value=0)
    log.info("%d original clips of %d speakers + %d synthetic clips of %d speakers; speakers per zip code: "
             "min %d, female - male in [%d, %d]", len(original), original["client_id"].nunique(), len(generated),
             generated["client_id"].nunique(), per_zip.sum(axis=1).min(), (per_zip["female"] - per_zip["male"]).min(),
             (per_zip["female"] - per_zip["male"]).max())
    pack_h5(Path(data_cfg.audio_h5), [(p, out_dir / "clips" / p) for p in generated["path"]],
            out_dir / Path(data_cfg.audio_h5).name)


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="anonymize_zip.yaml")
def main(cfg: DictConfig):
    if unknown := set(cfg.stages) - set(STAGES):
        raise ValueError(f"unknown stages {sorted(unknown)}, pick from {STAGES}")
    sc = cfg.synth
    out_dir = Path(sc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_sources(cfg.data, set(read_h5_index(cfg.data.audio_h5)[0].index))

    plan_file = out_dir / "plan.tsv"
    if "plan" in cfg.stages:
        speakers = speaker_table(meta).merge(
            meta.drop_duplicates("client_id")[["client_id", "split", "release"]], on="client_id")
        eligible = meta[meta["duration"] <= sc.max_target_seconds].groupby("client_id").size()
        new = plan_zip_speakers(speakers, eligible, sc.min_per_gender, sc.clips_per_speaker, cfg.seed)
        plan = plan_zip_clips(meta, new, sc.clips_per_speaker, sc.max_target_seconds, cfg.seed)
        log.info("plan: %d synthetic speakers, %d clips (%.1f h)", plan["client_id"].nunique(), len(plan),
                 plan["duration"].sum() / 3600)
        plan.to_csv(plan_file, sep="\t", index=False, quoting=3)
    elif not plan_file.is_file():
        raise FileNotFoundError(f"{plan_file} does not exist: run the plan stage first")
    plan = pd.read_csv(plan_file, sep="\t", quoting=3, dtype=str).astype({"duration": float})

    if "generate" in cfg.stages:
        gc = cfg.globi
        client = GlobiClient(gc.url, timeout=gc.timeout, retries=gc.retries, poll_interval=gc.poll_interval,
                             job_timeout=gc.job_timeout)
        generate(plan, meta, ClipReader(cfg.data.audio_h5), client, out_dir, sc.reference_seconds,
                 gc.concurrency, cfg.seed, gc.max_consecutive_failures, gc.status_interval)
    if "pack" in cfg.stages:
        write_dataset(cfg.data, meta, plan, out_dir, sc.clips_per_speaker, cfg.seed)


if __name__ == "__main__":
    main()
