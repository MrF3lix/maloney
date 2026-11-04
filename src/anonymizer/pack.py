"""Turn the generated clips into a dataset folder that replaces the source data folder as is.

    out_dir/
        clips/<client_id>/<name>.wav          globi's output (24 kHz)
        plan.tsv  results.jsonl               what was asked for, what came back
        speakers.csv                          synthetic speaker -> source speaker and donor
        test.tsv                              (source without split_dir) the synthetic test split
        splits/<name>/{train,valid,test}.tsv  (source with split_dir) train and valid as in the source
        zip_municipality.csv                  copy of the source's geo table
        audio_16k.h5                          the source's audio file plus the synthetic clips

`audio_16k.h5` has the layout of scripts/pack_stt4sg_h5.py, so the data module, the embedding
extraction and everything else read it unchanged. Its `audio` is an HDF5 virtual dataset: the first
part links to the `audio` of the source file (by a path relative to this file, nothing is copied),
the rest are the synthetic clips, resampled to the source's rate and stored in `synth_audio`. Moving
the two data folders together keeps the link; the data module checks that it resolves.

The test split holds the synthetic clips (`synthetic` = True) and, with `include_original`, also the
source's test clips, so the original test results are the subset `synthetic == False`.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import soxr

from maloney.data.stt4sg import SPLITS, read_h5_index, read_split

log = logging.getLogger(__name__)

PROVENANCE = ["source_path", "source_client_id", "donor_client_id", "donor_dialect_region", "donor_canton",
              "donor_zipcode", "language", "transcript", "job_id"]
CHUNK = 1 << 17  # as in scripts/pack_stt4sg_h5.py


def write_tsv(meta: pd.DataFrame, file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(file, sep="\t", index=False, quoting=3)  # as read by read_split


def synthetic_meta(plan: pd.DataFrame, results: dict[str, dict], clips_dir: Path) -> pd.DataFrame:
    """The plan's clips that were generated, with their true duration, as test.tsv rows."""
    ok = plan["path"].map(lambda p: results.get(p, {}).get("status") == "completed"
                          and (clips_dir / p).is_file())
    if missing := int((~ok).sum()):
        log.warning("%d/%d planned clips were not generated and are left out", missing, len(plan))
    meta = plan[ok].copy()
    meta["duration"] = [sf.info(clips_dir / p).duration for p in meta["path"]]
    meta["transcript"] = [results[p].get("transcript", "") for p in meta["path"]]
    meta["job_id"] = [results[p].get("job_id", "") for p in meta["path"]]
    meta["synthetic"] = True
    return meta


def pack_h5(base_h5: Path, clips: list[tuple[str, Path]], out: Path):
    """Write `out`: the clips of `base_h5` (linked) plus `clips` ((tsv path, wav file)), see the module doc."""
    base_index, sample_rate = read_h5_index(base_h5)
    if shared := set(base_index.index) & {path for path, _ in clips}:
        raise ValueError(f"{len(shared)} synthetic paths are already in {base_h5}, e.g. {sorted(shared)[:3]}")
    with h5py.File(base_h5, "r") as f:
        n_base = f["audio"].shape[0]
    tmp = out.with_name(out.name + ".tmp")
    offsets, lengths, total = [], [], 0
    with h5py.File(tmp, "w") as f:
        synth = f.create_dataset("synth_audio", shape=(0,), maxshape=(None,), dtype="int16", chunks=(CHUNK,))
        for i, (_, wav) in enumerate(clips):
            audio, sr = sf.read(wav, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sr != sample_rate:
                audio = soxr.resample(audio, sr, sample_rate)
            audio = np.round(np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            synth.resize((total + len(audio),))
            synth[total:] = audio
            offsets.append(n_base + total)
            lengths.append(len(audio))
            total += len(audio)
            if (i + 1) % 5000 == 0:
                log.info("packed %d/%d synthetic clips", i + 1, len(clips))

        layout = h5py.VirtualLayout(shape=(n_base + total,), dtype="int16")
        base_link = os.path.relpath(base_h5.resolve(), out.parent.resolve())
        layout[:n_base] = h5py.VirtualSource(base_link, "audio", shape=(n_base,))
        if total:
            layout[n_base:] = h5py.VirtualSource(".", "synth_audio", shape=(total,))
        f.create_virtual_dataset("audio", layout)
        f.create_dataset("path", data=[*base_index.index, *(path for path, _ in clips)], dtype=h5py.string_dtype())
        f.create_dataset("offset", data=np.concatenate([base_index["offset"].to_numpy(), offsets]).astype(np.int64))
        f.create_dataset("length", data=np.concatenate([base_index["length"].to_numpy(), lengths]).astype(np.int64))
        f.attrs["sample_rate"] = sample_rate
        f.attrs["base_audio_h5"] = base_link
        f.attrs["n_base_clips"] = len(base_index)
    os.replace(tmp, out)
    log.info("%s: %d source clips (linked to %s) + %d synthetic clips, %.1f h", out, len(base_index), base_link,
             len(clips), total / sample_rate / 3600)


def pack(data_cfg, plan: pd.DataFrame, results: dict[str, dict], out_dir: Path, split: str,
         include_original: bool):
    """Write the dataset folder (module doc) for the synthetic clips of `split`."""
    synth = synthetic_meta(plan, results, out_dir / "clips")
    if synth.empty:
        raise ValueError(f"no generated clip in {out_dir / 'clips'}: run the generate stage first")
    split_dir = Path(data_cfg.split_dir) if data_cfg.split_dir else None
    metas = {s: read_split(data_cfg.data_dir, s, data_cfg.train_split, split_dir) for s in SPLITS}
    columns = list(metas[split].columns)
    if include_original:
        original = metas[split].assign(synthetic=False)
        synth = pd.concat([original, synth], ignore_index=True)
    metas[split] = synth[[*columns, "synthetic", *PROVENANCE]]

    if split_dir is not None:
        target_dir = out_dir / "splits" / split_dir.name
        for name, meta in metas.items():
            write_tsv(meta, target_dir / f"{name}.tsv")
        log.info("wrote %s (%s from %s)", target_dir, ", ".join(s for s in SPLITS if s != split), split_dir)
    else:
        # the release layout: train comes from train_all/train_balanced.tsv, which are copied
        for name in ("train_all.tsv", "train_balanced.tsv", *(f"{s}.tsv" for s in SPLITS if s not in ("train", split))):
            if (Path(data_cfg.data_dir) / name).is_file():
                shutil.copyfile(Path(data_cfg.data_dir) / name, out_dir / name)
        write_tsv(metas[split], out_dir / f"{split}.tsv")
    if data_cfg.geo_table and Path(data_cfg.geo_table).is_file():
        shutil.copyfile(data_cfg.geo_table, out_dir / "zip_municipality.csv")

    speakers = (plan.drop_duplicates("client_id")
                .drop(columns=["path", "source_path", "duration", "sentence", "sentence_source"])
                .merge(synth[synth["synthetic"]].groupby("client_id").agg(
                    n_clips=("path", "size"), hours=("duration", lambda d: d.sum() / 3600)).reset_index(),
                    on="client_id", how="left"))
    speakers.round({"hours": 3}).to_csv(out_dir / "speakers.csv", index=False)

    new = synth[synth["synthetic"]]
    pack_h5(Path(data_cfg.audio_h5), [(p, out_dir / "clips" / p) for p in new["path"]],
            out_dir / Path(data_cfg.audio_h5).name)
    log.info("%s split: %d synthetic clips (%.1f h) of %d speakers%s", split, len(new),
             new["duration"].sum() / 3600, new["client_id"].nunique(),
             f" + {int((~synth['synthetic']).sum())} original clips" if include_original else "")
