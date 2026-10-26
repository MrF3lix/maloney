"""Decode every STT4SG-350 clip once and pack them into a single HDF5 file.

All clips referenced by the release's tsv files are decoded, mixed to mono, resampled to
`--sample-rate` and stored as int16 back to back in one flat dataset, so training reads one
contiguous slice per clip instead of opening and decoding a small file. Layout of the file:

    audio   int16 [total_samples]   all clips concatenated
    path    str   [n_clips]         the tsv `path` column (<client_id>/<hash>.<ext>), sorted
    offset  int64 [n_clips]         start of each clip in `audio`
    length  int64 [n_clips]         number of samples of each clip
    attrs:  sample_rate

Clips that fail to decode are left out and listed at the end (the data module skips them); if
more than --max-failed of them fail, something is wrong with the setup and no file is written.
The file is written to <out>.tmp and renamed when complete.

    python scripts/pack_stt4sg_h5.py data/stt4sg-350_v2.1                  # -> <data_dir>/audio_16k.h5
    python scripts/pack_stt4sg_h5.py data/stt4sg-350_v2.1 --workers 16 --out /scratch/audio_16k.h5
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import soxr

log = logging.getLogger("pack_stt4sg_h5")

TSVS = ("train_all.tsv", "train_balanced.tsv", "valid.tsv", "test.tsv")
CHUNK = 1 << 17           # samples per HDF5 chunk (256 KB); a typical clip spans 1-3 chunks
FLUSH = 1 << 26           # samples buffered in memory before they are appended to the file
# The tsv's extension does not always match the files in the tars (e.g. .flac listed, .mp3 shipped).
AUDIO_EXTS = (".mp3", ".flac", ".wav", ".ogg")


def find_clip(path: str, clips_dir: Path) -> Path | None:
    """The file for a tsv `path`, also under another audio extension; None if there is none."""
    file = clips_dir / path
    for candidate in (file, *(file.with_suffix(ext) for ext in AUDIO_EXTS if ext != file.suffix)):
        if candidate.is_file():
            return candidate
    return None


def decode(path: str, clips_dir: Path, sample_rate: int) -> np.ndarray | None:
    file = find_clip(path, clips_dir)
    if file is None:
        log.warning("not found: %s (also tried %s)", clips_dir / path, "/".join(AUDIO_EXTS))
        return None
    try:
        audio, sr = sf.read(file, dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - one broken clip should not end an hour-long job
        log.warning("cannot decode %s: %s", path, e)
        return None
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = soxr.resample(audio, sr, sample_rate)
    return np.round(np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="release folder with the tsv files and clips/")
    parser.add_argument("--clips-dir", type=Path, help="default: <data_dir>/clips")
    parser.add_argument("--out", type=Path, help="default: <data_dir>/audio_<sr/1000>k.h5")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-failed", type=float, default=0.01, help="fraction of clips allowed to fail")
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity") else os.cpu_count())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    clips_dir = args.clips_dir or args.data_dir / "clips"
    out = args.out or args.data_dir / f"audio_{args.sample_rate // 1000}k.h5"
    tsvs = [args.data_dir / t for t in TSVS if (args.data_dir / t).is_file()]
    # Sorted, the clips of one speaker (<client_id>/...) end up next to each other in the file.
    paths = sorted(set().union(*(pd.read_csv(t, sep="\t", quoting=3, usecols=["path"])["path"] for t in tsvs)))
    if not paths:
        raise SystemExit(f"no tsv with a path column found in {args.data_dir}")
    # Fail in seconds rather than after an hour if the clips are missing or unreadable.
    if all(decode(p, clips_dir, args.sample_rate) is None for p in paths[:10]):
        found = next((str(f.relative_to(clips_dir)) for f in clips_dir.rglob("*") if f.is_file()), None)
        raise SystemExit(f"cannot decode any of the first clips in {clips_dir} (libsndfile "
                         f"{sf.__libsndfile_version__}): the tsv lists e.g. {paths[0]}, the folder holds e.g. "
                         f"{found}. Are the clips extracted there, with <client_id>/<file> directly below it?")
    log.info("%d clips from %s -> %s (%d Hz, %d workers)", len(paths), ", ".join(t.name for t in tsvs),
             out, args.sample_rate, args.workers)

    kept, offsets, lengths, failed = [], [], [], []
    total, buffered, buf = 0, 0, []
    start = time.time()
    tmp = out.with_name(out.name + ".tmp")
    with h5py.File(tmp, "w") as f, Pool(args.workers) as pool:
        audio = f.create_dataset("audio", shape=(0,), maxshape=(None,), dtype="int16", chunks=(CHUNK,))

        def flush():
            nonlocal buffered
            if buf:
                audio.resize((total,))
                audio[total - buffered:] = np.concatenate(buf)
                buf.clear()
                buffered = 0

        job = functools.partial(decode, clips_dir=clips_dir, sample_rate=args.sample_rate)
        for i, (path, wav) in enumerate(zip(paths, pool.imap(job, paths, chunksize=64)), 1):
            if wav is None:
                failed.append(path)
            else:
                kept.append(path)
                offsets.append(total)
                lengths.append(len(wav))
                total += len(wav)
                buffered += len(wav)
                buf.append(wav)
                if buffered >= FLUSH:
                    flush()
            if i % 20000 == 0:
                rate = i / (time.time() - start)
                log.info("%d/%d clips, %.1f h of audio, %.0f clips/s, eta %.0f min", i, len(paths),
                         total / args.sample_rate / 3600, rate, (len(paths) - i) / rate / 60)
        flush()
        if len(failed) > args.max_failed * len(paths):
            raise SystemExit(f"{len(failed)}/{len(paths)} clips could not be decoded (e.g. {failed[:5]}), "
                             f"not writing {out}; see the warnings above")

        f.create_dataset("path", data=kept, dtype=h5py.string_dtype())
        f.create_dataset("offset", data=np.asarray(offsets, dtype=np.int64))
        f.create_dataset("length", data=np.asarray(lengths, dtype=np.int64))
        f.attrs["sample_rate"] = args.sample_rate
    os.replace(tmp, out)

    log.info("done: %d clips, %.1f h, %.1f GB in %.0f min", len(kept), total / args.sample_rate / 3600,
             out.stat().st_size / 1e9, (time.time() - start) / 60)
    if failed:
        log.warning("%d clips could not be decoded and were left out: %s", len(failed), failed[:20])


if __name__ == "__main__":
    main()
