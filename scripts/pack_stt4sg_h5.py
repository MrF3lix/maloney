"""Decode every STT4SG-350 clip straight from the release's tars into a single HDF5 file.

The clip tars (`clips__*.tar` in `data_dir`) are streamed once, without extracting them: every clip
referenced by the tsv files is decoded in memory, mixed to mono, resampled to `--sample-rate` and
stored as int16 back to back in one flat dataset, so training reads one contiguous slice per clip
instead of opening and decoding a small file. Layout of the file:

    audio   int16 [total_samples]   all clips concatenated
    path    str   [n_clips]         the tsv `path` column (<client_id>/<hash>.<ext>)
    offset  int64 [n_clips]         start of each clip in `audio`
    length  int64 [n_clips]         number of samples of each clip
    attrs:  sample_rate

Tar entries are matched to tsv rows by <client_id>/<file name> without the extension, so leading
folders inside the tars and an extension that differs from the tsv (.flac listed, .mp3 shipped)
do not matter. Clips that are missing from the tars or fail to decode are left out and listed at
the end (the data module skips them); if more than --max-failed of them fail, something is wrong
and no file is written. The file is written to <out>.tmp and renamed when complete.

    python scripts/pack_stt4sg_h5.py data/stt4sg-350_v2.1                  # -> <data_dir>/audio_16k.h5
    python scripts/pack_stt4sg_h5.py data/stt4sg-350_v2.1 --workers 16 --out /scratch/audio_16k.h5
"""

from __future__ import annotations

import argparse
import functools
import io
import logging
import os
import tarfile
import time
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import soxr

log = logging.getLogger("pack_stt4sg_h5")

TSVS = ("train_all.tsv", "train_balanced.tsv", "valid.tsv", "test.tsv")
AUDIO_EXTS = {".mp3", ".flac", ".wav", ".ogg"}
CHUNK = 1 << 17           # samples per HDF5 chunk (256 KB); a typical clip spans 1-3 chunks
FLUSH = 1 << 26           # samples buffered in memory before they are appended to the file
BATCH = 2000              # clips read from the tar and decoded per round (bounds memory)


def clip_key(name: str) -> str:
    """<client_id>/<stem> of a tsv path or tar entry, ignoring leading folders and the extension."""
    p = PurePosixPath(name)
    return f"{p.parent.name}/{p.stem}"


def decode(item: tuple[str, bytes], sample_rate: int) -> tuple[str, np.ndarray | None]:
    path, data = item
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - one broken clip should not end an hour-long job
        log.warning("cannot decode %s: %s", path, e)
        return path, None
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = soxr.resample(audio, sr, sample_rate)
    return path, np.round(np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def read_clips(tars: list[Path], wanted: dict[str, str], seen: set[str]) -> Iterator[tuple[str, bytes]]:
    """(tsv path, file bytes) for every audio entry of the tars that a tsv lists, each clip once."""
    for tar in tars:
        log.info("reading %s", tar)
        with tarfile.open(tar, "r|*") as tf:  # streaming: one sequential pass over the archive
            unmatched = 0
            for member in tf:
                name = PurePosixPath(member.name)
                # ._* are macOS resource forks that tar on a Mac adds next to every file.
                if not member.isfile() or name.suffix.lower() not in AUDIO_EXTS or name.name.startswith("._"):
                    continue
                path = wanted.get(clip_key(member.name))
                if path is None:
                    if not unmatched:
                        log.warning("%s: entry %s is in no tsv (tsv paths look like %s)",
                                    tar.name, member.name, next(iter(wanted.values())))
                    unmatched += 1
                    continue
                if path in seen:
                    continue
                seen.add(path)
                yield path, tf.extractfile(member).read()
            if unmatched:
                log.warning("%s: %d audio entries are in no tsv and were skipped", tar.name, unmatched)


def batches(it: Iterator, n: int) -> Iterator[list]:
    while batch := [x for _, x in zip(range(n), it)]:
        yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="release folder with the tsv files and clips__*.tar")
    parser.add_argument("--tars", type=Path, nargs="+", help="default: <data_dir>/clips__*.tar")
    parser.add_argument("--out", type=Path, help="default: <data_dir>/audio_<sr/1000>k.h5")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-failed", type=float, default=0.01, help="fraction of clips allowed to fail")
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity") else os.cpu_count())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    tars = args.tars or sorted(args.data_dir.glob("clips__*.tar"))
    if not tars:
        raise SystemExit(f"no clips__*.tar in {args.data_dir}; pass them with --tars")
    out = args.out or args.data_dir / f"audio_{args.sample_rate // 1000}k.h5"
    tsvs = [args.data_dir / t for t in TSVS if (args.data_dir / t).is_file()]
    paths = set().union(*(pd.read_csv(t, sep="\t", quoting=3, usecols=["path"])["path"] for t in tsvs))
    if not paths:
        raise SystemExit(f"no tsv with a path column found in {args.data_dir}")
    wanted = {clip_key(p): p for p in paths}
    log.info("%d clips from %s, reading %s -> %s (%d Hz, %d workers)", len(paths),
             ", ".join(t.name for t in tsvs), ", ".join(t.name for t in tars), out, args.sample_rate, args.workers)

    kept, offsets, lengths, failed, seen = [], [], [], [], set()
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

        def write(results):
            nonlocal total, buffered
            for path, wav in results:
                if wav is None:
                    failed.append(path)
                    continue
                kept.append(path)
                offsets.append(total)
                lengths.append(len(wav))
                total += len(wav)
                buffered += len(wav)
                buf.append(wav)
                if buffered >= FLUSH:
                    flush()

        # Decode one batch in the pool while the next one is read from the tar.
        job = functools.partial(decode, sample_rate=args.sample_rate)
        pending = None
        for i, batch in enumerate(batches(read_clips(tars, wanted, seen), BATCH), 1):
            running = pool.map_async(job, batch, chunksize=16)
            if pending is not None:
                write(pending.get())
            pending = running
            done = len(kept) + len(failed)
            if i % 10 == 0:
                rate = done / (time.time() - start)
                log.info("%d/%d clips, %.1f h of audio, %.0f clips/s, eta %.0f min", done, len(paths),
                         total / args.sample_rate / 3600, rate, (len(paths) - done) / rate / 60)
            if len(failed) == done >= BATCH:
                raise SystemExit(f"the first {done} clips all failed to decode (e.g. {failed[:3]}); see the "
                                 f"warnings above (libsndfile {sf.__libsndfile_version__})")
        if pending is not None:
            write(pending.get())
        flush()

        missing = sorted(paths - seen)
        if missing:
            log.warning("%d tsv clips are in none of the tars, e.g. %s", len(missing), missing[:5])
        bad = len(failed) + len(missing)
        if bad > args.max_failed * len(paths):
            raise SystemExit(f"{bad}/{len(paths)} clips are missing or could not be decoded, not writing {out}; "
                             "see the warnings above")

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
