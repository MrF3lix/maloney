"""Precompute Whisper encoder embeddings of every clip into one HDF5 file (see maloney.embeddings).

Reads the clips from the packed audio (`audio_16k.h5`) and stores, per clip, the mean-pooled hidden
states of several encoder layers, so experiments on frozen embeddings never touch the audio:

    path        str     [n_clips]             tsv path of the clip (same order as the audio file)
    frames      int32   [n_clips]             encoder frames averaged (50 per second of audio)
    done        bool    [n_clips]             written yet; a killed job resumes from here
    layer_KK    float16 [n_clips, dim]        mean-pooled output of encoder layer KK
    attrs:      model, layers, dim, pooling, audio_h5

The job resumes when rerun with the same output (it checks model, layers and clips); --force
starts over. By default all clips of the audio file are embedded; --tsv restricts them to the
clips of some tsv files (e.g. a split).

    uv run python scripts/extract_whisper_embeddings.py data/stt4sg-350_v2-1      # whisper-large-v3
    uv run python scripts/extract_whisper_embeddings.py data/stt4sg-350_v2-1 --model openai/whisper-tiny \\
        --tsv data/stt4sg-350_v2-1/splits/smoke/*.tsv --layers 2 4
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soxr
import torch
from torch.utils.data import DataLoader, Dataset

from maloney.data.stt4sg import read_h5_index
from maloney.embeddings.whisper import SAMPLE_RATE, WhisperEmbedder, featurize

log = logging.getLogger("extract_whisper_embeddings")


class ClipAudio(Dataset):
    """Waveforms of some rows of the packed audio file, at 16 kHz."""

    def __init__(self, audio_h5: Path, offsets: np.ndarray, lengths: np.ndarray, sample_rate: int):
        self.audio_h5, self.offsets, self.lengths, self.sample_rate = audio_h5, offsets, lengths, sample_rate
        self._audio = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i: int) -> np.ndarray:
        if self._audio is None:
            self._audio = h5py.File(self.audio_h5, "r")["audio"]
        audio = self._audio[self.offsets[i]: self.offsets[i] + self.lengths[i]].astype(np.float32) / 32767
        return audio if self.sample_rate == SAMPLE_RATE else soxr.resample(audio, self.sample_rate, SAMPLE_RATE)


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def open_output(out: Path, paths: np.ndarray, embedder: WhisperEmbedder, audio_h5: Path, force: bool) -> h5py.File:
    """The output file, created or (if it matches this run) reopened to resume."""
    if out.exists() and not force:
        f = h5py.File(out, "a")
        same = (f.attrs["model"] == embedder.model_name and list(f.attrs["layers"]) == embedder.layers
                and len(f["path"]) == len(paths) and (f["path"].asstr()[:] == paths).all())
        if not same:
            f.close()
            raise SystemExit(f"{out} was made with another model, layers or set of clips; pass --force to "
                             "overwrite it or --out to write elsewhere")
        return f
    out.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(out, "w")
    f.create_dataset("path", data=paths, dtype=h5py.string_dtype())
    f.create_dataset("frames", shape=(len(paths),), dtype="int32")
    f.create_dataset("done", shape=(len(paths),), dtype="bool")
    for layer in embedder.layers:
        f.create_dataset(f"layer_{layer:02d}", shape=(len(paths), embedder.dim), dtype="float16",
                         chunks=(min(1024, len(paths)), embedder.dim))
    f.attrs.update(model=embedder.model_name, layers=embedder.layers, dim=embedder.dim,
                   pooling="mean over encoder frames with audio", audio_h5=str(audio_h5))
    return f


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="release folder with audio_16k.h5")
    parser.add_argument("--model", default="openai/whisper-large-v3", help="any Whisper checkpoint on the HF Hub")
    parser.add_argument("--layers", type=int, nargs="+", help="encoder layers to keep (default: 8 evenly spaced)")
    parser.add_argument("--audio-h5", type=Path, help="default: <data_dir>/audio_16k.h5")
    parser.add_argument("--tsv", type=Path, nargs="+", help="only embed the clips listed in these tsv files")
    parser.add_argument("--out", type=Path, help="default: <data_dir>/embeddings/<model>.h5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--device", help="default: cuda if available, else cpu")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output instead of resuming")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    audio_h5 = args.audio_h5 or args.data_dir / "audio_16k.h5"
    index, sample_rate = read_h5_index(audio_h5)
    if args.tsv:
        wanted = pd.concat(pd.read_csv(t, sep="\t", quoting=3, usecols=["path"]) for t in args.tsv)["path"]
        index = index[index.index.isin(set(wanted))]
        log.info("%d of the clips in %s are listed in %s", len(index), audio_h5.name, ", ".join(t.name for t in args.tsv))
    if index.empty:
        raise SystemExit("no clips to embed")

    embedder = WhisperEmbedder(args.model, args.layers, args.device)
    out = args.out or args.data_dir / "embeddings" / f"{model_slug(args.model)}.h5"
    f = open_output(out, index.index.to_numpy(dtype=object), embedder, audio_h5, args.force)
    todo = np.flatnonzero(~f["done"][:])
    log.info("%s on %s (%s), layers %s, dim %d: %d of %d clips to embed -> %s", args.model, embedder.device,
             str(embedder.dtype).removeprefix("torch."), embedder.layers, embedder.dim, len(todo), len(index), out)

    dataset = ClipAudio(audio_h5, index["offset"].to_numpy()[todo], index["length"].to_numpy()[todo], sample_rate)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=functools.partial(featurize, embedder.features),
                        pin_memory=embedder.device.type == "cuda", prefetch_factor=4 if args.workers else None)
    start, done, last_log = time.time(), 0, time.time()
    for batch in loader:
        rows = todo[done: done + int(batch["n_clips"])]
        pooled, frames = embedder.embed_features(batch)
        for i, layer in enumerate(embedder.layers):
            f[f"layer_{layer:02d}"][rows] = pooled[:, i].astype(np.float16)
        f["frames"][rows] = frames
        f["done"][rows] = True
        done += len(rows)
        if time.time() - last_log > 60 or done == len(todo):
            f.flush()
            rate = done / (time.time() - start)
            log.info("%d/%d clips, %.1f clips/s, eta %.0f min", done, len(todo), rate, (len(todo) - done) / rate / 60)
            last_log = time.time()
    f.close()
    log.info("done: %s", out)


if __name__ == "__main__":
    main()
