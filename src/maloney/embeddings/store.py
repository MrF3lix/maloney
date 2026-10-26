"""Read precomputed clip embeddings (scripts/extract_whisper_embeddings.py) for experiments.

A layer is loaded into memory once (float32, ~1.4 GB for 270k clips at dim 1280) and clips are
looked up by their tsv path. Pooling several clips (a segment, a speaker) weights every clip by its
number of encoder frames, which equals averaging over all their frames, since every clip embedding
is itself the mean over its frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


class EmbeddingStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.model = str(f.attrs["model"])
            self.layers = [int(layer) for layer in f.attrs["layers"]]
            self.dim = int(f.attrs["dim"])
            self.paths = f["path"].asstr()[:]
            self.frames = f["frames"][:].astype(np.float64)
            done = f["done"][:]
        if not done.all():
            raise ValueError(f"{self.path} is incomplete ({(~done).sum()} of {len(done)} clips missing); "
                             "rerun the extraction to resume it")
        self.row = pd.Series(np.arange(len(self.paths)), index=self.paths)
        self._layers: dict[int, np.ndarray] = {}

    def __repr__(self) -> str:
        return f"EmbeddingStore({self.model}, {len(self.paths)} clips, layers {self.layers}, dim {self.dim})"

    def layer(self, layer: int) -> np.ndarray:
        """All clip embeddings of one layer, [n_clips, dim] float32 (cached)."""
        if layer not in self.layers:
            raise ValueError(f"layer {layer} was not extracted; {self.path.name} has {self.layers}")
        if layer not in self._layers:
            with h5py.File(self.path, "r") as f:
                self._layers[layer] = f[f"layer_{layer:02d}"][:].astype(np.float32)
        return self._layers[layer]

    def rows(self, paths) -> np.ndarray:
        rows = self.row.reindex(np.asarray(paths, dtype=object))
        if rows.isna().any():
            missing = rows.index[rows.isna()]
            raise KeyError(f"{len(missing)} clips are not in {self.path.name}, e.g. {list(missing[:3])}")
        return rows.to_numpy(np.int64)

    def get(self, paths, layer: int) -> np.ndarray:
        """Embeddings [len(paths), dim] of clips given by tsv path."""
        return self.layer(layer)[self.rows(paths)]

    def for_meta(self, meta: pd.DataFrame, layer: int) -> np.ndarray:
        """One embedding per row of clip metadata (e.g. STT4SGDataset.meta), in its order."""
        return self.get(meta["path"], layer)

    def pooled(self, groups: Sequence[Sequence[str]], layer: int) -> np.ndarray:
        """Frame-weighted mean embedding [len(groups), dim] of each group of clip paths."""
        rows = [self.rows(group) for group in groups]
        owner = np.repeat(np.arange(len(groups)), [len(r) for r in rows])
        flat = np.concatenate(rows)
        weights = self.frames[flat]
        sums = np.zeros((len(groups), self.dim))
        np.add.at(sums, owner, self.layer(layer)[flat] * weights[:, None])
        return (sums / np.bincount(owner, weights, minlength=len(groups))[:, None]).astype(np.float32)

    def for_segments(self, meta: pd.DataFrame, segments: Sequence[tuple[int, ...]], layer: int) -> np.ndarray:
        """Pooled embedding of each segment of clip indices into `meta` (from SpeakerSegmentSampler)."""
        paths = meta["path"].to_numpy()
        return self.pooled([paths[list(segment)] for segment in segments], layer)

    def by_speaker(self, meta: pd.DataFrame, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Speaker ids and their pooled embedding over all their clips in `meta`."""
        groups = meta.groupby("client_id", sort=True)["path"]
        return np.array(list(groups.groups)), self.pooled([g.to_numpy() for _, g in groups], layer)
