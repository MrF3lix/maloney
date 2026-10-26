"""Turn clip metadata and precomputed clip embeddings into evaluation units.

    clip      one unit per clip
    segment   clips of one speaker grouped into ~`data.segment_duration` seconds (SpeakerSegmentSampler;
              random grouping for train, fixed for valid/test), embeddings pooled over their frames
    speaker   all clips of a speaker, pooled

A unit keeps the metadata columns of its first clip (speaker, municipality, location, region; the
same for all clips of a unit) plus `n_clips`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maloney.data.stt4sg import STT4SGDataModule
from maloney.embeddings import EmbeddingStore

UNITS = ("clip", "segment", "speaker")


def unit_table(datamodule: STT4SGDataModule, split: str, unit: str,
               shuffle: bool | None = None) -> tuple[pd.DataFrame, list[tuple[int, ...]] | None]:
    """Metadata of the clip or segment units of a split (one row each) and the segments.

    The order is that of `datamodule.dataloader(split, shuffle)`; for train without `shuffle=False`
    the segment grouping is the random one of the first epoch.
    """
    meta = datamodule.datasets[split].meta
    if unit == "clip":
        return meta.assign(n_clips=1).reset_index(drop=True), None
    if unit == "segment":
        sampler = datamodule.sampler(split, shuffle)
        if sampler is None:
            raise ValueError("unit=segment needs data.segment_duration (e.g. 30)")
        segments = sampler.segments(0)
        units = meta.iloc[[segment[0] for segment in segments]].assign(n_clips=[len(s) for s in segments])
        return units.reset_index(drop=True), segments
    raise ValueError(f"unit must be clip or segment here, got {unit!r}")


def build_units(datamodule: STT4SGDataModule, split: str, store: EmbeddingStore, layer: int,
                unit: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Embeddings [n_units, D] and metadata (one row per unit) of one split."""
    meta = datamodule.datasets[split].meta
    if unit == "clip":
        return store.for_meta(meta, layer), unit_table(datamodule, split, unit)[0]
    if unit == "segment":
        units, segments = unit_table(datamodule, split, unit)
        return store.for_segments(meta, segments, layer), units
    if unit == "speaker":
        speakers, x = store.by_speaker(meta, layer)
        units = meta.drop_duplicates("client_id").set_index("client_id").loc[speakers].reset_index()
        return x, units.assign(n_clips=meta.groupby("client_id").size().loc[speakers].to_numpy())
    raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")


def speaker_scores(scores: np.ndarray, units: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Average the class scores of all units of a speaker: scores [n_speakers, K] and one row each."""
    codes, speakers = pd.factorize(units["client_id"])
    sums = np.zeros((len(speakers), scores.shape[1]))
    np.add.at(sums, codes, scores)
    first = units.drop_duplicates("client_id").set_index("client_id").loc[speakers].reset_index()
    counts = np.bincount(codes)
    return sums / counts[:, None], first.assign(n_clips=units.groupby("client_id")["n_clips"].sum().loc[speakers].to_numpy())
