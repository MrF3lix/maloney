"""STT4SG-350 (Swiss German speech), with its official or a custom train/valid/test split.

Expected layout of `data_dir` (the release as downloaded, plus the packed audio):

    data_dir/
        train_all.tsv  train_balanced.tsv  valid.tsv  test.tsv
        audio_16k.h5                          # all clips decoded to int16, see below

`scripts/extract_stt4sg.submit` extracts the mp3s to `clips/`, then `scripts/pack_stt4sg_h5.submit`
decodes them into `audio_16k.h5`: the flat int16 dataset `audio` plus, per clip, its tsv `path`,
`offset` and `length`. Each tsv row has the columns
path, duration, sentence, sentence_source, client_id, dialect_region, canton, zipcode, age, gender.

A custom split (`split_dir`, e.g. from scripts/make_speaker_split.py) is a folder with train.tsv,
valid.tsv and test.tsv in the same format. With `geo_table` (zip_municipality.csv from
scripts/build_zip_geo.py), every item also carries its speaker's municipality and its centroid.

With `segment_duration`, an item is not one clip but several clips of the same speaker joined into
one waveform of about that many seconds (see SpeakerSegmentSampler).
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soxr
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

log = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test")
# The release ships two training lists: everything, or one balanced across dialect regions.
TRAIN_FILES = {"train_all": "train_all.tsv", "train_balanced": "train_balanced.tsv"}


# Columns of zip_municipality.csv attached to every clip: the speaker's municipality, its canton
# (`canton` there, renamed so it does not clash with the speaker's canton in the tsv) and its centroid
# in LV95 (E/N, metres) and WGS84 (lat/lon).
GEO_COLUMNS = ["bfs_nr", "municipality", "muni_canton", "muni_E", "muni_N", "muni_lat", "muni_lon"]


def read_split(data_dir: str | Path, split: str, train_split: str = "train_all",
               split_dir: str | Path | None = None) -> pd.DataFrame:
    """Read one split's metadata; `split` is train/valid/test.

    From `split_dir/<split>.tsv` if given, else from the release in `data_dir`, where `train_split`
    picks the train list.
    """
    if split_dir is not None:
        file = Path(split_dir) / f"{split}.tsv"
    else:
        file = Path(data_dir) / (TRAIN_FILES[train_split] if split == "train" else f"{split}.tsv")
    # quoting=3 (QUOTE_NONE): sentences contain quotes, and zipcodes must stay strings.
    return pd.read_csv(file, sep="\t", quoting=3, dtype={"zipcode": str, "client_id": str})


def read_geo_table(geo_table: str | Path) -> pd.DataFrame:
    """zip code -> municipality and centroid coordinates (GEO_COLUMNS), indexed by zip code."""
    geo = pd.read_csv(geo_table, dtype={"zipcode": str}).rename(columns={"canton": "muni_canton"})
    geo = geo[["zipcode", *GEO_COLUMNS]]
    return geo.dropna(subset=["bfs_nr", "muni_E"]).astype({"bfs_nr": int}).set_index("zipcode")


class LabelVocab:
    """Bidirectional mapping between string labels and contiguous class ids."""

    def __init__(self, labels):
        self.itos = sorted(set(labels))
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, label: str) -> int:
        return self.stoi[label]


def read_h5_index(audio_h5: str | Path) -> tuple[pd.DataFrame, int]:
    """The packed file's clip index (path -> offset, length) and its sample rate."""
    with h5py.File(audio_h5, "r") as f:
        index = pd.DataFrame({"offset": f["offset"][:], "length": f["length"][:]},
                             index=pd.Index(f["path"].asstr()[:], name="path"))
        return index, int(f.attrs["sample_rate"])


def pack_segment(clips: np.ndarray, durations: np.ndarray, segment_duration: float) -> list[tuple[int, ...]]:
    """Group consecutive clips into segments of at most `segment_duration` seconds.

    A clip longer than that is a segment of its own; a leftover at the end shorter than half the
    target is added to the previous segment instead, so no audio is dropped.
    """
    segments, current, total = [], [], 0.0
    for clip, duration in zip(clips.tolist(), durations.tolist()):
        if current and total + duration > segment_duration:
            segments.append(current)
            current, total = [], 0.0
        current.append(clip)
        total += duration
    if current:
        if segments and total < segment_duration / 2:
            segments[-1].extend(current)
        else:
            segments.append(current)
    return [tuple(segment) for segment in segments]


class SpeakerSegmentSampler(Sampler):
    """Yields tuples of clip indices of one speaker that together last about `segment_duration` s.

    With `shuffle`, every epoch (every iteration over the sampler) shuffles the clips within each
    speaker before grouping them, so the segments are new combinations each time, and shuffles the
    segments. Without it the grouping is fixed (clips in tsv order), for evaluation. The sampler
    runs in the main process, so this also works with persistent dataloader workers.
    """

    def __init__(self, speakers, durations, segment_duration: float, shuffle: bool, seed: int = 0):
        self.speakers = pd.Series(speakers).groupby(speakers, sort=True).indices
        self.durations = np.asarray(durations, dtype=float)
        self.segment_duration = segment_duration
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._len = len(self.segments(0))  # varies by a few segments between shuffled epochs

    def segments(self, epoch: int) -> list[tuple[int, ...]]:
        rng = np.random.default_rng((self.seed, epoch))
        segments = []
        for clips in self.speakers.values():
            if self.shuffle:
                clips = rng.permutation(clips)
            segments += pack_segment(clips, self.durations[clips], self.segment_duration)
        if self.shuffle:
            segments = [segments[i] for i in rng.permutation(len(segments))]
        return segments

    def __iter__(self):
        segments = self.segments(self.epoch if self.shuffle else 0)
        self.epoch += 1
        return iter(segments)

    def __len__(self):
        return self._len


class STT4SGDataset(Dataset):
    """One split of STT4SG-350; items are mono waveforms at `sample_rate` plus speaker metadata.

    `meta` needs the columns `offset` and `length` locating each clip in the packed `audio_h5`.
    An index is one clip, or a tuple of clips of the same speaker (from SpeakerSegmentSampler) that
    are joined into one waveform with `segment_gap` seconds of silence between them; `path` is then
    the tuple of their paths.
    """

    def __init__(self, meta: pd.DataFrame, audio_h5: str | Path, vocabs: dict[str, LabelVocab],
                 sample_rate: int = 16000, h5_sample_rate: int = 16000, segment_gap: float = 0.0):
        self.meta = meta.reset_index(drop=True)
        self.audio_h5 = Path(audio_h5)
        self.vocabs = vocabs
        self.sample_rate = sample_rate
        self.h5_sample_rate = h5_sample_rate
        self.segment_gap = segment_gap
        self._audio = None  # opened lazily, so every dataloader worker gets its own handle

    def __len__(self):
        return len(self.meta)

    def __getstate__(self):
        return {**self.__dict__, "_audio": None}

    def load_audio(self, offset: int, length: int) -> np.ndarray:
        if self._audio is None:
            self._audio = h5py.File(self.audio_h5, "r")["audio"]
        audio = self._audio[offset: offset + length].astype(np.float32) / 32767
        if self.h5_sample_rate != self.sample_rate:
            audio = soxr.resample(audio, self.h5_sample_rate, self.sample_rate)
        return audio

    @property
    def durations(self) -> np.ndarray:
        """Exact duration of every clip in seconds (from the packed audio, not the rounded tsv)."""
        return self.meta["length"].to_numpy() / self.h5_sample_rate

    def __getitem__(self, idx: int | tuple[int, ...]) -> dict:
        clips = idx if isinstance(idx, tuple) else (idx,)
        rows = self.meta.iloc[list(clips)]
        if rows["client_id"].nunique() > 1:
            raise ValueError(f"a segment must come from one speaker, got clips {clips}")
        parts = [self.load_audio(offset, length) for offset, length in zip(rows["offset"], rows["length"])]
        if len(parts) > 1 and self.segment_gap > 0:
            gap = np.zeros(round(self.segment_gap * self.sample_rate), dtype=np.float32)
            parts = [p for part in parts for p in (part, gap)][:-1]
        row = rows.iloc[0]
        item = {
            "audio": torch.from_numpy(np.concatenate(parts)),
            "n_clips": len(clips),
            "dialect": self.vocabs["dialect_region"][row["dialect_region"]],
            "canton": self.vocabs["canton"][row["canton"]],
            "zipcode": row["zipcode"],
            "client_id": row["client_id"],
            "path": tuple(rows["path"]) if isinstance(idx, tuple) else row["path"],
        }
        if "bfs_nr" in row:
            item.update(bfs_nr=row["bfs_nr"], municipality=row["municipality"],
                        lv95=(row["muni_E"], row["muni_N"]), latlon=(row["muni_lat"], row["muni_lon"]))
        return item


def collate(items: list[dict]) -> dict:
    """Zero-pad waveforms to the longest one in the batch; `audio_lengths` holds the true lengths."""
    lengths = torch.tensor([len(it["audio"]) for it in items])
    audio = torch.zeros(len(items), int(lengths.max()))
    for i, it in enumerate(items):
        audio[i, : lengths[i]] = it["audio"]
    batch = {
        "audio": audio,
        "audio_lengths": lengths,
        "n_clips": torch.tensor([it["n_clips"] for it in items]),
        "dialect": torch.tensor([it["dialect"] for it in items]),
        "canton": torch.tensor([it["canton"] for it in items]),
        "zipcode": [it["zipcode"] for it in items],
        "client_id": [it["client_id"] for it in items],
        "path": [it["path"] for it in items],
    }
    if "bfs_nr" in items[0]:
        # Speaker location: municipality centroid, [batch, 2] as (E, N) in metres and (lat, lon) in degrees.
        batch.update(bfs_nr=torch.tensor([it["bfs_nr"] for it in items]),
                     municipality=[it["municipality"] for it in items],
                     lv95=torch.tensor([it["lv95"] for it in items], dtype=torch.float64),
                     latlon=torch.tensor([it["latlon"] for it in items], dtype=torch.float64))
    return batch


class STT4SGDataModule:
    """Builds the three splits with shared label vocabularies and hands out their dataloaders.

    `split_dir` replaces the official split with train/valid/test.tsv from that folder. With
    `geo_table`, every item gets its speaker's municipality and coordinates; clips whose zip code has
    no municipality are dropped.

    `segment_duration` (seconds) joins clips of one speaker into items of about that length: the
    train grouping changes every epoch, valid and test use a fixed one. `null` keeps single clips.

    With `only_split` (e.g. "test"), only that split's tsv is read and it stands in for train, valid
    and test, so the whole pipeline runs on a partial download (smoke tests). Label ids then come
    from that split alone.
    """

    def __init__(self, data_dir: str, audio_h5: str, train_split: str = "train_all",
                 split_dir: str | None = None, geo_table: str | None = None,
                 only_split: str | None = None, sample_rate: int = 16000, max_duration: float | None = None,
                 segment_duration: float | None = None, segment_gap: float = 0.0, seed: int = 1234,
                 batch_size: int = 32, num_workers: int = 4):
        if only_split is not None and only_split not in SPLITS:
            raise ValueError(f"only_split must be one of {SPLITS}, got {only_split!r}")
        self.data_dir = Path(data_dir)
        self.audio_h5 = Path(audio_h5)
        self.train_split = train_split
        self.split_dir = Path(split_dir) if split_dir is not None else None
        self.geo_table = Path(geo_table) if geo_table is not None else None
        self.only_split = only_split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.segment_duration = segment_duration
        self.segment_gap = segment_gap
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets: dict[str, STT4SGDataset] = {}
        self.vocabs: dict[str, LabelVocab] = {}

    def setup(self):
        if not self.audio_h5.is_file():
            raise FileNotFoundError(
                f"{self.audio_h5} does not exist: pack the clips first "
                "(sbatch scripts/pack_stt4sg_h5.submit <data_dir>) or set data.audio_h5")
        index, h5_sample_rate = read_h5_index(self.audio_h5)
        if h5_sample_rate != self.sample_rate:
            log.warning("%s holds %d Hz audio, resampling every clip to %d Hz on the fly",
                        self.audio_h5, h5_sample_rate, self.sample_rate)
        if self.split_dir is not None and not self.split_dir.is_dir():
            raise FileNotFoundError(f"{self.split_dir} does not exist: create it with "
                                    "scripts/make_speaker_split.py or set data.split_dir=null")
        if self.only_split is not None:
            log.warning("only_split=%s: using it as train, valid and test", self.only_split)
            meta = read_split(self.data_dir, self.only_split, self.train_split, self.split_dir)
            metas = {split: meta for split in SPLITS}
        else:
            metas = {split: read_split(self.data_dir, split, self.train_split, self.split_dir) for split in SPLITS}
        if self.split_dir is not None and self.only_split is None:
            self._check_disjoint(metas)
        if self.geo_table is not None:
            metas = self._add_locations(metas)
        # Vocabularies come from every split so a label missing from train still gets an id.
        all_meta = pd.concat(metas.values())
        self.vocabs = {col: LabelVocab(all_meta[col]) for col in ("dialect_region", "canton")}
        for split, meta in metas.items():
            if self.max_duration is not None:
                meta = meta[meta["duration"] <= self.max_duration]
            # Clips that failed to decode while packing are missing from the file.
            packed = meta.join(index, on="path", how="inner")
            if missing := len(meta) - len(packed):
                log.warning("%s: %d clips are not in %s and are skipped", split, missing, self.audio_h5)
            meta = packed
            if meta.empty:
                raise ValueError(
                    f"no {split} clip found in {self.audio_h5}: it holds {len(index)} clips "
                    f"(e.g. {list(index.index[:3])}), the tsv in {self.data_dir} lists e.g. "
                    f"{list(metas[split]['path'][:3])}. An empty file means packing could not decode the "
                    "mp3s (see the pack job's log); otherwise it was packed from a different release.")
            self.datasets[split] = STT4SGDataset(meta, self.audio_h5, self.vocabs, self.sample_rate,
                                                 h5_sample_rate, self.segment_gap)
            log.info("%s: %d clips, %.1f h, %d speakers", split, len(meta),
                     meta["duration"].sum() / 3600, meta["client_id"].nunique())

    def _add_locations(self, metas: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        if not self.geo_table.is_file():
            raise FileNotFoundError(f"{self.geo_table} does not exist: run scripts/build_zip_geo.py "
                                    f"{self.data_dir} or set data.geo_table=null")
        geo = read_geo_table(self.geo_table)
        located = {}
        for split, meta in metas.items():
            located[split] = meta.join(geo, on="zipcode", how="inner")
            if missing := len(meta) - len(located[split]):
                zips = sorted(set(meta["zipcode"].dropna()) - set(geo.index))
                log.warning("%s: %d clips have a zip code missing from %s and are dropped: %s",
                            split, missing, self.geo_table, zips[:20])
        return located

    @staticmethod
    def _check_disjoint(metas: dict[str, pd.DataFrame]):
        speakers = {split: set(meta["client_id"]) for split, meta in metas.items()}
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            if shared := speakers[a] & speakers[b]:
                raise ValueError(f"{len(shared)} speakers are in both {a} and {b}, e.g. {sorted(shared)[:3]}")

    def sampler(self, split: str, shuffle: bool | None = None) -> SpeakerSegmentSampler | None:
        """Segment sampler of a split (None without segment_duration); shuffles train by default."""
        if self.segment_duration is None:
            return None
        dataset = self.datasets[split]
        return SpeakerSegmentSampler(dataset.meta["client_id"], dataset.durations, self.segment_duration,
                                     shuffle=split == "train" if shuffle is None else shuffle, seed=self.seed)

    def dataloader(self, split: str, shuffle: bool | None = None, collate_fn=None,
                   batch_size: int | None = None) -> DataLoader:
        """Dataloader of a split. Train is shuffled by default; `shuffle=False` gives the fixed order of
        `sampler(split, shuffle=False).segments(0)` (or of the clips). `collate_fn` replaces `collate`."""
        shuffle = split == "train" if shuffle is None else shuffle
        sampler = self.sampler(split, shuffle)
        return DataLoader(self.datasets[split], batch_size=batch_size or self.batch_size,
                          shuffle=shuffle and sampler is None, sampler=sampler,
                          num_workers=self.num_workers, collate_fn=collate_fn or collate,
                          pin_memory=torch.cuda.is_available(), persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> DataLoader:
        return self.dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.dataloader("valid")

    def test_dataloader(self) -> DataLoader:
        return self.dataloader("test")
