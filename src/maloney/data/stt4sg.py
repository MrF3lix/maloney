"""STT4SG-350 (Swiss German speech) with its official train/valid/test split.

Expected layout of `data_dir` (the release as downloaded, plus the packed audio):

    data_dir/
        train_all.tsv  train_balanced.tsv  valid.tsv  test.tsv
        audio_16k.h5                          # all clips decoded to int16, see below

`scripts/extract_stt4sg.submit` extracts the mp3s to `clips/`, then `scripts/pack_stt4sg_h5.submit`
decodes them into `audio_16k.h5`: the flat int16 dataset `audio` plus, per clip, its tsv `path`,
`offset` and `length`. Each tsv row has the columns
path, duration, sentence, sentence_source, client_id, dialect_region, canton, zipcode, age, gender.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soxr
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test")
# The release ships two training lists: everything, or one balanced across dialect regions.
TRAIN_FILES = {"train_all": "train_all.tsv", "train_balanced": "train_balanced.tsv"}


def read_split(data_dir: str | Path, split: str, train_split: str = "train_all") -> pd.DataFrame:
    """Read one split's metadata; `split` is train/valid/test, `train_split` picks the train list."""
    file = TRAIN_FILES[train_split] if split == "train" else f"{split}.tsv"
    # quoting=3 (QUOTE_NONE): sentences contain quotes, and zipcodes must stay strings.
    return pd.read_csv(Path(data_dir) / file, sep="\t", quoting=3, dtype={"zipcode": str, "client_id": str})


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


class STT4SGDataset(Dataset):
    """One split of STT4SG-350; items are mono waveforms at `sample_rate` plus speaker metadata.

    `meta` needs the columns `offset` and `length` locating each clip in the packed `audio_h5`.
    """

    def __init__(self, meta: pd.DataFrame, audio_h5: str | Path, vocabs: dict[str, LabelVocab],
                 sample_rate: int = 16000, h5_sample_rate: int = 16000):
        self.meta = meta.reset_index(drop=True)
        self.audio_h5 = Path(audio_h5)
        self.vocabs = vocabs
        self.sample_rate = sample_rate
        self.h5_sample_rate = h5_sample_rate
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

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]
        return {
            "audio": torch.from_numpy(self.load_audio(row["offset"], row["length"])),
            "dialect": self.vocabs["dialect_region"][row["dialect_region"]],
            "canton": self.vocabs["canton"][row["canton"]],
            "zipcode": row["zipcode"],
            "client_id": row["client_id"],
            "path": row["path"],
        }


def collate(items: list[dict]) -> dict:
    """Zero-pad waveforms to the longest one in the batch; `audio_lengths` holds the true lengths."""
    lengths = torch.tensor([len(it["audio"]) for it in items])
    audio = torch.zeros(len(items), int(lengths.max()))
    for i, it in enumerate(items):
        audio[i, : lengths[i]] = it["audio"]
    return {
        "audio": audio,
        "audio_lengths": lengths,
        "dialect": torch.tensor([it["dialect"] for it in items]),
        "canton": torch.tensor([it["canton"] for it in items]),
        "zipcode": [it["zipcode"] for it in items],
        "client_id": [it["client_id"] for it in items],
        "path": [it["path"] for it in items],
    }


class STT4SGDataModule:
    """Builds the three splits with shared label vocabularies and hands out their dataloaders."""

    def __init__(self, data_dir: str, audio_h5: str, train_split: str = "train_all",
                 sample_rate: int = 16000, max_duration: float | None = None,
                 batch_size: int = 32, num_workers: int = 4):
        self.data_dir = Path(data_dir)
        self.audio_h5 = Path(audio_h5)
        self.train_split = train_split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
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
        metas = {split: read_split(self.data_dir, split, self.train_split) for split in SPLITS}
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
                raise ValueError(f"no {split} clip found in {self.audio_h5}; was it packed from {self.data_dir}?")
            self.datasets[split] = STT4SGDataset(meta, self.audio_h5, self.vocabs, self.sample_rate, h5_sample_rate)
            log.info("%s: %d clips, %.1f h, %d speakers", split, len(meta),
                     meta["duration"].sum() / 3600, meta["client_id"].nunique())

    def dataloader(self, split: str) -> DataLoader:
        return DataLoader(self.datasets[split], batch_size=self.batch_size, shuffle=split == "train",
                          num_workers=self.num_workers, collate_fn=collate,
                          pin_memory=torch.cuda.is_available(), persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> DataLoader:
        return self.dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.dataloader("valid")

    def test_dataloader(self) -> DataLoader:
        return self.dataloader("test")
