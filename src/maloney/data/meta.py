"""Metadata of STT4SG-350 without torch: the split tsv files, the geo table and the packed audio's index.

Imported by maloney.data.stt4sg (which re-exports everything here) and by tools that only handle
files, such as the anonymizer, so they run without a working torch (e.g. on a CPU node).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd

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
    return read_tsv(file)


def read_tsv(file: str | Path, **kwargs) -> pd.DataFrame:
    """Read a tsv in the release's format, with zip codes as strings ("8001")."""
    # quoting=3 (QUOTE_NONE): sentences contain quotes, and zipcodes must stay strings. Some copies of
    # train_all/train_balanced.tsv store them as floats ("8001.0"), which would match no zip elsewhere.
    meta = pd.read_csv(file, sep="\t", quoting=3, dtype={"zipcode": str, "client_id": str}, **kwargs)
    if "zipcode" in meta:
        meta["zipcode"] = meta["zipcode"].str.strip().str.removesuffix(".0")
    return meta


def read_geo_table(geo_table: str | Path) -> pd.DataFrame:
    """zip code -> municipality and centroid coordinates (GEO_COLUMNS), indexed by zip code."""
    geo = pd.read_csv(geo_table, dtype={"zipcode": str}).rename(columns={"canton": "muni_canton"})
    geo = geo[["zipcode", *GEO_COLUMNS]]
    return geo.dropna(subset=["bfs_nr", "muni_E"]).astype({"bfs_nr": int}).set_index("zipcode")


def read_h5_index(audio_h5: str | Path) -> tuple[pd.DataFrame, int]:
    """The packed file's clip index (path -> offset, length) and its sample rate.

    `audio` may be a virtual dataset linking to other files (the synthetic test split of
    anonymizer.pack); a missing linked file would read as silence, so it is an error here.
    """
    with h5py.File(audio_h5, "r") as f:
        if f["audio"].is_virtual:
            for source in {vs.file_name for vs in f["audio"].virtual_sources()} - {"."}:
                if not (Path(audio_h5).parent / source).is_file():
                    raise FileNotFoundError(f"{audio_h5} links to the audio of {source} (relative to its folder), "
                                            "which does not exist")
        index = pd.DataFrame({"offset": f["offset"][:], "length": f["length"][:]},
                             index=pd.Index(f["path"].asstr()[:], name="path"))
        return index, int(f.attrs["sample_rate"])

