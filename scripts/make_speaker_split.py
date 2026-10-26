"""Make a speaker-disjoint train/valid/test split of STT4SG-350, balanced across dialect regions.

All clips of the release's tsv files are pooled and every speaker is assigned to exactly one split:
test and valid get the same number of speakers from every dialect region, all other speakers go to
train. Within a region, speakers from municipalities not yet in the split are taken first, so the
held-out speakers also spread out geographically. Only speakers whose zip code maps to a municipality
(`zip_municipality.csv` from scripts/build_zip_geo.py) are used, since the location is the target.

Writes <data_dir>/splits/<name>/{train,valid,test}.tsv (the release's columns) and speakers.csv (one
row per speaker with split, dialect region, municipality and amount of audio). Load it with
`data.split_dir=<data_dir>/splits/<name>`.

    uv run python scripts/make_speaker_split.py data/stt4sg-350_v2-1 --name speakers_v1
    uv run python scripts/make_speaker_split.py data/stt4sg-350_v2-1 --name speakers_v1 --test-per-region 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("make_speaker_split")

TSVS = ("train_all.tsv", "valid.tsv", "test.tsv")  # train_balanced is a subset of train_all
HELD_OUT = ("test", "valid")  # picked in this order, so test does not depend on --valid-per-region


def read_clips(data_dir: Path) -> pd.DataFrame:
    tsvs = [data_dir / t for t in TSVS if (data_dir / t).is_file()]
    if not tsvs:
        raise SystemExit(f"no tsv files in {data_dir}")
    log.info("pooling %s", ", ".join(t.name for t in tsvs))
    clips = pd.concat(pd.read_csv(t, sep="\t", quoting=3, dtype={"zipcode": str, "client_id": str}) for t in tsvs)
    return clips.drop_duplicates("path").reset_index(drop=True)


def speaker_table(clips: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    speakers = clips.groupby("client_id").agg(
        dialect_region=("dialect_region", lambda s: s.mode().iloc[0]),
        n_dialect_regions=("dialect_region", "nunique"), zipcode=("zipcode", lambda s: s.mode().iloc[0]),
        n_clips=("path", "size"), hours=("duration", lambda s: s.sum() / 3600))
    if (several := speakers["n_dialect_regions"] > 1).any():
        log.warning("%d speakers have clips with several dialect regions; using the most frequent one",
                    several.sum())
    speakers = speakers.drop(columns="n_dialect_regions").reset_index()
    speakers = speakers.merge(geo[["zipcode", "bfs_nr", "municipality"]], on="zipcode", how="left")
    if (unmapped := speakers["bfs_nr"].isna()).any():
        log.warning("%d speakers (%d clips) have a zip code without municipality and are left out: %s",
                    unmapped.sum(), speakers.loc[unmapped, "n_clips"].sum(),
                    speakers.loc[unmapped, "zipcode"].unique().tolist())
    return speakers[~unmapped].astype({"bfs_nr": int}).reset_index(drop=True)


def pick(region: pd.DataFrame, n: int, rng: np.random.Generator, taken: set[int]) -> pd.Index:
    """n random speakers of one region, those from municipalities not in `taken` first."""
    region = region.sample(frac=1, random_state=rng)
    # Rank 0 = first speaker of a new municipality, 1 = second one, ...; already used ones go last.
    rank = region.groupby("bfs_nr").cumcount() + region["bfs_nr"].isin(taken) * len(region)
    chosen = region.loc[rank.sort_values(kind="stable").index[:n]]
    taken.update(chosen["bfs_nr"])
    return chosen.index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="release folder with the tsv files and zip_municipality.csv")
    parser.add_argument("--name", default="speakers_v1", help="written to <data_dir>/splits/<name>/")
    parser.add_argument("--test-per-region", type=int, help="default: 10%% of all speakers, divided by regions")
    parser.add_argument("--valid-per-region", type=int, help="default: same as --test-per-region")
    parser.add_argument("--max-held-out", type=float, default=0.5,
                        help="largest fraction of a region's speakers that test + valid may take")
    parser.add_argument("--geo-table", type=Path, help="default: <data_dir>/zip_municipality.csv")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    out_dir = args.data_dir / "splits" / args.name
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} exists; pick another --name or pass --force (a split should not change "
                         "once results were reported on it)")
    geo_table = args.geo_table or args.data_dir / "zip_municipality.csv"
    if not geo_table.is_file():
        raise SystemExit(f"{geo_table} not found: run scripts/build_zip_geo.py {args.data_dir} first")

    clips = read_clips(args.data_dir)
    speakers = speaker_table(clips, pd.read_csv(geo_table, dtype={"zipcode": str}))
    per_region = speakers.groupby("dialect_region").size()
    n_test = args.test_per_region or max(1, round(0.1 * len(speakers) / len(per_region)))
    n_valid = args.valid_per_region if args.valid_per_region is not None else n_test
    too_small = per_region[per_region * args.max_held_out < n_test + n_valid]
    if len(too_small):
        raise SystemExit(f"{n_test} test + {n_valid} valid speakers per region would take more than "
                         f"{args.max_held_out:.0%} of these regions' speakers: {too_small.to_dict()}; "
                         "lower --test-per-region / --valid-per-region")

    rng = np.random.default_rng(args.seed)
    speakers["split"] = "train"
    for region, group in speakers.groupby("dialect_region"):
        taken: set[int] = set()
        for split, n in zip(HELD_OUT, (n_test, n_valid)):
            group = group[speakers.loc[group.index, "split"] == "train"]
            speakers.loc[pick(group, n, rng, taken), "split"] = split

    out_dir.mkdir(parents=True, exist_ok=True)
    clips = clips.merge(speakers[["client_id", "split"]], on="client_id")  # drops unmapped speakers
    for split, part in clips.groupby("split"):
        part.drop(columns="split").to_csv(out_dir / f"{split}.tsv", sep="\t", index=False, quoting=3)
    speakers.round({"hours": 3}).sort_values(["split", "dialect_region", "client_id"]).to_csv(
        out_dir / "speakers.csv", index=False)

    summary = speakers.pivot_table(index="dialect_region", columns="split", values="client_id",
                                   aggfunc="count", fill_value=0)[["train", "valid", "test"]]
    summary.loc["total"] = summary.sum()
    hours = speakers.groupby("split")["hours"].sum()
    log.info("speakers per dialect region and split:\n%s", summary.to_string())
    log.info("hours: %s", ", ".join(f"{s} {hours.get(s, 0):.1f}" for s in ("train", "valid", "test")))
    log.info("wrote %s; load it with data.split_dir=%s", out_dir, out_dir)


if __name__ == "__main__":
    main()
