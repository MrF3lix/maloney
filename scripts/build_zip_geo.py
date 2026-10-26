"""Map the speakers' zip codes of STT4SG-350 to municipalities, with polygons and coordinates.

Reference data, swisstopo's official directory of localities (Amtliches Ortschaftenverzeichnis,
https://www.swisstopo.admin.ch/de/amtliches-ortschaftenverzeichnis), in `--geo-dir` (default data/geo/):
  AMTOVZ/AMTOVZ_CSV_LV95.csv    every zip with its municipalities (BFS number) and their address shares
  AMTOVZ/AMTOVZ_ZIP.shp         zip polygons, LV95 (EPSG:2056)

Each zip is assigned to the municipality holding most addresses of its main locality (`muni_share`, %).
Municipality polygons are the zip polygons dissolved by that municipality, so a zip crossing a
municipal border counts entirely towards its main municipality. Coordinates are polygon centroids,
both in LV95 (E/N, metres) and WGS84 (lat/lon).

Outputs, next to the tsv files in `data_dir`:
  zip_municipality.csv      one row per zip code in the dataset: municipality, canton, zip and
                            municipality coordinates, speakers/clips/hours, dialect region
  geo.gpkg                  layers `municipalities` and `zips`: every polygon in Switzerland with
                            the dataset counts (0 where no speaker lives), for plotting

    uv run --extra geo python scripts/build_zip_geo.py data/stt4sg-350_v2-1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

log = logging.getLogger("build_zip_geo")

TSVS = ("train_all.tsv", "valid.tsv", "test.tsv")  # train_balanced is a subset of train_all
LV95, WGS84 = 2056, 4326


def read_speakers(data_dir: Path) -> pd.DataFrame:
    tsvs = [data_dir / t for t in TSVS if (data_dir / t).is_file()]
    if not tsvs:
        raise SystemExit(f"no tsv files in {data_dir}")
    log.info("reading %s", ", ".join(t.name for t in tsvs))
    clips = pd.concat(pd.read_csv(t, sep="\t", quoting=3, dtype={"zipcode": str, "client_id": str},
                                  usecols=["path", "duration", "client_id", "zipcode", "canton", "dialect_region"])
                      for t in tsvs).drop_duplicates("path")
    clips["zipcode"] = clips["zipcode"].str.strip()
    per_zip = clips.groupby("zipcode").agg(
        n_speakers=("client_id", "nunique"), n_clips=("path", "size"), hours=("duration", "sum"),
        dialect_region=("dialect_region", lambda s: s.mode().iloc[0]),
        n_dialect_regions=("dialect_region", "nunique"),
        canton_tsv=("canton", lambda s: s.mode().iloc[0]))
    per_zip["hours"] = (per_zip["hours"] / 3600).round(3)
    return per_zip.reset_index()


def with_coords(gdf: gpd.GeoDataFrame, prefix: str) -> gpd.GeoDataFrame:
    centroid = gdf.geometry.centroid
    wgs = centroid.to_crs(WGS84)
    return gdf.assign(**{f"{prefix}_E": centroid.x.round(1), f"{prefix}_N": centroid.y.round(1),
                         f"{prefix}_lat": wgs.y.round(6), f"{prefix}_lon": wgs.x.round(6)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="release folder with the tsv files")
    parser.add_argument("--geo-dir", type=Path, default=Path("data/geo"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    per_zip = read_speakers(args.data_dir)

    # One row per (zip + additional digit, municipality). The additional digits mark sub-localities,
    # often tiny (6010 00 Kriens LU, 6010 02 Pilatus Kulm OW), and the shares are per sub-locality,
    # so the zip goes to the majority municipality of its main sub-locality (lowest digit).
    amtovz = pd.read_csv(args.geo_dir / "AMTOVZ" / "AMTOVZ_CSV_LV95.csv", sep=";", encoding="utf-8-sig")
    amtovz = amtovz.rename(columns={"PLZ4": "zipcode", "BFS-Nr": "bfs_nr", "Gemeindename": "municipality",
                                    "Kantonskürzel": "canton"})
    amtovz["zipcode"] = amtovz["zipcode"].astype(str)
    amtovz["muni_share"] = amtovz["Adressenanteil"].str.rstrip(" %").astype(float)
    main = amtovz[amtovz["Zusatzziffer"] == amtovz.groupby("zipcode")["Zusatzziffer"].transform("min")]
    bfs = main.sort_values("muni_share").drop_duplicates("zipcode", keep="last")
    bfs = bfs[["zipcode", "bfs_nr", "municipality", "canton", "muni_share"]]
    bfs["muni_share"] = bfs["muni_share"].round(1)

    # A zip can have several polygons (additional digits, e.g. 6315 00 and 6315 02): merge them.
    zips = gpd.read_file(args.geo_dir / "AMTOVZ" / "AMTOVZ_ZIP.shp").to_crs(LV95)
    zips = zips.assign(zipcode=zips["ZIP4"].astype(int).astype(str))[["zipcode", "geometry"]]
    zips = zips.dissolve("zipcode").reset_index().merge(bfs, on="zipcode", how="left")
    zips = with_coords(zips, "zip")

    munis = zips.dropna(subset=["bfs_nr"]).drop(columns="muni_share").dissolve(
        "bfs_nr", aggfunc={"municipality": "first", "canton": "first"})
    munis = with_coords(munis.reset_index().astype({"bfs_nr": int}), "muni")

    table = per_zip.merge(zips.drop(columns="geometry"), on="zipcode", how="left")
    table = table.merge(munis[["bfs_nr", "muni_E", "muni_N", "muni_lat", "muni_lon"]], on="bfs_nr", how="left")
    table["bfs_nr"] = table["bfs_nr"].astype("Int64")

    unmapped = table[table["bfs_nr"].isna() | table["zip_E"].isna()]
    if len(unmapped):
        log.warning("%d zips (%d speakers) have no municipality or polygon: %s", len(unmapped),
                    unmapped["n_speakers"].sum(), unmapped["zipcode"].tolist())
    mismatch = table[table["canton"].notna() & (table["canton"] != table["canton_tsv"])]
    if len(mismatch):
        log.warning("%d zips lie in another canton than the speakers' tsv canton (zip canton/tsv): %s",
                    len(mismatch), [f"{r.zipcode} {r.canton}/{r.canton_tsv}" for r in mismatch.itertuples()])

    cols = ["zipcode", "bfs_nr", "municipality", "muni_share", "canton", "canton_tsv", "dialect_region", "n_dialect_regions",
            "n_speakers", "n_clips", "hours", "zip_E", "zip_N", "zip_lat", "zip_lon",
            "muni_E", "muni_N", "muni_lat", "muni_lon"]
    out_csv = args.data_dir / "zip_municipality.csv"
    table[cols].sort_values("zipcode").to_csv(out_csv, index=False)

    counts = ["n_speakers", "n_clips", "hours"]
    zips_out = zips.merge(per_zip[["zipcode", *counts, "dialect_region"]], on="zipcode", how="left")
    munis_out = munis.merge(table.groupby("bfs_nr")[counts].sum().reset_index().astype({"bfs_nr": int}),
                            on="bfs_nr", how="left")
    for gdf in (zips_out, munis_out):
        gdf[counts] = gdf[counts].fillna(0)
    out_gpkg = args.data_dir / "geo.gpkg"
    out_gpkg.unlink(missing_ok=True)
    munis_out.to_file(out_gpkg, layer="municipalities", driver="GPKG")
    zips_out.to_file(out_gpkg, layer="zips", driver="GPKG")

    log.info("%d zips of %d speakers -> %d municipalities; wrote %s and %s (layers municipalities, zips)",
             len(table), table["n_speakers"].sum(), table["bfs_nr"].nunique(), out_csv, out_gpkg)


if __name__ == "__main__":
    main()
