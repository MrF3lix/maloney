# Maloney

Dialect identification from speech on STT4SG-350, using the speakers' geography (ZIP codes)
rather than only dialect labels.

## Setup

```bash
uv sync --extra cu128 --extra wandb   # Volta+ GPUs; cu126 for Pascal (P40), rocm for AMD
uv sync --extra wandb                 # macOS / CPU: PyPI torch
```

### Data

The loader expects `${DATA_DIR}/stt4sg-350_v2.1/` (`DATA_DIR` defaults to `data/`, can be set in `.env`)
with the release's tsv files and clip tars. The clips are decoded straight from the tars into one
HDF5 file, `audio_16k.h5` (~40 GB, 16 kHz int16), which is what training reads; nothing is extracted:

```bash
sbatch scripts/pack_stt4sg_h5.submit data/stt4sg-350_v2.1
```

Once packed, the tars are no longer needed for training. Point `data.audio_h5` elsewhere (e.g. a node-local disk) if needed.

### Speaker locations

`scripts/build_zip_geo.py` maps every speaker zip code to its municipality (BFS number) with
centroid coordinates (LV95 and WGS84), using swisstopo's official directory of localities
(`AMTOVZ_CSV_LV95.csv` and `AMTOVZ_ZIP.shp` in `data/geo/AMTOVZ/`). It writes
`zip_municipality.csv` and `geo.gpkg` (polygons for plotting) next to the tsv files;
`playground/01_plot_speaker_map.ipynb` plots them.

```bash
uv sync --extra geo
uv run --extra geo python scripts/build_zip_geo.py data/stt4sg-350_v2-1
```

### Speaker-disjoint split

`scripts/make_speaker_split.py` pools all clips and assigns every speaker to one split: test and valid
get the same number of speakers per dialect region (from different municipalities where possible),
the rest goes to train. Only speakers with a known municipality are kept. Run it after
`build_zip_geo.py`; a split never changes once written (new `--name` for a new one):

```bash
uv run python scripts/make_speaker_split.py data/stt4sg-350_v2-1 --name speakers_v1 --test-per-region 8
uv run train run_name=dev logger=none data.split_dir=data/stt4sg-350_v2-1/splits/speakers_v1
```

Batches then carry each speaker's location: `bfs_nr`, `municipality`, `lv95` (E, N in m) and
`latlon` (municipality centroid).

`data.segment_duration=30` turns items into ~30 s of one speaker: consecutive clips are joined
(regrouped every epoch for train, fixed for valid/test, nothing dropped), optionally with
`data.segment_gap` seconds of silence in between.

Without `data.split_dir`, the official split is used: `train_all.tsv` (or `data.train_split=train_balanced`), `valid.tsv`, `test.tsv`.

```bash
uv run train run_name=dev logger=none
```

For a smoke test with only part of the release downloaded (e.g. `test.tsv` + `clips__test.tar`,
packed as above), `data.only_split=test` reads just that split and uses it for train, valid and test:

```bash
uv run train run_name=smoke logger=none data.only_split=test
```
