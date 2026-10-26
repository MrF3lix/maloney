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

The official split is used: `train_all.tsv` (or `data.train_split=train_balanced`), `valid.tsv`, `test.tsv`.

```bash
uv run train run_name=dev logger=none
```

For a smoke test with only part of the release downloaded (e.g. `test.tsv` + `clips__test.tar`,
packed as above), `data.only_split=test` reads just that split and uses it for train, valid and test:

```bash
uv run train run_name=smoke logger=none data.only_split=test
```
