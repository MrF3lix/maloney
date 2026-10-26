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

### Grid

`maloney.geo` puts Switzerland on a square LV95 grid (`geo.grid.cell_size`, default 10 km; only cells
overlapping Switzerland are kept, built from `data/geo/cantons.geojson`) and maps municipalities onto
it as in Ahmadi & Sennrich (2026): `geo.mapping.mode=centroid` (the cell of the municipality centroid),
`uniform` (evenly over the area of the municipality and its neighbours in the same canton,
`geo.mapping.neighbour_distance` metres, 0 = adjacent; needs `data/geo/municipalities.geojson`) or
`gaussian` (kernel of `geo.mapping.sigma` metres).
`playground/02_grid_modes.ipynb` shows the three modes; `uv run pytest` tests them.

### Whisper embeddings

Experiments on frozen embeddings read them precomputed: `scripts/extract_whisper_embeddings.py` runs
a Whisper encoder (default `openai/whisper-large-v3`, any Whisper checkpoint via `--model`) over every
clip of `audio_16k.h5` and stores the mean-pooled hidden states of 8 evenly spaced layers (`--layers`)
in `<data_dir>/embeddings/<model>.h5`. Clips are padded to Whisper's 30 s input, and only frames with
audio are averaged. A job that hits the time limit resumes when submitted again:

```bash
sbatch scripts/extract_whisper_embeddings.submit data/stt4sg-350_v2-1
```

`maloney.embeddings.EmbeddingStore` loads a layer and returns embeddings per clip (`for_meta`), per
segment (`for_segments`, for `SpeakerSegmentSampler` groups) or per speaker (`by_speaker`), pooling
clips weighted by their number of frames.

### Baselines

`uv run baseline` fits one method on the precomputed embeddings of the train split and evaluates
valid and test with the paper's metrics (dialect-region accuracy/F1, cell accuracy, Acc@0-3,
mean/median km), per unit and per speaker. `method=` picks B1-M2 (`configs/method/`):

| method | | mapping mode matters for |
|---|---|---|
| `b1` | logistic regression on dialect region | nothing (no grid) |
| `b2` | nearest cell prototype (cosine) | the predictions |
| `m1` | logistic regression on centroid cells | recovering the region |
| `m2` | prototype network (projection + temperature) | recovering the region |

```bash
uv run baseline method=b2 geo.mapping.mode=gaussian data.split_dir=data/stt4sg-350_v2-1/splits/speakers_v1
uv run baseline -m method=b1,b2,m1,m2 geo.mapping.mode=centroid,uniform,gaussian unit=segment data.segment_duration=30
```

`unit=clip|segment|speaker` sets what is classified; `embeddings.layer` the Whisper layer. Metrics go to
`metrics.json`, per-unit predictions to `predictions_<split>_<unit>.csv` in the run directory.

### M3: fine-tuned Whisper

`uv run train` fine-tunes a Whisper encoder (`model.name`, default whisper-large-v3) with a linear head
over the centroid cells of the training municipalities (cross-entropy, AdamW, warmup + linear decay,
bf16/fp16 on CUDA). Items are clips, or segments with `data.segment_duration`. Every epoch the head is
evaluated on valid; `checkpoints/best.pt` is kept by `monitor` (default valid median km) and
`checkpoints/last.pt` allows resuming (`train.submit` does this when given the same `RUN_DIR`).
Afterwards, as in the paper, the head is replaced by nearest-prototype inference: train/valid/test are
re-embedded with the best weights and evaluated in every mode of `prototype_modes` (plus the head) into
`metrics.json`, with predictions per mode.

```bash
RUN_NAME=m3 sbatch train.submit optim.lr=2e-5 trainer.epochs=5 model.pool_layer=24
uv run train run_name=smoke logger=none model.name=openai/whisper-tiny trainer.device=mps \
    trainer.epochs=1 trainer.limit_train_batches=10 trainer.limit_eval_batches=5   # local smoke test
```

### Results

`playground/03_results.ipynb` collects the runs in `logs/` (config, `metrics.json`, predictions) into tables
(best run per method and mode, chosen on valid, reported on test), sweep curves, maps, dialect-region
confusions and the error against the distance to the nearest training municipality. Its first cell selects
the runs to report. Copy the results from the cluster without the checkpoints:

```bash
rsync -av --include='*/' --include='metrics.json' --include='*.yaml' --include='predictions_*.csv' --exclude='*' <cluster>:<repo>/logs/ logs/
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
