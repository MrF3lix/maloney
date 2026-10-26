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

The official split is used: `train_all.tsv` (or `data.train_split=train_balanced`), `valid.tsv`, `test.tsv`.

```bash
uv run train run_name=dev logger=none
```
