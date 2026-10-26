# Maloney

Dialect identification from speech on STT4SG-350, using the speakers' geography (ZIP codes)
rather than only dialect labels.

## Setup

```bash
uv sync
```

### Data

The loader expects `${DATA_DIR}/stt4sg-350_v2.1/` (`DATA_DIR` defaults to `data/`, can be set in `.env`)
with the release's tsv files and the clips extracted to `clips/`:

```bash
sbatch scripts/extract_stt4sg.submit data/stt4sg-350_v2.1
```

The official split is used: `train_all.tsv` (or `data.train_split=train_balanced`), `valid.tsv`, `test.tsv`.

```bash
uv run train run_name=dev logger=none
```
