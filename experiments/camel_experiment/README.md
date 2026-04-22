# CAMEL Experiment

ATLAS-style CAMEL data preparation for two dataset variants:

- `camel.bin`: raw bytes of the downloaded CAMEL HDF5 snapshot
- `camel_float32.bin`: extracted float32 feature matrix from `PartType0`

The float32 dataset is also split into:

- `camel_train.bin`
- `camel_val.bin`
- `camel_test.bin`

The raw-byte dataset is split into:

- `camel_raw_train.bin`
- `camel_raw_val.bin`
- `camel_raw_test.bin`

## Quick Start

From repository root:

```bash
# Local machine (smaller prepared dataset)
python experiments/camel_experiment/download.py --profile local --all-steps

# Server / MI300X (full-size pipeline)
python experiments/camel_experiment/download.py --profile server --all-steps
```

This downloads `snapshot_024.hdf5`, creates both `camel.bin` and
`camel_float32.bin`, and writes all split files.

Notes:

- `--profile local` defaults to float32-only preparation with `--max-rows 200000`.
- `--profile local` also defaults to `--max-h5-mb 500`, so `data/camel_snapshot_024.hdf5` is compacted to about 500 MB.
- `--profile server` keeps full raw + float32 preparation.
- If you need a smaller network download, pass a smaller source URL via `--src`.

Explicit 500 MB local command:

```bash
python experiments/camel_experiment/download.py --profile local --all-steps --max-h5-mb 500
```

## Step-by-step

```bash
# 1) Download snapshot
python experiments/camel_experiment/download.py --download

# 2) Build both binary variants
python experiments/camel_experiment/download.py --extract-raw --extract-float32

# 3) Split raw-byte corpus
python experiments/camel_experiment/download.py --split-raw

# 4) Split float32 corpus (record-aligned by row width)
python experiments/camel_experiment/download.py --split-float32
```

Optional quick test extraction:

```bash
python experiments/camel_experiment/download.py --profile local --all-steps --max-rows 200000
```

## Training Configs

- `experiments/camel_experiment/camel_experiment.yaml` for float32 split files (local GPU smoke-test profile)
- `experiments/camel_experiment/camel_raw_experiment.yaml` for raw-byte split files (local GPU smoke-test profile)

For MI300X parallel training, the repository-level config still uses:

- `configs/camel_mi300x.yaml`
