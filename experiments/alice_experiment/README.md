# ALICE Experiment

This experiment adds the same dataset workflow used by ATLAS/HEPMC for ALICE ROOT data:

1. get a ROOT file (URL or local path)
2. convert selected branches to padded float32 `.bin` + `.meta.json`
3. create event-level `train/val/test` splits (80/10/10)
4. train with `scripts/train.py`

## Requirements

The converter uses ROOT Python readers:

```bash
pip install uproot awkward
```

## Prepare data

From repository root:

```bash
# Atlas-style default (auto-download ALICE from CERN OpenData record 1105)
python experiments/alice_experiment/download.py --all-steps

# Quick local smoke test (small event sample)
python experiments/alice_experiment/download.py --all-steps --small-test

# Option A: local ALICE ROOT file
python experiments/alice_experiment/download.py \
    --input-root /path/to/alice.root

# Option B: download from URL first
python experiments/alice_experiment/download.py \
    --url https://example.org/alice_sample.root
```

Outputs are written under `experiments/alice_experiment/data/`:

- `alice.root` (if `--url` was used)
- `alice_events.bin`
- `alice_events.meta.json`
- `alice_events_train.bin`
- `alice_events_val.bin`
- `alice_events_test.bin`

Note: for ALICE ROOT files with complex custom classes, the converter automatically skips
branches that `uproot` cannot interpret and keeps readable numeric/list-numeric branches.

Useful options:

```bash
# Choose a specific tree in the ROOT file
python experiments/alice_experiment/download.py --input-root /path/to/alice.root --tree Events

# Limit events and overwrite outputs
python experiments/alice_experiment/download.py \
    --input-root /path/to/alice.root \
    --nmax 200000 \
    --force

# Choose a different CERN OpenData record id for auto-download
python experiments/alice_experiment/download.py --all-steps --record-id 1105

# Large corpus mode (downloads multiple ROOT files until target size)
python experiments/alice_experiment/download.py \
    --target-gb 6 \
    --record-id 1105 \
    --force

# Optional: raw-byte corpus instead of float32 features
python experiments/alice_experiment/download.py \
    --target-gb 6 \
    --large-format raw \
    --record-id 1105 \
    --force
```

`--small-test` changes defaults to a compact run for local validation:

- `nmax=2048`
- `max_list_len=8`
- output files: `alice_small.bin`, `alice_small.meta.json`, and split variants

`--target-gb` mode defaults to a multi-file float32 corpus and writes:

- `alice_large.bin`
- `alice_large_train.bin`
- `alice_large_val.bin`
- `alice_large_test.bin`
- `alice_large.meta.json`

If `--large-format raw` is used, it creates a raw-byte corpus and writes:

- `alice_raw.bin`
- `alice_raw_train.bin`
- `alice_raw_val.bin`
- `alice_raw_test.bin`
- `alice_raw.meta.json`

## Train

```bash
python scripts/train.py --config experiments/alice_experiment/alice_experiment.yaml

# For large raw-byte corpus mode
python scripts/train.py --config experiments/alice_experiment/alice_raw_large.yaml
```

## Evaluate

```bash
python scripts/evaluate.py \
    --model experiments/alice_experiment/checkpoints/best.pt \
    --data experiments/alice_experiment/data/alice_events_test.bin \
    --config experiments/alice_experiment/alice_experiment.yaml \
    --device cuda \
    --batch-size 256
```
