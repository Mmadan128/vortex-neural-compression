# HEPMC Experiment

This folder contains a helper script to fetch a HEPMC sample from CERN EOS Open Data, extract it, and produce binary outputs for training and benchmarking.

## What it does
- Downloads the archive from EOS:
  - `root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc16_13TeV/HEPMC.43646133._000001.tar.gz.1`
- Safely extracts the archive to a temporary directory
- Locates the contained HEPMC payload
- Writes to `data/`:
  - `hepmc_full.hepmc` — full HEPMC byte stream
  - `hepmc_full_train.hepmc` — 80% train split
  - `hepmc_full_val.hepmc` — 10% validation split
  - `hepmc_full_test.hepmc` — 10% test split

## Requirements
- Python 3.8+
- Packages (already listed in the repo `requirements.txt`):
  - `requests`, `tqdm`
- Optional fallback tool:
  - `xrdcp` (from XRootD) — used only if HTTPS streaming fails

Install dependencies at the repo root:

```bash
pip install -r requirements.txt
```

## Usage
Run from this directory (or pass the path):

```bash
# Download ATLAS HEPMC sample, extract, and write full + train/val/test splits
python download.py

# Force re-download and overwrite all outputs
python download.py --force

# Use a custom URL (supports root:// or https://)
python download.py --url <your-url>
```

Outputs will be created in `data/`:
- `hepmc_full.hepmc`
- `hepmc_full_train.hepmc`
- `hepmc_full_val.hepmc`
- `hepmc_full_test.hepmc`

## Training (AMD MI300X)

```bash
python scripts/train.py --config experiments/hepmc_experiment/hepmc_experiment.yaml
```

Config is pre-tuned for MI300X: batch=256, bf16, 25M params, 300k steps.

## Notes
- Splits are at raw byte offsets (80/10/10). Suitable for byte-level autoregressive models.
- If the archive contains `*.hepmc.gz`, the script transparently gunzips it.
- If neither `.hepmc` nor `.hepmc.gz` is found, the largest file in the archive is used as a heuristic fallback.
- If HTTPS fails and `xrdcp` is available, the script automatically falls back to `xrdcp`.
