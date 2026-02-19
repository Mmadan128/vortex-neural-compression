# ATLAS Neural Compression Experiment

Byte-level neural lossless compression on ATLAS jet b-tagging data.

This experiment evaluates autoregressive entropy modeling on structured HEP data and benchmarks it against classical compressors.

---

## Overview

This pipeline:

- Converts ATLAS HDF5 data into a raw byte stream
- Trains a byte-level neural predictor
- Compresses using arithmetic coding
- Verifies exact reconstruction
- Benchmarks against gzip and zstd

All compression is strictly lossless.

---

## Dataset

**Source**  
ATLAS open data  
`ATLAS-FTAG-2023-05/mc-flavtag-ttbar-small.h5`

**Content**

Each jet record contains 30 structured fields including:

- Tagger scores (GN2v01, DL1 variants)
- Kinematics (pt, eta, phi, mass)
- Truth labels
- Event identifiers

After conversion to raw bytes:

- Record size: 102 bytes per jet
- Data is stored as a contiguous binary stream

For faster experimentation, a 200 MB slice is used during training and benchmarking.

---

## Quick Start

From repository root:

1) Install dependencies

```bash
pip install -r requirements.txt
```
2) Download ATLAS data and extract raw binary
```bash
python experiments/atlas_experiment/download.py --all-steps
```
3) Create validation and test splits
```bash
python experiments/atlas_experiment/prepare.py

```

4) Train model
```bash
python scripts/train.py \
    --config experiments/atlas_experiment/config.yaml
```
5) Compress 200 MB slice
```bash
python scripts/compress.py \
    --model  experiments/atlas_experiment/checkpoints/best.pt \
    --input  experiments/atlas_experiment/data/atlas_200m.bin \
    --output experiments/atlas_experiment/data/atlas_200m.vxc \
    --config experiments/atlas_experiment/config.yaml
```
6) Decompress
```bash
python scripts/decompress.py \
    --model  experiments/atlas_experiment/checkpoints/best.pt \
    --input  experiments/atlas_experiment/data/atlas_200m.vxc \
    --output experiments/atlas_experiment/data/atlas_200m_recovered.bin \
    --config experiments/atlas_experiment/config.yaml
```