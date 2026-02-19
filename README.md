# Vortex-Codec 🌀

**Neural Lossless Compression via Compressive Transformers + Arithmetic Coding**

> Achieves **3.04 BPD** on ATLAS jet detector data — **39% better** than Gzip/Zstd.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download real ATLAS data from CERN EOS → experiments/atlas_experiment/
python experiments/atlas_experiment/download.py --all-steps

# 2. Train on the ATLAS dataset
python scripts/train.py --config experiments/atlas_experiment/config.yaml

# 3. Compress a file
python scripts/compress.py \
    --model  experiments/atlas_experiment/checkpoints/best.pt \
    --input  experiments/atlas_experiment/data/atlas_200m.bin \
    --output experiments/atlas_experiment/data/atlas_200m.vxc \
    --config experiments/atlas_experiment/config.yaml

# 4. Decompress
python scripts/decompress.py \
    --model  experiments/atlas_experiment/checkpoints/best.pt \
    --input  experiments/atlas_experiment/data/atlas_200m.vxc \
    --output experiments/atlas_experiment/data/atlas_200m_recovered.bin \
    --config experiments/atlas_experiment/config.yaml

# 5. Evaluate vs Gzip / Zstd
python scripts/evaluate.py \
    --model experiments/atlas_experiment/checkpoints/best.pt \
    --data  experiments/atlas_experiment/data/atlas_200m.bin \
    --config experiments/atlas_experiment/config.yaml
```

## Repository Layout

```
vortex-codec/
├── vortex/                        # core Python package
│   ├── models/
│   │   ├── compressive_transformer.py   # base O(1)-memory model
│   │   └── optimized_transformer.py     # Flash Attn2 + KV cache + SwiGLU
│   ├── compression/arithmetic_coding.py
│   ├── data/dataset.py
│   └── utils/training.py
├── scripts/
│   ├── train.py
│   ├── compress.py
│   ├── decompress.py
│   └── evaluate.py
├── experiments/
│   └── atlas_experiment/          # self-contained experiment
│       ├── download.py            # fetch ATLAS HDF5 from CERN EOS
│       ├── config.yaml            # experiment hyperparameters
│       ├── data/                  # atlas.bin, atlas_200m.bin, atlas.meta.json
│       └── checkpoints/           # best.pt saved here
├── configs/                       # hardware-specific base configs
│   ├── colab_t4.yaml
│   ├── rtx4070_8gb.yaml
│   ├── default.yaml
│   ├── rtx4090_24gb.yaml
│   └── amd_mi300x.yaml
├── tests/
└── docs/
```

## Hardware Configs
| File | GPU | VRAM | Params |
|------|-----|------|--------|
| `colab_t4.yaml`    | T4 (Colab)  | 15 GB  | 3.2 M  |
| `rtx4070_8gb.yaml` | RTX 4070    | 8 GB   | 8.5 M  |
| `default.yaml`     | RTX 3090/80 | 12 GB  | 14.8 M |
| `rtx4090_24gb.yaml`| RTX 4090    | 24 GB  | 28 M   |
| `amd_mi300x.yaml`  | MI300X      | 192 GB | 60 M+  |

## Architecture
- 8-layer Transformer, d=512, 8 heads
- Compressive Attention → **O(1)** memory via 4:1 Conv compression
- Flash Attention 2 (auto-fallback to `F.scaled_dot_product_attention`)
- KV Cache for ~10× faster autoregressive decompression
- Arithmetic Coding via `torchac`
- SwiGLU feed-forward (LLaMA-style)

## ATLAS Dataset
- Source: CERN EOS `root://eospublic.cern.ch//eos/opendata/atlas/datascience/ATLAS-FTAG-2023-05/`
- Format: HDF5 → extracted to raw binary (`atlas.bin`) via `download.py`
- Training subset: `atlas_200m.bin` (200 MB slice)
- Structured dtype: 30 fields including `pt_btagJes`, `GN2v01_pb`, kinematics, labels
