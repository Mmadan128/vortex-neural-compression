# Vortex-Codec

**Neural Lossless Compression via Compressive Transformers + Arithmetic Coding**

Byte-level neural codec targeting IEEE-754 float32 binary data (e.g. ATLAS FTAG jet physics).  
Outperforms gzip/zlib/lzma by learning byte-level entropy structure directly from data.

---

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
    --input  experiments/atlas_experiment/data/mc-flavtag-ttbar-medium.bin \
    --output experiments/atlas_experiment/data/mc-flavtag-ttbar-medium.vxc \
    --config experiments/atlas_experiment/config.yaml

# 4. Decompress
python scripts/decompress.py \
    --model  experiments/atlas_experiment/checkpoints/best.pt \
    --input  experiments/atlas_experiment/data/mc-flavtag-ttbar-medium.vxc \
    --output experiments/atlas_experiment/data/mc-flavtag-ttbar-medium_recovered.bin \
    --config experiments/atlas_experiment/config.yaml

# 5. Evaluate vs Gzip / Zstd  (1 GB sample, AMD MI300X)
python scripts/evaluate.py \
    --model      experiments/atlas_experiment/checkpoints/best.pt \
    --data       experiments/atlas_experiment/data/mc-flavtag-ttbar-medium.bin \
    --config     experiments/atlas_experiment/config.yaml \
    --device     cuda \
    --batch-size 256

# Other HEP datasets (same train/eval flow)

# HEPMC
python experiments/hepmc_experiment/download.py --num-files 10
python scripts/train.py --config experiments/hepmc_experiment/hepmc_experiment.yaml

# CMS (NanoAOD)
python experiments/cms_experiment/download.py
python scripts/train.py --config experiments/cms_experiment/cms_experiment.yaml

# ALICE (ROOT)
python experiments/alice_experiment/download.py --all-steps
python scripts/train.py --config experiments/alice_experiment/alice_experiment.yaml
```

---

## Repository Layout

```
vortex-codec/
├── vortex/                              # core Python package
│   ├── models/
│   │   ├── __init__.py                  # re-exports all public symbols
│   │   ├── compressive_transformer.py   # base model (CompressiveTransformer)
│   │   └── optimized_transformer.py     # production model (OptimisedCompressiveTransformer)
│   ├── compression/
│   │   └── arithmetic_coding.py         # torchac encode/decode + BPD metric
│   ├── data/
│   │   └── dataset.py                   # make_loaders() for binary / HDF5 files
│   └── utils/
│       ├── training.py                  # LR schedule, checkpointing, EarlyStopping
│       └── zipnn.py                     # Huffman post-training weight compression
├── scripts/
│   ├── train.py                         # full training loop (CATWrapper, AMP, TensorBoard)
│   ├── compress.py                      # file → .vxc bitstream
│   ├── decompress.py                    # .vxc bitstream → file
│   ├── evaluate.py                      # BPD vs gzip / zlib / lzma baselines
│   └── compress_weights.py              # apply ZipNN compression to a checkpoint
├── experiments/
│   ├── atlas_experiment/                # ATLAS FTAG HDF5 -> .bin splits
│   ├── camel_experiment/                # CAMEL HDF5 -> raw + float32 .bin splits
│   ├── hepmc_experiment/                # ATLAS HEPMC tarballs -> .hepmc splits
│   ├── cms_experiment/                  # CMS NanoAOD ROOT -> padded float32 .bin
│   ├── cms_experiment_lg/               # Original large-dataset CMS pipeline
│   └── alice_experiment/                # ALICE ROOT -> padded float32 .bin
├── configs/                             # hardware-specific base configs
│   ├── colab_t4.yaml
│   ├── rtx4070_8gb.yaml
│   ├── default.yaml
│   ├── rtx4090_24gb.yaml
│   └── amd_mi300x.yaml
├── tests/
│   └── test_basic.py
└── docs/
    ├── ARCHITECTURE_COMPARISON.md       # v1 vs v3 component-by-component diff
    └── HARDWARE_GUIDE.md
```

---

## Architecture

### Overview

Vortex-Codec is a **byte-level autoregressive model**: given a stream of bytes it predicts a probability distribution over the next byte, and uses arithmetic coding (`torchac`) to encode/decode the stream losslessly. Lower predicted cross-entropy = better compression.

The codebase contains two model variants, both in `vortex/models/`:

| Class | File | Use |
|---|---|---|
| `CompressiveTransformer` | `compressive_transformer.py` | Reference / lightweight |
| `OptimisedCompressiveTransformer` | `optimized_transformer.py` | Production (Flash Attn2, KV cache, RMSNorm) |
| `CATWrapper` | `optimized_transformer.py` | Dynamic chunk scheduler wrapping either model |

---

### `compressive_transformer.py` — Base Model

#### `TDTEmbedding`
Per-type embedding for IEEE-754 float32 byte streams.  
Each of the 4 byte positions within a `float32` (mantissa-low through sign/exponent-high) gets its own `nn.Embedding(256, d_model)` lookup table, since they have very different entropy profiles. An additional learnable `type_scale` vector (softmax-normalised) gates each table's contribution.

```
byte (0–255) ──► table[ t % 4 ]  (one of 4 typed tables, scale-gated)
                       ↓
                 h  (B, T, d_model)
```

#### `LearnableTokenEviction` (LTE)
Content-adaptive token selection replacing strided `Conv1d` downsampling.  
A lightweight depthwise + pointwise scorer produces per-token importance scores; the top-`k` (where `k = ceil(T / rate)`) tokens are kept in original temporal order. A straight-through soft gate (sigmoid-weighted) keeps the operation end-to-end differentiable. A final `Conv1d` projection + `LayerNorm` produces the memory representation.

```
acts (B, T, D) ──► scorer ──► topk ──► soft-gate ──► proj+norm ──► (B, k, D)
```

#### `MemoryManager`
Thin wrapper around `LearnableTokenEviction`. Provides a `.compress(acts)` method used by attention layers to build compressed memory from past activations.

#### `CompressiveAttention`
Multi-head attention with two-tier memory:
- **Local stream**: causal `scaled_dot_product_attention` over the current window (`Q`, `K`, `V`).
- **Memory stream**: cross-attention from current queries into compressed past (`Km`, `Vm` from `MemoryManager`).
- **Infini-β gating**: a per-head learnable scalar `β = sigmoid(infini_beta)` mixes the two streams: `out = β·out_mem + (1−β)·out_local`. Initialised at 0 (all local) so training starts stable.
- Compressed memory is accumulated across chunks and capped at `window // 2` tokens (oldest dropped).

#### `SwiGLU`
Gated feed-forward block (Shazeer 2020). No bias, no dropout.  
`out = down( silu(gate(x)) * up(x) )`  — two parallel projections to `d_ff`, one is SiLU-activated and used as a gate.

#### `TransformerBlock`
`LayerNorm` → `CompressiveAttention` → residual → `LayerNorm` → `SwiGLU` → residual.

#### `CompressiveTransformer`
Full byte-level model:
- Embedding: standard `nn.Embedding` or `TDTEmbedding` (`use_tdt=True`)
- Sinusoidal `PositionalEncoding` (max 8192)
- Stack of `TransformerBlock` layers
- Final `LayerNorm` + linear projection to vocab logits
- Optional per-layer gradient checkpointing (`enable_gradient_checkpointing()`)

Default config: `vocab_size=256`, `d_model=512`, `n_layers=8`, `n_heads=8`, `d_ff=2048`, `window=512`, `compression_rate=4`.

---

### `optimized_transformer.py` — Production Model

All components from `compressive_transformer.py` are reused (imported directly). The optimised variant swaps or adds:

#### `RMSNorm`
Root-Mean-Square normalisation (no mean-centering). ~15 % faster than `LayerNorm` at the same quality.

#### `OptimisedCompressiveAttention`
Extends `CompressiveAttention` with:
- **Flash Attention 2** (`flash_attn_func`) for causal attention when CUDA is available; falls back to PyTorch `scaled_dot_product_attention` automatically.
- **KV cache**: concatenates previously seen `K`/`V` tensors for O(1)-per-step autoregressive inference. Returns `new_cache = {"k": K, "v": V}` each forward pass.
- **Infini-β** init changed to `−3.0` (sigmoid → ~0.047) so training starts almost entirely local.

#### `OptimisedBlock`
`RMSNorm` → `OptimisedCompressiveAttention` → residual → `RMSNorm` → `SwiGLU` → residual.  
Forward signature: `(x, comp_mem, kv_cache) → (x, new_comp, new_cache)`.

#### `OptimisedCompressiveTransformer`
Drop-in replacement for `CompressiveTransformer` with all optimised components.  
Extra method: `vram_estimate_gb(batch_size, seq_len)` — returns a dict with parameter, activation, optimizer-state, and total VRAM estimates in GB.

#### `CATWrapper`
Dynamic chunk scheduler wrapping any model.
- **Training**: randomly samples chunk size from `chunk_sizes=(128, 256, 512)` each forward pass, enabling multi-scale learning.
- **Inference**: defaults to the largest chunk size; override with `chunk_size=` argument.
- Handles sequences longer than the chunk size by iterating and accumulating `memories` and `kv_caches` across chunks (detached between chunks to limit graph size).
- Transparent proxy: delegates `parameters()`, `named_parameters()`, `state_dict()`, `load_state_dict()`, `enable_gradient_checkpointing()`, and `vram_estimate_gb()` to the inner model, so checkpoints are portable without the wrapper.

---

### `vortex/compression/arithmetic_coding.py`

Lossless arithmetic coding via `torchac`:

| Function | Description |
|---|---|
| `probs_to_cdf(probs)` | Converts model output probabilities to a cumulative CDF (with ε-smoothing) |
| `encode(probs, symbols)` | Encodes a `(B, T)` symbol tensor to `bytes` |
| `decode(bitstring, probs)` | Decodes `bytes` back to `(B, T)` int16 symbols |
| `theoretical_bpd(probs, symbols)` | Cross-entropy bits-per-byte — the training objective |

---

### `vortex/utils/zipnn.py` — Post-Training Weight Compression

Huffman-based lossless checkpoint size reduction (30–60 % smaller files).  
Splits each float32 weight tensor into sign + exponent + mantissa bytes. Exponents and signs are Huffman-coded (low entropy); raw mantissa bytes are stored unmodified (near-random, high entropy). Decompression is exact.

```python
from vortex.utils.zipnn import compress_model_weights, decompress_model_weights

compressed = compress_model_weights(model)
torch.save(compressed, "weights.zipnn.pt")

model2 = MyModel(...)
decompress_model_weights(model2, compressed)
```

---

## Hardware Configs

| File | GPU | VRAM | Params |
|------|-----|------|--------|
| `colab_t4.yaml`    | T4 (Colab)  | 15 GB  | 3.2 M  |
| `rtx4070_8gb.yaml` | RTX 4070    | 8 GB   | 8.5 M  |
| `default.yaml`     | RTX 3090/80 | 12 GB  | 14.8 M |
| `rtx4090_24gb.yaml`| RTX 4090    | 24 GB  | 28 M   |
| `amd_mi300x.yaml`  | MI300X      | 192 GB | 60 M+  |

---

## Training Details

The `scripts/train.py` loop uses `OptimisedCompressiveTransformer` wrapped in `CATWrapper`.  
Key features:
- **Mixed precision** (`torch.amp`) with `bfloat16` on ROCm/Ampere+, `float16` otherwise
- **Cosine LR schedule with linear warmup** (`vortex.utils.training.cosine_with_warmup`)
- **Gradient clipping** (`grad_clip=1.0`) + AdamW weight decay
- **EarlyStopping** on validation BPD (patience=5, min_delta=1e-4)
- **TensorBoard** logging + live ASCII scoreboard with BPD trend vs baselines
- **Gradient checkpointing** (enabled per config; ~40 % VRAM reduction)

Default hyperparameters (`configs/default.yaml`):

```
d_model: 512  |  n_layers: 8  |  n_heads: 8  |  d_ff: 2048
window: 512   |  compression_rate: 4          |  dropout: 0.1
batch_size: 32  |  lr: 3e-4  |  warmup: 4000  |  max_steps: 100000
```

---

## ATLAS Dataset

- **Source**: CERN EOS `root://eospublic.cern.ch//eos/opendata/atlas/datascience/ATLAS-FTAG-2023-05/`
- **Format**: HDF5 → extracted to raw binary (`atlas.bin`) via `download.py`
- **Benchmark sample**: `mc-flavtag-ttbar-medium.bin` (1 GB) — used for both baseline and Vortex evaluation
- **Structured dtype**: 30 fields including `pt_btagJes`, `GN2v01_pb`, kinematics, labels
- See `docs/ARCHITECTURE_COMPARISON.md` for a detailed v1 → v3 component diff and BPD benchmarks.
