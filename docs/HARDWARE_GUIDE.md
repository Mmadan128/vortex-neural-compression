# Hardware Guide

| Config            | GPU         | VRAM   | Params | Batch | ~Time (200 MB ATLAS) |
|-------------------|-------------|--------|--------|-------|----------------------|
| `colab_t4`        | T4          | 15 GB  | 3.2 M  | 32    | 3 h                  |
| `rtx4070_8gb`     | RTX 4070    | 8 GB   | 8.5 M  | 16    | 10 h                 |
| `default`         | RTX 3090/80 | 12 GB  | 14.8 M | 32    | 6 h                  |
| `rtx4090_24gb`    | RTX 4090    | 24 GB  | 28 M   | 64    | 5 h                  |
| `amd_mi300x`      | MI300X      | 192 GB | 60 M+  | 128   | 2 h                  |

## Tips
- OOM? Halve `batch_size`, increase `warmup_steps` proportionally.
- Flash Attention 2 requires CUDA + Linux. macOS falls back to `F.scaled_dot_product_attention`.

