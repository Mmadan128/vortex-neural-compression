# Changelog

## v0.3.0 (2026-02)
- **Real ATLAS data**: removed all synthetic data generation
- Experiment structure: everything under `experiments/atlas_experiment/`
- `download.py` downloads CERN EOS HDF5 and extracts binary slices
- `prepare.py` creates val/test splits from full atlas.bin
- `config.yaml` per experiment, paths point into experiment data dir

## v0.2.0 (2026-02)
- Flash Attention 2 integration (2-4x faster, auto-fallback)
- KV cache for ~10x faster autoregressive decompression
- SwiGLU feed-forward (LLaMA/PaLM style)
- 5 hardware-specific config files
- .vxc file format with arithmetic coding

## v0.1.0 (2026-01)
- Initial compressive transformer (Rae et al. 2019)
- Arithmetic coding via torchac
- Training pipeline with TensorBoard + early stopping
