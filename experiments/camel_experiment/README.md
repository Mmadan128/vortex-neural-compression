# CAMEL Experiment

**CAMEL** — Cosmology and Astrophysics with MachinE Learning Simulations  
Website: https://www.camel-simulations.org/

## What is CAMEL?

CAMEL is a suite of cosmological hydrodynamical simulations covering a grid of cosmological and astrophysical parameters.  
Each simulation produces snapshots of gas, dark matter and stellar particles.  
We use **PartType0 (gas)** particles from the **IllustrisTNG** simulation physics at **z = 0** (snapshot 033).

## Data we extract

11 float32 fields per gas particle → **44 bytes/record**:

| Index | Field             | Description                  |
|-------|-------------------|------------------------------|
| 0–2   | x, y, z           | Comoving coordinates (Mpc/h) |
| 3–5   | vx, vy, vz        | Peculiar velocities (km/s)   |
| 6     | density           | Gas density (M☉/Mpc³)       |
| 7     | mass              | Particle mass (M☉/h)         |
| 8     | internal_energy   | Specific thermal energy      |
| 9     | electron_abundance | xe = n_e / n_H              |
| 10    | metallicity       | Total metal mass fraction    |

## Recommended dataset size

| Set | # Sims | ~Size | Notes |
|-----|-------|-------|-------|
| **CV** (default) | **27** | **~10 GB** | Systematically varies Ω_m and σ_8 — standard benchmark |
| 1P | 61 | ~22 GB | One-parameter-at-a-time variations |
| LH (50 sims) | 50 | ~18 GB | Latin Hypercube partial — broader parameter coverage |
| LH (full) | 1000 | ~350 GB | Full parameter space; only if you need maximum diversity |

**For a research paper: CV set (27 sims, ~10 GB) is the right choice.**  
It covers the cosmological parameter subspace that all CAMEL papers benchmark on.

## Quick start

```bash
# Estimate only (no download)
python experiments/camel_experiment/download.py --info

# Full pipeline (CV set, z=0, ~10 GB)
python experiments/camel_experiment/download.py --all-steps

# Larger: LH set first 50 sims
python experiments/camel_experiment/download.py --all-steps --sim-set LH --n-sims 50
```

## Train

```bash
python scripts/train.py --config configs/camel_mi300x.yaml
```

Or via the parallel launcher (runs alongside atlas, era5, ligo on the same MI300X):

```bash
./scripts/launch_parallel.sh atlas camel
```

## Directory layout

```
camel_experiment/
  download.py          ← this pipeline
  data/
    camel_combined.bin   ← shuffled, all sims merged
    camel_train.bin      ← 80% of combined (record-aligned)
    camel_val.bin        ← 10%
    camel_test.bin       ← 10%
    IllustrisTNG_CV_0_snap033.hdf5   ← raw HDF5 (can delete after extract)
    IllustrisTNG_CV_0_snap033.bin    ← per-sim binary
    IllustrisTNG_CV_0_snap033.meta.json
    ...
  checkpoints/
  runs/
```
