# Usage:
#   python experiments/camel_experiment/download.py --all-steps
#
#   Choose a different simulation set or limit simulations:
#   python experiments/camel_experiment/download.py --all-steps --sim-set CV
#   python experiments/camel_experiment/download.py --all-steps --sim-set LH --n-sims 50
#
#   Print MI300X training estimate without downloading:
#   python experiments/camel_experiment/download.py --info
#
# Data source:
#   CAMEL (Cosmology and Astrophysics with MachinE Learning Simulations)
#   IllustrisTNG suite, PartType0 gas particles, 11 float32 fields.
#   Website : https://www.camel-simulations.org/
#   FlatIron: https://users.flatironinstitute.org/~camels/

from __future__ import annotations
import argparse, json, os, shutil, sys
from typing import List

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required.  pip install h5py")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

COMBINED_BIN = os.path.join(DATA_DIR, "camel_combined.bin")
TRAIN_BIN    = os.path.join(DATA_DIR, "camel_train.bin")
VAL_BIN      = os.path.join(DATA_DIR, "camel_val.bin")
TEST_BIN     = os.path.join(DATA_DIR, "camel_test.bin")

TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10

# ---------------------------------------------------------------------------
# CAMEL data layout
#
# Each gas particle → 11 float32 scalars → 44 bytes/record:
#   Coordinates   : x, y, z           (3 floats)
#   Velocities    : vx, vy, vz        (3 floats)
#   Density       : ρ                 (1 float)
#   Masses        : m                 (1 float)
#   InternalEnergy: u                 (1 float)
#   ElectronAbund : x_e               (1 float)
#   Metallicity   : Z (total, 1st elem)(1 float)
#
# Total: 11 × 4 = 44 bytes per particle
# ---------------------------------------------------------------------------
FIELDS = [
    ("Coordinates",     3),   # → x, y, z
    ("Velocities",      3),   # → vx, vy, vz
    ("Density",         1),
    ("Masses",          1),
    ("InternalEnergy",  1),
    ("ElectronAbundance", 1),
    ("Metallicity",     1),   # scalar or take index 0 of element array
]
N_FIELDS     = sum(n for _, n in FIELDS)   # 11
RECORD_BYTES = N_FIELDS * 4                # 44

# ---------------------------------------------------------------------------
# Simulation set definitions
#
# Set    | # sims | purpose
# -------+--------+--------------------------------------------------
# CV     |    27  | Cosmo Variations — standard ±Ω_m/σ_8 benchmark  ← default
# 1P     |    61  | One-parameter-at-a-time variations
# LH     |  1000  | Latin Hypercube (full parameter-space coverage)
# ---------------------------------------------------------------------------
SIM_SETS = {
    "CV": 27,
    "1P": 61,
    "LH": 1000,
}

BASE_URL   = "https://users.flatironinstitute.org/~camels/Sims"
SUITE      = "IllustrisTNG"
SNAP_DEFAULT = 90   # z = 0 (present day) — CAMEL TNG uses 090, not 033


def snapshot_url(sim_set: str, sim_idx: int, snap: int) -> str:
    return (f"{BASE_URL}/{SUITE}/{sim_set}/{sim_set}_{sim_idx}"
            f"/snapshot_{snap:03d}.hdf5")


def local_h5(sim_set: str, sim_idx: int, snap: int) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR,
                        f"{SUITE}_{sim_set}_{sim_idx}_snap{snap:03d}.hdf5")


def local_bin(sim_set: str, sim_idx: int, snap: int) -> str:
    return local_h5(sim_set, sim_idx, snap).replace(".hdf5", ".bin")


def local_meta(sim_set: str, sim_idx: int, snap: int) -> str:
    return local_h5(sim_set, sim_idx, snap).replace(".hdf5", ".meta.json")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_snapshot(sim_set: str, sim_idx: int, snap: int) -> bool:
    """Download one HDF5 snapshot. Returns True if file is ready."""
    dst = local_h5(sim_set, sim_idx, snap)
    if os.path.exists(dst):
        print(f"[download] Already present: {os.path.basename(dst)}")
        return True

    url = snapshot_url(sim_set, sim_idx, snap)
    print(f"[download] {url}")
    print(f"        -> {dst}")

    aria2c = shutil.which("aria2c")
    if aria2c:
        import subprocess
        try:
            subprocess.check_call([
                aria2c, "-x16", "-s16", "-j16", "-k1M",
                "--check-certificate=false", "--summary-interval=10",
                "--allow-overwrite=true",
                "-o", os.path.basename(dst),
                "-d", os.path.dirname(dst),
                url,
            ])
            return True
        except subprocess.CalledProcessError as e:
            print(f"[download] aria2c failed ({e}), falling back to urllib")

    import ssl, urllib.request
    ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(url, context=ctx) as r, open(dst, "wb") as f:
            total      = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            chunk      = 1 << 20          # 1 MB
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total:
                    print(f"\r  {downloaded/1e6:.0f}/{total/1e6:.0f} MB "
                          f"({100*downloaded/total:.1f}%)", end="", flush=True)
        print()
        return True
    except Exception as exc:
        print(f"[download] FAILED: {exc}")
        if os.path.exists(dst):
            os.remove(dst)
        return False


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
def extract_snapshot(sim_set: str, sim_idx: int, snap: int) -> bool:
    """
    Extract PartType0 gas particle data from HDF5 → flat binary.

    Layout: N_particles × 11 float32 scalars, row-major (C order).
    Each row = [x, y, z, vx, vy, vz, rho, m, u, x_e, Z]
    """
    h5f  = local_h5(sim_set, sim_idx, snap)
    dst  = local_bin(sim_set, sim_idx, snap)
    meta = local_meta(sim_set, sim_idx, snap)

    if os.path.exists(dst):
        print(f"[extract] Already present: {os.path.basename(dst)}")
        return True

    if not os.path.exists(h5f):
        print(f"[extract] HDF5 not found: {h5f}  — skipping")
        return False

    print(f"[extract] {os.path.basename(h5f)} -> {os.path.basename(dst)}")

    with h5py.File(h5f, "r") as f:
        if "PartType0" not in f:
            print(f"[extract] WARN: no PartType0 in {h5f}, skipping")
            return False
        gas = f["PartType0"]

        parts: List[np.ndarray] = []

        for field, n_cols in FIELDS:
            if field not in gas:
                print(f"[extract] WARN: field '{field}' missing, filling zeros")
                # Determine n_particles from the first loaded part or a fallback
                n_ref = parts[0].shape[0] if parts else 0
                if n_ref == 0:
                    print(f"[extract] Cannot infer n_particles yet; skipping snapshot")
                    return False
                arr = np.zeros((n_ref, n_cols), dtype=np.float32)
            else:
                arr = gas[field][:]

            # Ensure 2-D (n_particles, n_cols)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            # For multi-element fields (e.g. per-element metallicity), take col 0
            if arr.shape[1] > n_cols:
                arr = arr[:, :n_cols]

            parts.append(arr.astype(np.float32, copy=False))

        data = np.concatenate(parts, axis=1)   # (N, 11)

    assert data.shape[1] == N_FIELDS, (
        f"Expected {N_FIELDS} fields, got {data.shape[1]}"
    )

    # Flatten to raw bytes
    data.tofile(dst)

    n_particles = data.shape[0]
    with open(meta, "w") as mf:
        json.dump({
            "sim_set": sim_set,
            "sim_idx": sim_idx,
            "snap":    snap,
            "n_particles": n_particles,
            "n_fields": N_FIELDS,
            "record_bytes": RECORD_BYTES,
            "field_names": [
                "x", "y", "z", "vx", "vy", "vz",
                "density", "mass", "internal_energy",
                "electron_abundance", "metallicity",
            ],
            "dtype": "float32",
            "order": "C",
        }, mf, indent=2)

    size_mb = os.path.getsize(dst) / 1e6
    print(f"[extract] Done: {n_particles:,} particles  "
          f"× {N_FIELDS} fields = {size_mb:.1f} MB")
    return True


# ---------------------------------------------------------------------------
# Combine (shuffle across simulations)
# ---------------------------------------------------------------------------
def combine_bins(sim_jobs: list) -> None:
    """Concatenate all per-snapshot binaries into one shuffled stream."""
    if os.path.exists(COMBINED_BIN):
        print(f"[combine] Already present: {COMBINED_BIN}")
        return

    parts: List[np.ndarray] = []
    total_particles = 0

    for sim_set, sim_idx, snap in sim_jobs:
        bp = local_bin(sim_set, sim_idx, snap)
        if not os.path.exists(bp):
            print(f"[combine] WARNING: {bp} missing, skipping")
            continue
        arr = np.fromfile(bp, dtype=np.float32).reshape(-1, N_FIELDS)
        parts.append(arr)
        total_particles += arr.shape[0]
        print(f"[combine] Loaded {os.path.basename(bp)}: "
              f"{arr.shape[0]:,} particles")

    if not parts:
        sys.exit("[combine] No binary files found — run --extract first")

    combined = np.concatenate(parts, axis=0)   # (N_total, 11)
    print(f"[combine] Total: {combined.shape[0]:,} particles  "
          f"({combined.nbytes / 1e9:.2f} GB)")

    print("[combine] Shuffling (rng seed=42)...")
    rng = np.random.default_rng(42)
    rng.shuffle(combined)

    print(f"[combine] Writing {COMBINED_BIN} ...")
    os.makedirs(DATA_DIR, exist_ok=True)
    combined.tofile(COMBINED_BIN)
    print(f"[combine] Done: {os.path.getsize(COMBINED_BIN)/1e9:.2f} GB")


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def make_splits() -> None:
    """
    Split combined binary into train / val / test at record-aligned boundaries.
    train=80%  val=10%  test=10%
    """
    if not os.path.exists(COMBINED_BIN):
        sys.exit(f"[split] {COMBINED_BIN} not found — run --combine first")

    total = os.path.getsize(COMBINED_BIN)
    assert total % RECORD_BYTES == 0, (
        f"[split] File size {total} is not a multiple of {RECORD_BYTES} B/record"
    )

    n_records  = total // RECORD_BYTES
    train_end  = int(n_records * TRAIN_FRAC)          * RECORD_BYTES
    val_end    = int(n_records * (TRAIN_FRAC + VAL_FRAC)) * RECORD_BYTES

    print(f"[split] {n_records:,} particles × {RECORD_BYTES} B  "
          f"(record-aligned)")
    print(f"[split] train={train_end/1e9:.2f} GB  "
          f"val={(val_end-train_end)/1e9:.2f} GB  "
          f"test={(total-val_end)/1e9:.2f} GB")

    _copy_slice(COMBINED_BIN, TRAIN_BIN, 0,         train_end)
    _copy_slice(COMBINED_BIN, VAL_BIN,   train_end, val_end)
    _copy_slice(COMBINED_BIN, TEST_BIN,  val_end,   total)

    print("\n[split] Final sizes:")
    for p in [TRAIN_BIN, VAL_BIN, TEST_BIN]:
        if os.path.exists(p):
            print(f"  {os.path.basename(p):28s}  "
                  f"{os.path.getsize(p)/1e9:6.2f} GB  "
                  f"({os.path.getsize(p)//RECORD_BYTES:,} particles)")


def _copy_slice(src: str, dst: str, start: int, end: int) -> None:
    if os.path.exists(dst):
        print(f"[split] Already present: {dst}")
        return
    size = end - start
    print(f"[split] Writing {os.path.basename(dst)} ({size/1e9:.2f} GB) ...")
    buf_size = 1 << 20
    remaining = size
    with open(src, "rb") as sf, open(dst, "wb") as df:
        sf.seek(start)
        while remaining > 0:
            buf = sf.read(min(buf_size, remaining))
            if not buf:
                break
            df.write(buf)
            remaining -= len(buf)


# ---------------------------------------------------------------------------
# MI300X estimate
# ---------------------------------------------------------------------------
def _mi300x_estimate(n_jobs: int, snap: int) -> None:
    """Rough VRAM + training time estimate for CAMEL on MI300X."""
    # Typical gas particle count per CAMEL IllustrisTNG snapshot (varies ~5-10M)
    PARTICLES_PER_SIM = 3_500_000    # conservative; actual depends on z
    total_particles   = n_jobs * PARTICLES_PER_SIM
    binary_gb         = total_particles * RECORD_BYTES / 1e9
    train_gb          = binary_gb * TRAIN_FRAC

    # 25M param model, batch 128, seq 512, bf16 → ~49 steps/sec (1/4 GPU)
    steps_sec          = 49
    max_steps          = 500_000
    tokens_per_step    = 128 * 512
    steps_per_epoch    = max(1, int(train_gb * 1e9) // tokens_per_step)
    time_500k_h        = max_steps / steps_sec / 3600
    time_1epoch_h      = steps_per_epoch / steps_sec / 3600
    epochs_in_budget   = max_steps / steps_per_epoch

    # VRAM: 25M params × (4+2+8)B + activations
    param_vram_gb      = 25e6 * 14 / 1e9
    act_vram_gb        = 128 * 512 * 512 * 8 * 2 / 1e9
    total_vram_gb      = param_vram_gb + act_vram_gb

    lines = [
        "",
        "═" * 58,
        "  MI300X Training Estimate — CAMEL IllustrisTNG",
        "═" * 58,
        f"  Simulation set : {n_jobs} simulations, snap {snap:03d} (z=0)",
        f"  ~Particles     : {total_particles:,}",
        f"  Binary size    : ~{binary_gb:.1f} GB total  "
          f"/ {train_gb:.1f} GB train",
        "─" * 58,
        f"  VRAM (camel job): ~{total_vram_gb:.1f} GB  /  192 GB   ✓",
        "─" * 58,
        f"  Throughput     : ~{steps_sec} steps/sec  (25M params, 1/4 GPU)",
        f"  500k steps     : ~{time_500k_h:.1f} h",
        f"  1 full epoch   : ~{time_1epoch_h:.1f} h  ({steps_per_epoch:,} steps)",
        f"  Epochs in 500k : {epochs_in_budget:.1f}×",
        "─" * 58,
        "  Run alongside atlas, era5, ligo with:",
        "    ./scripts/launch_parallel.sh",
        "═" * 58,
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="CAMEL IllustrisTNG data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default: CV set (27 sims, snap z=0)\n"
            "  python download.py --all-steps\n\n"
            "  # LH set, first 50 sims:\n"
            "  python download.py --all-steps --sim-set LH --n-sims 50\n\n"
            "  # Estimate only:\n"
            "  python download.py --info --sim-set CV\n"
        ),
    )
    p.add_argument("--sim-set", default="CV", choices=list(SIM_SETS),
                   help="CAMEL simulation set to use (default: CV = 27 sims)")
    p.add_argument("--n-sims", type=int, default=None,
                   help="Limit number of simulations to download "
                        "(default: all in the chosen set)")
    p.add_argument("--snap", type=int, default=SNAP_DEFAULT,
                   help=f"Snapshot number (default: {SNAP_DEFAULT} = z=0)")
    p.add_argument("--download",  action="store_true")
    p.add_argument("--extract",   action="store_true")
    p.add_argument("--combine",   action="store_true")
    p.add_argument("--split",     action="store_true")
    p.add_argument("--info",      action="store_true",
                   help="Print MI300X estimate and exit")
    p.add_argument("--all-steps", action="store_true",
                   help="download + extract + combine + split")
    args = p.parse_args(argv)

    if args.all_steps:
        args.download = args.extract = args.combine = args.split = True

    max_sims  = SIM_SETS[args.sim_set]
    n_sims    = min(args.n_sims or max_sims, max_sims)
    sim_jobs  = [(args.sim_set, i, args.snap) for i in range(n_sims)]

    if args.info:
        _mi300x_estimate(n_sims, args.snap)
        if not args.all_steps:
            return 0

    print(f"[config] Suite={SUITE}  Set={args.sim_set}  "
          f"Sims=0..{n_sims-1}  Snap={args.snap:03d}")
    print(f"[config] Output dir: {DATA_DIR}")
    print(f"[config] Record size: {RECORD_BYTES} B/particle  "
          f"({N_FIELDS} float32 fields)")

    if args.download:
        failed = 0
        for sim_set, sim_idx, snap in sim_jobs:
            ok = download_snapshot(sim_set, sim_idx, snap)
            if not ok:
                failed += 1
        if failed:
            print(f"[download] {failed}/{n_sims} snapshots failed. "
                  "The rest will be processed.")

    if args.extract:
        failed = 0
        for sim_set, sim_idx, snap in sim_jobs:
            if os.path.exists(local_h5(sim_set, sim_idx, snap)):
                ok = extract_snapshot(sim_set, sim_idx, snap)
                if not ok:
                    failed += 1
            else:
                print(f"[extract] Skipping {sim_set}_{sim_idx} "
                      "(HDF5 not downloaded)")
        if failed:
            print(f"[extract] {failed} snapshots skipped/failed.")

    if args.combine:
        # Only combine jobs whose .bin files actually exist
        available = [j for j in sim_jobs
                     if os.path.exists(local_bin(*j))]
        if not available:
            sys.exit("[combine] No .bin files found — run --extract first")
        print(f"[combine] Combining {len(available)}/{n_sims} available binaries")
        combine_bins(available)

    if args.split:
        make_splits()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
