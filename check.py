#!/usr/bin/env python3
# Usage: python check.py --config configs/amd_mi300x.yaml
import sys
import os
import time
import argparse
import math

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import psutil
    PSUTIL = True
except ImportError:
    PSUTIL = False

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

def section(title):
    print(f"\n{'─'*54}")
    print(f"  {title}")
    print(f"{'─'*54}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/amd_mi300x.yaml")
    args = p.parse_args()

    ok = True

    section("1 / 6  Config")
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print(f"{PASS} Loaded: {args.config}")
    except Exception as e:
        print(f"{FAIL} Cannot load config: {e}")
        sys.exit(1)

    section("2 / 6  Data files")
    paths = cfg.get("paths", {})
    eval_cfg = cfg.get("evaluation", {})
    files_to_check = {
        "train_data": paths.get("train_data"),
        "val_data":   paths.get("val_data"),
        "test_data":  eval_cfg.get("test_data"),
    }
    for label, path in files_to_check.items():
        if not path:
            print(f"{WARN} {label}: not set in config")
            continue
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / 1e9
            print(f"{PASS} {label}: {path}  ({size_gb:.2f} GB)")
        else:
            print(f"{FAIL} {label}: MISSING — {path}")
            ok = False

    section("3 / 6  Memory")
    if PSUTIL:
        ram_gb = psutil.virtual_memory().total / 1e9
        avail_gb = psutil.virtual_memory().available / 1e9
        print(f"{PASS} Total RAM : {ram_gb:.0f} GB")
        print(f"{PASS} Available : {avail_gb:.0f} GB")
        train_path = paths.get("train_data")
        if train_path and os.path.exists(train_path):
            train_gb = os.path.getsize(train_path) / 1e9
            if not cfg.get("data", {}).get("streaming", False):
                print(f"{FAIL} streaming=false with {train_gb:.1f} GB file — will OOM! "
                      f"Set streaming: true")
                ok = False
            else:
                print(f"{PASS} streaming=true — memmap will be used, no RAM load")
    else:
        print(f"{WARN} psutil not installed — skipping RAM check  (pip install psutil)")

    section("4 / 6  ROCm + PyTorch")
    print(f"{PASS} PyTorch version  : {torch.__version__}")
    if torch.version.hip:
        print(f"{PASS} ROCm HIP version : {torch.version.hip}")
    else:
        print(f"{WARN} Not a ROCm build — torch.version.hip is None")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{PASS} GPU              : {name}")
        print(f"{PASS} VRAM             : {vram:.0f} GB")
        if "MI300X" not in name and "MI300" not in name:
            print(f"{WARN} Expected MI300X — got '{name}'")
    else:
        print(f"{FAIL} torch.cuda.is_available() = False")
        ok = False

    section("5 / 6  Dependencies")
    deps = ["torch", "numpy", "h5py", "yaml", "tqdm",
            "tensorboard", "psutil"]
    for dep in deps:
        try:
            __import__(dep)
            print(f"{PASS} {dep}")
        except ImportError:
            print(f"{FAIL} {dep} — pip install {dep}")
            ok = False

    section("6 / 6  Dataset smoke test")
    train_path = paths.get("train_data")
    if train_path and os.path.exists(train_path):
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from vortex.data.dataset import MemmapWindowDataset
            window = cfg["data"]["window_size"]
            stride = cfg["data"]["stride"]
            t0 = time.time()
            ds = MemmapWindowDataset(train_path, window, stride)
            elapsed = time.time() - t0
            print(f"{PASS} MemmapWindowDataset init  : {elapsed:.3f}s")
            print(f"{PASS} Windows in train set      : {len(ds):,}")
            # Read 10 random windows
            t0 = time.time()
            dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
            batch = next(iter(dl))
            elapsed = time.time() - t0
            print(f"{PASS} First batch shape         : {list(batch.shape)}")
            print(f"{PASS} First batch load time     : {elapsed:.3f}s")
            print(f"{PASS} dtype / range             : {batch.dtype}  "
                  f"[{batch.min().item()}, {batch.max().item()}]")
        except ImportError as e:
            print(f"{WARN} Could not import MemmapWindowDataset: {e}")
        except Exception as e:
            print(f"{FAIL} Dataset smoke test failed: {e}")
            ok = False
    else:
        print(f"{WARN} Skipping — train file not present")

    print(f"\n{'═'*54}")
    if ok:
        print(f"  ALL CHECKS PASSED — safe to launch training")
        print(f"  python scripts/train.py --config {args.config}")
    else:
        print(f"  SOME CHECKS FAILED — fix above before launching")
    print(f"{'═'*54}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())