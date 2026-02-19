#!/usr/bin/env python3
# Usage: python experiments/atlas_experiment/prepare.py
import os, sys
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

SPLITS = {
    "atlas_val.bin":  (0,          20 * 1024 * 1024),   
    "atlas_test.bin": (20*1024*1024, 40 * 1024 * 1024), 
}

def slice_bin(src: str, dst: str, start: int, end: int):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    size = end - start
    with open(src, "rb") as f:
        f.seek(start)
        data = f.read(size)
    with open(dst, "wb") as f:
        f.write(data)
    print(f"  {dst}  ({len(data)/1024/1024:.1f} MB)")

def main():
    src = os.path.join(DATA_DIR, "atlas.bin")
    if not os.path.exists(src):
        sys.exit(f"ERROR: {src} not found. Run download.py --all-steps first.")
    src_size = os.path.getsize(src)
    print(f"Source: {src}  ({src_size/1024/1024:.1f} MB)")
    for name, (start, end) in SPLITS.items():
        if end > src_size:
            print(f"  WARN: {name} — file too small, skipping")
            continue
        slice_bin(src, os.path.join(DATA_DIR, name), start, end)
    print("\nDone. Edit experiments/atlas_experiment/config.yaml if needed.")

if __name__ == "__main__":
    main()
