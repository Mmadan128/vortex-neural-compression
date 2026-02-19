# Usage: python experiments/atlas_experiment/download.py --all-steps

from __future__ import annotations
import subprocess
import argparse
import json
import math
import os
import shutil
import sys
import urllib.request
from typing import Iterable

import numpy as np

try:
    import h5py
except Exception:
    print("h5py is required. pip install h5py", file=sys.stderr)
    raise

# ── paths ────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(HERE, "data")

EOS_SRC   = (
    "root://eospublic.cern.ch//eos/opendata/atlas/datascience/"
    "ATLAS-FTAG-2023-05/mc-flavtag-ttbar-small.h5"
)

H5_PATH   = os.path.join(DATA_DIR, "atlas.h5")
BIN_PATH  = os.path.join(DATA_DIR, "atlas.bin")
META_PATH = os.path.join(DATA_DIR, "atlas.meta.json")
NPZ_PATH  = os.path.join(DATA_DIR, "atlas.npz")
M200_PATH = os.path.join(DATA_DIR, "atlas_200m.bin")
RECON_H5  = os.path.join(DATA_DIR, "atlas_reconstructed.h5")

# ── helpers ───────────────────────────────────────────────────────────────────

def root_to_https(url: str) -> str:
    if url.startswith("root://eospublic.cern.ch//"):
        return "https://eospublic.cern.ch/" + url.split("//", 2)[-1]
    return url


def download_atlas_h5(src: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"[download] Already present: {dst}")
        return
    
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    # 1. Try aria2c (Fastest) - Bypassing Certs
    aria2c = shutil.which("aria2c")
    if aria2c:
        https_url = root_to_https(src)
        print(f"[download] aria2c (insecure) {https_url} -> {dst}")
        try:
            subprocess.check_call([
                aria2c, "-x16", "-s16", "-j16", "-k1M", 
                "--check-certificate=false",  # <--- Bypass SSL check
                "--summary-interval=5", "--allow-overwrite=true",
                "-o", os.path.basename(dst), "-d", os.path.dirname(dst),
                https_url
            ])
            return
        except subprocess.CalledProcessError as e:
            print(f"[download] aria2c failed: {e}")

    # 2. Try xrdcp (CERN Native) - Bypassing Certs
    xrdcp = shutil.which("xrdcp")
    if src.startswith("root://") and xrdcp:
        print(f"[download] xrdcp (insecure) {src} -> {dst}")
        try:
            # env variable or flag depending on version; -insecure is common
            subprocess.check_call([xrdcp, "-f", "--insecure", src, dst])
            return
        except subprocess.CalledProcessError:
            print("[download] xrdcp failed, trying HTTPS fallback...")

    # 3. Final Fallback: Standard HTTPS - Bypassing Certs
    https_url = root_to_https(src)
    print(f"[download] HTTPS fallback (insecure) {https_url} -> {dst}")
    
    # Create an unverified context for urllib
    import ssl
    context = ssl._create_unverified_context()
    
    with urllib.request.urlopen(https_url, context=context) as r, open(dst, "wb") as f:
        total      = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        chunk      = 1 << 20
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total:
                print(f"\r[download] {downloaded}/{total} ({100*downloaded/total:.1f}%)", end="")
    print()


def iter_slices(n_rows: int, chunk_rows: int) -> Iterable[slice]:
    for start in range(0, n_rows, chunk_rows):
        yield slice(start, min(n_rows, start + chunk_rows))


def save_bin(h5_path: str = H5_PATH, bin_path: str = BIN_PATH,
             meta_path: str = META_PATH) -> None:
    print(f"[bin] Extracting 'jets' -> {bin_path}")
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    with h5py.File(h5_path, "r") as h5:
        if "jets" not in h5:
            raise KeyError("Dataset 'jets' not found in HDF5 file")
        dset       = h5["jets"]
        shape      = tuple(dset.shape)
        dtype      = dset.dtype
        n_rows     = shape[0] if dset.ndim >= 1 else 1
        chunk_rows = max(1, 1 << 12)
        with open(bin_path, "wb") as f:
            for sl in iter_slices(n_rows, chunk_rows):
                f.write(np.ascontiguousarray(dset[sl]).tobytes(order="C"))
    with open(meta_path, "w") as m:
        json.dump({"shape": list(shape), "dtype": dtype.str,
                   "dtype_descr": dtype.descr, "order": "C"}, m, indent=2)
    size_mb = os.path.getsize(bin_path) / 1024 / 1024
    print(f"[bin] Done — {bin_path} ({size_mb:.1f} MB)")


def save_npz(bin_path: str = BIN_PATH, meta_path: str = META_PATH,
             npz_path: str = NPZ_PATH) -> None:
    with open(meta_path) as m:
        meta = json.load(m)
    dtype = np.dtype([(tuple(f) if isinstance(f, list) else f) for f in meta["dtype_descr"]])
    arr   = np.fromfile(bin_path, dtype=dtype)
    print(f"[npz] Writing {npz_path} (compressed)")
    np.savez_compressed(npz_path, jets=arr)


def save_200m(bin_path: str = BIN_PATH, out_path: str = M200_PATH,
              limit_bytes: int = 200 * 1024 * 1024) -> None:
    print(f"[200m] Slicing first {limit_bytes//1024//1024} MB -> {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bufsize   = 1 << 20
    remaining = limit_bytes
    with open(bin_path, "rb") as src, open(out_path, "wb") as dst:
        while remaining > 0:
            buf = src.read(min(bufsize, remaining))
            if not buf:
                break
            dst.write(buf)
            remaining -= len(buf)
    print(f"[200m] Done — {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB)")


def reconstruct_h5(bin_path: str = BIN_PATH, meta_path: str = META_PATH,
                   out_h5: str = RECON_H5) -> None:
    with open(meta_path) as m:
        meta = json.load(m)
    dtype = np.dtype([(tuple(f) if isinstance(f, list) else f) for f in meta["dtype_descr"]])
    shape = tuple(meta["shape"])
    arr   = np.fromfile(bin_path, dtype=dtype, count=int(np.prod(shape))).reshape(shape)
    print(f"[reconstruct] Writing {out_h5}")
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("jets", data=arr, chunks=True, compression="gzip", compression_opts=4)


def compare_h5_jets(a_path: str = H5_PATH, b_path: str = RECON_H5) -> bool:
    with h5py.File(a_path, "r") as fa, h5py.File(b_path, "r") as fb:
        da, db = fa["jets"], fb["jets"]
        if da.shape != db.shape or da.dtype != db.dtype:
            print(f"[compare] Shape/dtype mismatch: {da.shape} {da.dtype} vs {db.shape} {db.dtype}")
            return False
        n_rows     = da.shape[0]
        chunk_rows = 1 << 12
        total      = math.ceil(n_rows / chunk_rows)
        for i, sl in enumerate(iter_slices(n_rows, chunk_rows), 1):
            a, b = da[sl], db[sl]
            if a.dtype.fields:
                for name in a.dtype.names:
                    av, bv = a[name], b[name]
                    ok = (np.allclose(av, bv, equal_nan=True)
                          if np.dtype(a.dtype[name]).kind == "f"
                          else np.array_equal(av, bv))
                    if not ok:
                        print(f"[compare] Field '{name}' mismatch in rows {sl.start}:{sl.stop}")
                        return False
            elif not np.allclose(a, b, equal_nan=True):
                print(f"[compare] Mismatch in rows {sl.start}:{sl.stop}")
                return False
            if i % 50 == 0 or i == total:
                print(f"[compare] {i}/{total} chunks OK")
    print("[compare] All chunks equal — lossless round-trip verified")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="ATLAS jets round-trip helper")
    p.add_argument("--src",       default=EOS_SRC)
    p.add_argument("--download",  action="store_true", help="Download HDF5 from CERN EOS")
    p.add_argument("--extract",   action="store_true", help="Extract jets -> .bin + .meta.json + 200m slice")
    p.add_argument("--npz",       action="store_true", help="Also save compressed .npz")
    p.add_argument("--reconstruct", action="store_true", help="Reconstruct HDF5 from .bin")
    p.add_argument("--compare",   action="store_true", help="Compare original vs reconstructed HDF5")
    p.add_argument("--all-steps", action="store_true", help="Run download + extract + reconstruct + compare")
    args = p.parse_args(argv)

    if args.all_steps:
        args.download = args.extract = args.reconstruct = args.compare = True

    if args.download:
        download_atlas_h5(args.src, H5_PATH)
    if args.extract:
        save_bin()
        save_200m()
        if args.npz:
            save_npz()
    if args.reconstruct:
        reconstruct_h5()
    ok = True
    if args.compare:
        ok = compare_h5_jets()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
