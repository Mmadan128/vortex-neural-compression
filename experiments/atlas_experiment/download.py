# Usage: python experiments/atlas_experiment/download.py --all-steps
from __future__ import annotations
import subprocess, argparse, json, math, os, shutil, ssl, sys, urllib.request
from typing import Iterable
import numpy as np

try:
    import h5py
except Exception:
    print("h5py is required. pip install h5py", file=sys.stderr)
    raise

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(HERE, "data")

SOURCES = {
    "mc-flavtag-ttbar-medium.h5": (
        "root://eospublic.cern.ch//eos/opendata/atlas/datascience/"
        "ATLAS-FTAG-2023-05/mc-flavtag-ttbar-medium.h5"
    ),
}

def h5_path(name):   return os.path.join(DATA_DIR, name)
def bin_path(name):  return os.path.join(DATA_DIR, name.replace(".h5", ".bin"))
def meta_path(name): return os.path.join(DATA_DIR, name.replace(".h5", ".meta.json"))

COMBINED_BIN  = os.path.join(DATA_DIR, "atlas_combined.bin")
TRAIN_BIN     = os.path.join(DATA_DIR, "atlas_train.bin")
VAL_BIN       = os.path.join(DATA_DIR, "atlas_val.bin")
TEST_BIN      = os.path.join(DATA_DIR, "atlas_test.bin")

TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10


def root_to_https(url: str) -> str:
    if url.startswith("root://eospublic.cern.ch//"):
        return "https://eospublic.cern.ch/" + url.split("//", 2)[-1]
    return url


def download_file(src: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"[download] Already present: {dst}  ({os.path.getsize(dst)/1e6:.0f} MB)")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    aria2c = shutil.which("aria2c")
    if aria2c:
        https_url = root_to_https(src)
        print(f"[download] aria2c -> {dst}")
        try:
            subprocess.check_call([
                aria2c, "-x16", "-s16", "-j16", "-k1M",
                "--check-certificate=false", "--summary-interval=10",
                "--allow-overwrite=true",
                "-o", os.path.basename(dst), "-d", os.path.dirname(dst),
                https_url
            ])
            return
        except subprocess.CalledProcessError as e:
            print(f"[download] aria2c failed: {e}")

    xrdcp = shutil.which("xrdcp")
    if src.startswith("root://") and xrdcp:
        print(f"[download] xrdcp -> {dst}")
        try:
            subprocess.check_call([xrdcp, "-f", "--insecure", src, dst])
            return
        except subprocess.CalledProcessError:
            print("[download] xrdcp failed, trying HTTPS...")

    https_url = root_to_https(src)
    print(f"[download] HTTPS -> {dst}")
    ctx = ssl._create_unverified_context()
    print(f"[download] HTTPS -> {dst}")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(https_url, context=ctx) as r, open(dst, "wb") as f:
        total, downloaded, chunk = int(r.headers.get("Content-Length", 0)), 0, 1 << 20
        while True:
            buf = r.read(chunk)
            if not buf: break
            f.write(buf); downloaded += len(buf)
            if total:
                print(f"\r  {downloaded/1e6:.0f}/{total/1e6:.0f} MB "
                      f"({100*downloaded/total:.1f}%)", end="", flush=True)
    print()


def iter_slices(n: int, chunk: int):
    for s in range(0, n, chunk):
        yield slice(s, min(n, s + chunk))


def extract_bin(h5_file: str) -> None:
    """Extract 'jets' dataset from HDF5 to raw binary + meta JSON."""
    dst = bin_path(os.path.basename(h5_file))
    meta = meta_path(os.path.basename(h5_file))
    if os.path.exists(dst):
        print(f"[extract] Already present: {dst}")
        return
    print(f"[extract] {h5_file} -> {dst}")
    with h5py.File(h5_file, "r") as h5:
        if "jets" not in h5:
            raise KeyError(f"'jets' not found in {h5_file}")
        dset = h5["jets"]
        shape, dtype = tuple(dset.shape), dset.dtype
        n_rows = shape[0]
        with open(dst, "wb") as f:
            for sl in iter_slices(n_rows, 1 << 12):
                f.write(np.ascontiguousarray(dset[sl]).tobytes(order="C"))
    with open(meta, "w") as m:
        json.dump({"shape": list(shape), "dtype": dtype.str,
                   "dtype_descr": dtype.descr, "order": "C",
                   "source": os.path.basename(h5_file)}, m, indent=2)
    print(f"[extract] Done: {dst}  ({os.path.getsize(dst)/1e6:.0f} MB)")


def combine_bins() -> None:
    """Concatenate ttbar + zprime binaries into one shuffled stream."""
    if os.path.exists(COMBINED_BIN):
        print(f"[combine] Already present: {COMBINED_BIN}")
        return

    parts = []
    for name in SOURCES:
        bp = bin_path(name)
        if not os.path.exists(bp):
            print(f"[combine] WARNING: {bp} missing, skipping")
            continue
        mp = meta_path(name)
        with open(mp) as f:
            meta = json.load(f)
        dtype = np.dtype([(tuple(x) if isinstance(x, list) else x)
                          for x in meta["dtype_descr"]])
        arr = np.fromfile(bp, dtype=dtype)
        parts.append(arr)
        print(f"[combine] Loaded {name}: {len(arr):,} jets")

    if not parts:
        sys.exit("[combine] No binary files found — run --extract first")

    combined = np.concatenate(parts, axis=0)
    print(f"[combine] Total: {len(combined):,} jets  "
          f"({combined.nbytes/1e6:.0f} MB)")

    print("[combine] Shuffling...")
    rng = np.random.default_rng(42)
    rng.shuffle(combined)

    print(f"[combine] Writing {COMBINED_BIN}")
    os.makedirs(DATA_DIR, exist_ok=True)
    combined.tofile(COMBINED_BIN)
    print(f"[combine] Done: {COMBINED_BIN}  "
          f"({os.path.getsize(COMBINED_BIN)/1e6:.0f} MB)")


def make_splits() -> None:
    """
    Split combined binary into train / val / test at byte level.
    Splits are made AFTER jet-level shuffle in combine_bins(), so
    there is NO data leakage between splits.

    Sizes (from ~13 GB medium file):
        train : ~10.4 GB  (80%)
        val   :  ~1.3 GB  (10%)
        test  :  ~1.3 GB  (10%)
    """
    if not os.path.exists(COMBINED_BIN):
        sys.exit(f"[split] {COMBINED_BIN} not found — run --combine first")

    total = os.path.getsize(COMBINED_BIN)
    train_end = int(total * TRAIN_FRAC)
    val_end   = int(total * (TRAIN_FRAC + VAL_FRAC))

    splits = {
        TRAIN_BIN: (0,         train_end),
        VAL_BIN:   (train_end, val_end),
        TEST_BIN:  (val_end,   total),
    }

    for dst, (start, end) in splits.items():
        if os.path.exists(dst):
            print(f"[split] Already present: {dst}")
            continue
        size = end - start
        print(f"[split] {dst}  ({size/1e9:.2f} GB)")
        bufsize, rem = 1 << 20, size
        with open(COMBINED_BIN, "rb") as src, open(dst, "wb") as out:
            src.seek(start)
            while rem > 0:
                buf = src.read(min(bufsize, rem))
                if not buf: break
                out.write(buf); rem -= len(buf)

    print("\n[split] Final sizes:")
    for p in [TRAIN_BIN, VAL_BIN, TEST_BIN]:
        if os.path.exists(p):
            print(f"  {os.path.basename(p):30s} {os.path.getsize(p)/1e9:6.2f} GB")


def reconstruct_h5(bin_file: str, meta_file: str, out_h5: str) -> None:
    with open(meta_file) as m:
        meta = json.load(m)
    dtype = np.dtype([(tuple(f) if isinstance(f, list) else f)
                      for f in meta["dtype_descr"]])
    shape = tuple(meta["shape"])
    arr   = np.fromfile(bin_file, dtype=dtype,
                        count=int(np.prod(shape))).reshape(shape)
    print(f"[reconstruct] Writing {out_h5}")
    os.makedirs(DATA_DIR, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("jets", data=arr, chunks=True,
                         compression="gzip", compression_opts=4)


def compare_h5(a_path: str, b_path: str) -> bool:
    with h5py.File(a_path, "r") as fa, h5py.File(b_path, "r") as fb:
        da, db = fa["jets"], fb["jets"]
        if da.shape != db.shape or da.dtype != db.dtype:
            print(f"[compare] Shape/dtype mismatch")
            return False
        for i, sl in enumerate(iter_slices(da.shape[0], 1 << 12), 1):
            a, b = da[sl], db[sl]
            if a.dtype.fields:
                for name in a.dtype.names:
                    av, bv = a[name], b[name]
                    ok = (np.allclose(av, bv, equal_nan=True)
                          if np.dtype(a.dtype[name]).kind == "f"
                          else np.array_equal(av, bv))
                    if not ok:
                        print(f"[compare] Mismatch in field '{name}'")
                        return False
            elif not np.allclose(a, b, equal_nan=True):
                return False
            if i % 50 == 0: print(f"[compare] {i} chunks OK")
    print("[compare] All equal — lossless round-trip verified")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="ATLAS FTAG data pipeline")
    p.add_argument("--download",    action="store_true", help="Download both HDF5 files")
    p.add_argument("--extract",     action="store_true", help="Extract jets -> .bin")
    p.add_argument("--combine",     action="store_true", help="Shuffle & concatenate bins")
    p.add_argument("--split",       action="store_true", help="Create train/val/test splits")
    p.add_argument("--reconstruct", action="store_true", help="Reconstruct HDF5 from bin")
    p.add_argument("--compare",     action="store_true", help="Verify round-trip")
    p.add_argument("--all-steps",   action="store_true",
                   help="Run download + extract + combine + split")
    args = p.parse_args(argv)

    if args.all_steps:
        args.download = args.extract = args.combine = args.split = True

    sources = SOURCES

    if args.download:
        for name, src in sources.items():
            download_file(src, h5_path(name))

    if args.extract:
        for name in sources:
            hp = h5_path(name)
            if os.path.exists(hp):
                extract_bin(hp)
            else:
                print(f"[extract] {hp} not found — run --download first")

    if args.combine:
        bp = bin_path("mc-flavtag-ttbar-medium.h5")
        if not os.path.exists(COMBINED_BIN):
            if not os.path.exists(bp):
                print(f"[combine] {bp} not found — run --extract first")
            else:
                shutil.copy2(bp, COMBINED_BIN)
                print(f"[combine] copied {bp} -> {COMBINED_BIN}")

    if args.split:
        make_splits()

    if args.reconstruct:
        for name in sources:
            bp = bin_path(name)
            mp = meta_path(name)
            out = h5_path(name.replace(".h5", "_reconstructed.h5"))
            if os.path.exists(bp) and os.path.exists(mp):
                reconstruct_h5(bp, mp, out)

    if args.compare:
        for name in sources:
            orig = h5_path(name)
            recon = h5_path(name.replace(".h5", "_reconstructed.h5"))
            if os.path.exists(orig) and os.path.exists(recon):
                compare_h5(orig, recon)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())