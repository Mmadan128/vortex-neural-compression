from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import sys
import urllib.request

import numpy as np

try:
    import h5py
except Exception:
    print("h5py is required. Install with: pip install h5py", file=sys.stderr)
    raise

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

DEFAULT_SRC = (
    "https://users.flatironinstitute.org/~camels/Sims/"
    "IllustrisTNG/CV/CV_0/snapshot_024.hdf5"
)

H5_FILE = os.path.join(DATA_DIR, "camel_snapshot_024.hdf5")
RAW_BIN = os.path.join(DATA_DIR, "camel.bin")
FLOAT32_BIN = os.path.join(DATA_DIR, "camel_float32.bin")
FLOAT32_META = os.path.join(DATA_DIR, "camel_float32.meta.json")

TRAIN_BIN = os.path.join(DATA_DIR, "camel_train.bin")
VAL_BIN = os.path.join(DATA_DIR, "camel_val.bin")
TEST_BIN = os.path.join(DATA_DIR, "camel_test.bin")

RAW_TRAIN_BIN = os.path.join(DATA_DIR, "camel_raw_train.bin")
RAW_VAL_BIN = os.path.join(DATA_DIR, "camel_raw_val.bin")
RAW_TEST_BIN = os.path.join(DATA_DIR, "camel_raw_test.bin")

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10

LOCAL_DEFAULT_MAX_ROWS = 200_000
LOCAL_DEFAULT_MAX_H5_MB = 500


def download_file(src: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"[download] Already present: {dst}")
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[download] {src} -> {dst}")

    aria2c = shutil.which("aria2c")
    if aria2c:
        try:
            import subprocess

            subprocess.check_call(
                [
                    aria2c,
                    "-x16",
                    "-s16",
                    "-j16",
                    "-k1M",
                    "--check-certificate=false",
                    "--summary-interval=10",
                    "--allow-overwrite=true",
                    "-o",
                    os.path.basename(dst),
                    "-d",
                    os.path.dirname(dst),
                    src,
                ]
            )
            return
        except Exception as exc:
            print(f"[download] aria2c failed: {exc}; falling back to urllib")

    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(src, context=ctx) as response, open(dst, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 20
        while True:
            buf = response.read(chunk)
            if not buf:
                break
            out.write(buf)
            downloaded += len(buf)
            if total:
                pct = 100.0 * downloaded / total
                print(
                    f"\r  {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
    print()


def _estimate_parttype0_row_bytes(h5: h5py.File) -> int:
    if "PartType0" not in h5:
        raise KeyError("PartType0 not found in CAMEL snapshot")
    gas = h5["PartType0"]
    bytes_per_row = 0
    for key in ["Coordinates", "Velocities", "Density", "Masses", "InternalEnergy", "ElectronAbundance", "Metallicity"]:
        if key not in gas:
            continue
        dset = gas[key]
        itemsize = int(dset.dtype.itemsize)
        width = int(np.prod(dset.shape[1:])) if dset.ndim > 1 else 1
        bytes_per_row += itemsize * width
    if bytes_per_row <= 0:
        raise RuntimeError("Unable to estimate PartType0 row width")
    return bytes_per_row


def cap_h5_size(h5_path: str, target_mb: int) -> None:
    if target_mb <= 0:
        return
    if not os.path.exists(h5_path):
        sys.exit(f"[cap-h5] Source not found: {h5_path}")

    current_mb = os.path.getsize(h5_path) / (1024 * 1024)
    if current_mb <= target_mb:
        print(f"[cap-h5] Already <= {target_mb} MB: {h5_path} ({current_mb:.1f} MB)")
        return

    print(f"[cap-h5] Reducing {h5_path} from {current_mb:.1f} MB to <= {target_mb} MB")
    tmp_path = h5_path + ".tmp"
    with h5py.File(h5_path, "r") as src:
        if "PartType0" not in src:
            raise KeyError("PartType0 not found; cannot build compact local snapshot")

        p0 = src["PartType0"]
        n_rows = int(next(iter(p0.values())).shape[0])
        row_bytes = _estimate_parttype0_row_bytes(src)
        max_rows = max(1, int((target_mb * 1024 * 1024) / row_bytes))
        keep_rows = min(n_rows, max_rows)

        with h5py.File(tmp_path, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            g = dst.create_group("PartType0")
            for key in ["Coordinates", "Velocities", "Density", "Masses", "InternalEnergy", "ElectronAbundance", "Metallicity"]:
                if key not in p0:
                    continue
                arr = p0[key][:keep_rows]
                g.create_dataset(key, data=arr, compression="gzip", compression_opts=4, shuffle=True)

    os.replace(tmp_path, h5_path)
    final_mb = os.path.getsize(h5_path) / (1024 * 1024)
    print(f"[cap-h5] Done: {h5_path} ({final_mb:.1f} MB, rows={keep_rows:,})")


def extract_raw(h5_path: str, raw_bin: str) -> None:
    if os.path.exists(raw_bin):
        print(f"[extract-raw] Already present: {raw_bin}")
        return

    print(f"[extract-raw] {h5_path} -> {raw_bin}")
    os.makedirs(os.path.dirname(raw_bin), exist_ok=True)
    with open(h5_path, "rb") as src, open(raw_bin, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"[extract-raw] Done: {raw_bin} ({os.path.getsize(raw_bin) / 1e6:.0f} MB)")


def _load_parttype0_fields(h5_path: str) -> tuple[np.ndarray, list[str]]:
    with h5py.File(h5_path, "r") as h5:
        if "PartType0" not in h5:
            raise KeyError("PartType0 not found in CAMEL snapshot")

        gas = h5["PartType0"]
        cols = []
        names = []

        field_map = [
            ("Coordinates", ["x", "y", "z"]),
            ("Velocities", ["vx", "vy", "vz"]),
            ("Density", ["density"]),
            ("Masses", ["mass"]),
            ("InternalEnergy", ["internal_energy"]),
            ("ElectronAbundance", ["electron_abundance"]),
            ("Metallicity", ["metallicity"]),
        ]

        for key, feature_names in field_map:
            if key not in gas:
                continue

            arr = gas[key][:]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif key == "Metallicity" and arr.ndim == 2 and arr.shape[1] > 1:
                arr = arr[:, :1]

            arr = np.asarray(arr, dtype=np.float32)
            cols.append(arr)
            names.extend(feature_names[: arr.shape[1]])

        if not cols:
            raise RuntimeError("No supported PartType0 fields found in snapshot")

        n_rows = cols[0].shape[0]
        for col in cols:
            if col.shape[0] != n_rows:
                raise RuntimeError("Field row counts are inconsistent")

        data = np.concatenate(cols, axis=1).astype(np.float32, copy=False)
        return data, names


def extract_float32(h5_path: str, out_bin: str, out_meta: str, max_rows: int | None) -> None:
    if os.path.exists(out_bin) and os.path.exists(out_meta):
        print(f"[extract-float32] Already present: {out_bin}")
        return

    print(f"[extract-float32] {h5_path} -> {out_bin}")
    data, names = _load_parttype0_fields(h5_path)

    if max_rows is not None and max_rows > 0 and max_rows < data.shape[0]:
        rng = np.random.default_rng(42)
        idx = rng.choice(data.shape[0], size=max_rows, replace=False)
        data = data[idx]
        print(f"[extract-float32] Subsampled to {max_rows:,} rows")

    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    data.tofile(out_bin)

    meta = {
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "dtype": "float32",
        "order": "C",
        "feature_names": names,
        "source": os.path.basename(h5_path),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[extract-float32] Done: {out_bin} ({os.path.getsize(out_bin) / 1e6:.0f} MB)")


def split_file(src: str, train_dst: str, val_dst: str, test_dst: str, record_bytes: int = 1) -> None:
    if not os.path.exists(src):
        sys.exit(f"[split] Source not found: {src}")

    total = os.path.getsize(src)
    n_records = total // record_bytes

    train_end = int(n_records * TRAIN_FRAC) * record_bytes
    val_end = int(n_records * (TRAIN_FRAC + VAL_FRAC)) * record_bytes

    splits = {
        train_dst: (0, train_end),
        val_dst: (train_end, val_end),
        test_dst: (val_end, total),
    }

    for dst, (start, end) in splits.items():
        if os.path.exists(dst):
            print(f"[split] Already present: {dst}")
            continue

        size = end - start
        print(f"[split] {dst} ({size / 1e9:.3f} GB)")
        with open(src, "rb") as inp, open(dst, "wb") as out:
            inp.seek(start)
            rem = size
            chunk = 1 << 20
            while rem > 0:
                buf = inp.read(min(chunk, rem))
                if not buf:
                    break
                out.write(buf)
                rem -= len(buf)


def _float32_record_bytes(meta_file: str) -> int:
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    shape = meta.get("shape", [0, 0])
    if len(shape) != 2 or shape[1] <= 0:
        raise ValueError(f"Invalid shape in {meta_file}: {shape}")
    return int(shape[1]) * 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CAMEL raw + float32 binary datasets")
    parser.add_argument(
        "--profile",
        choices=["local", "server"],
        default="server",
        help=(
            "Execution profile. local: smaller float32 subset for laptop/desktop tests; "
            "server: full-size pipeline defaults"
        ),
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help="CAMEL snapshot URL")
    parser.add_argument(
        "--max-h5-mb",
        type=int,
        default=None,
        help="Cap downloaded HDF5 size by rebuilding a smaller local snapshot (PartType0 subset)",
    )
    parser.add_argument("--download", action="store_true", help="Download CAMEL HDF5 snapshot")
    parser.add_argument("--extract-raw", action="store_true", help="Create camel.bin as raw HDF5 bytes")
    parser.add_argument(
        "--extract-float32",
        action="store_true",
        help="Create camel_float32.bin + camel_float32.meta.json from PartType0 fields",
    )
    parser.add_argument("--split-raw", action="store_true", help="Split camel.bin into train/val/test")
    parser.add_argument(
        "--split-float32",
        action="store_true",
        help="Split camel_float32.bin into camel_train/val/test at row boundaries",
    )
    parser.add_argument("--all-steps", action="store_true", help="Run download + extract + split for both")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional random row cap for float32 extraction (for quick tests)",
    )
    args = parser.parse_args()

    if args.all_steps:
        args.download = True
        if args.profile == "local":
            # Local profile keeps the pipeline lightweight by default.
            args.extract_raw = False
            args.split_raw = False
            args.extract_float32 = True
            args.split_float32 = True
            if args.max_rows is None:
                args.max_rows = LOCAL_DEFAULT_MAX_ROWS
        else:
            args.extract_raw = True
            args.extract_float32 = True
            args.split_raw = True
            args.split_float32 = True

    if args.profile == "local" and args.max_rows is None and args.extract_float32:
        args.max_rows = LOCAL_DEFAULT_MAX_ROWS
        print(f"[profile] local -> defaulting --max-rows to {LOCAL_DEFAULT_MAX_ROWS:,}")

    if args.profile == "local" and args.max_h5_mb is None:
        args.max_h5_mb = LOCAL_DEFAULT_MAX_H5_MB
        print(f"[profile] local -> defaulting --max-h5-mb to {LOCAL_DEFAULT_MAX_H5_MB}")

    if args.profile == "local" and args.download:
        print(
            "[profile] local -> reduced prepared dataset via --max-rows and float32-only path; "
            "for smaller network download pass a smaller snapshot URL with --src"
        )

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.download:
        download_file(args.src, H5_FILE)
        if args.max_h5_mb is not None:
            cap_h5_size(H5_FILE, args.max_h5_mb)

    if args.extract_raw:
        extract_raw(H5_FILE, RAW_BIN)

    if args.extract_float32:
        extract_float32(H5_FILE, FLOAT32_BIN, FLOAT32_META, args.max_rows)

    if args.split_raw:
        split_file(RAW_BIN, RAW_TRAIN_BIN, RAW_VAL_BIN, RAW_TEST_BIN, record_bytes=1)

    if args.split_float32:
        record_bytes = _float32_record_bytes(FLOAT32_META)
        split_file(FLOAT32_BIN, TRAIN_BIN, VAL_BIN, TEST_BIN, record_bytes=record_bytes)

    print("\n[done] CAMEL data preparation complete")


if __name__ == "__main__":
    main()
