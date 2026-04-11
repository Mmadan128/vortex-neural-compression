#!/usr/bin/env python3
"""
Prepare ALICE ROOT data for byte-level Vortex training.

Pipeline:
1) Download (or reuse) a ROOT file
2) Select numeric / jagged-numeric branches from a TTree
3) Encode to padded float32 matrix and write .bin + .meta.json
4) Split by events into train / val / test (80/10/10)

Usage examples:
  # Use a local ROOT file
  python experiments/alice_experiment/download.py \
      --input-root /path/to/alice.root

  # Download from URL first
  python experiments/alice_experiment/download.py \
      --url https://example.org/alice.root
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import uproot


DEFAULT_ALICE_RECORD = "1105"
SMALL_TEST_EVENTS = 2048
SMALL_TEST_MAX_LIST_LEN = 8


@dataclass
class BranchMeta:
    name: str
    is_list: bool
    max_len: int
    dtype: str = "float32"
    col_offset: int = 0


@dataclass
class BinMeta:
    n_events: int
    tree_key: str
    branches: List[BranchMeta]
    lengths: Dict[str, List[int]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_events": self.n_events,
                "tree_key": self.tree_key,
                "branches": [asdict(b) for b in self.branches],
                "lengths": self.lengths,
            },
            indent=2,
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_valid_root_file(path: Path) -> bool:
    """Return True if ROOT file can be opened and contains at least one key."""
    try:
        with uproot.open(path) as f:
            # Force key read; catches many truncated/corrupt ROOT files.
            _ = list(f.keys())
        return True
    except Exception:
        return False


def root_to_https(url: str) -> str:
    prefix = "root://eospublic.cern.ch/"
    if not url.startswith(prefix):
        return url
    path = url[len(prefix):]
    if not path.startswith("/"):
        path = "/" + path
    return "https://eospublic.cern.ch" + path


def resolve_alice_uri_from_record(record_id: str, prefer_small: bool = False) -> str:
    api_url = f"https://opendata.cern.ch/api/records/{record_id}"
    with urllib.request.urlopen(api_url, timeout=60) as resp:
        payload = json.load(resp)

    metadata = payload.get("metadata", {})

    def _first_uri(entries: object) -> Optional[str]:
        if not isinstance(entries, list):
            return None

        root_candidates: List[Tuple[int, str]] = []
        any_candidates: List[Tuple[int, str]] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri")
            if not isinstance(uri, str) or not uri:
                continue
            size = item.get("size")
            if not isinstance(size, int):
                size = 0
            any_candidates.append((size, uri))
            if uri.lower().endswith(".root"):
                root_candidates.append((size, uri))

        candidates = root_candidates if root_candidates else any_candidates
        if not candidates:
            return None

        if prefer_small:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return candidates[0][1]

    def _from_file_indices(indices: object) -> Optional[str]:
        if not isinstance(indices, list):
            return None

        all_entries: List[dict] = []
        for idx in indices:
            if not isinstance(idx, dict):
                continue
            files = idx.get("files")
            if isinstance(files, list):
                all_entries.extend([x for x in files if isinstance(x, dict)])

        return _first_uri(all_entries)

    files = metadata.get("files") or []
    uri = _first_uri(files)
    if uri:
        return uri

    # Some ALICE records expose downloadable objects via _file_indices.
    uri = _from_file_indices(metadata.get("_file_indices") or [])
    if uri:
        return uri

    raise RuntimeError(
        f"No downloadable file URI found in OpenData record {record_id} "
        "(checked metadata.files and metadata._file_indices)"
    )


def list_record_root_files(record_id: str) -> List[dict]:
    """Return root-file entries with uri/size from an ALICE OpenData record."""
    api_url = f"https://opendata.cern.ch/api/records/{record_id}"
    with urllib.request.urlopen(api_url, timeout=60) as resp:
        payload = json.load(resp)

    metadata = payload.get("metadata", {})
    out: List[dict] = []

    files = metadata.get("files") or []
    for f in files:
        if not isinstance(f, dict):
            continue
        uri = f.get("uri")
        if isinstance(uri, str) and uri.lower().endswith(".root"):
            out.append({"uri": uri, "size": int(f.get("size", 0))})

    indices = metadata.get("_file_indices") or []
    for idx in indices:
        if not isinstance(idx, dict):
            continue
        sub = idx.get("files")
        if not isinstance(sub, list):
            continue
        for f in sub:
            if not isinstance(f, dict):
                continue
            uri = f.get("uri")
            if isinstance(uri, str) and uri.lower().endswith(".root"):
                out.append({"uri": uri, "size": int(f.get("size", 0))})

    # Deduplicate by URI while preserving first occurrence.
    uniq: Dict[str, dict] = {}
    for item in out:
        uniq.setdefault(item["uri"], item)
    return list(uniq.values())


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        if is_valid_root_file(dest):
            print(f"[download] Reusing existing file: {dest}")
            return
        print(f"[download] Existing file is invalid, re-downloading: {dest}")
        dest.unlink(missing_ok=True)
        part = Path(str(dest) + ".aria2")
        part.unlink(missing_ok=True)

    ensure_dir(dest.parent)
    https_url = root_to_https(url)

    aria2c = shutil.which("aria2c")
    if aria2c:
        cmd = [
            aria2c,
            "-x16", "-s16", "-j16", "-k1M",
            "--check-certificate=false",
            "--allow-overwrite=true",
            "-o", dest.name,
            "-d", str(dest.parent),
            https_url,
        ]
        try:
            subprocess.check_call(cmd)
            if is_valid_root_file(dest):
                return
            print("[download] aria2c produced an invalid ROOT file, retrying via fallback")
            dest.unlink(missing_ok=True)
            Path(str(dest) + ".aria2").unlink(missing_ok=True)
        except subprocess.CalledProcessError:
            pass

    if url.startswith("root://"):
        xrdcp = shutil.which("xrdcp")
        if xrdcp:
            try:
                subprocess.check_call([xrdcp, "-f", "--insecure", url, str(dest)])
                if is_valid_root_file(dest):
                    return
                print("[download] xrdcp produced an invalid ROOT file, retrying via HTTPS stream")
                dest.unlink(missing_ok=True)
            except subprocess.CalledProcessError:
                pass

    def _stream_http(context: Optional[ssl.SSLContext] = None) -> None:
        req = urllib.request.Request(https_url, headers={"User-Agent": "vortex-codec/alice"})
        with urllib.request.urlopen(req, timeout=120, context=context) as resp, open(dest, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            got = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if total > 0:
                    print(f"\r[download] {got / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
        if total > 0:
            print()

    try:
        _stream_http()
    except urllib.error.URLError as e:
        # Some server images (containers/proxies) expose self-signed chains.
        if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            print("[download] SSL verify failed, retrying without certificate verification")
            ctx = ssl._create_unverified_context()
            _stream_http(context=ctx)
        else:
            raise

    if not is_valid_root_file(dest):
        raise RuntimeError(f"Downloaded file is not a valid ROOT file: {dest}")


def open_tree(root_path: Path, preferred_tree: Optional[str]) -> Tuple[str, uproot.behaviors.TBranch.TTree]:
    f = uproot.open(root_path)
    if preferred_tree:
        if preferred_tree not in f:
            raise KeyError(f"Tree '{preferred_tree}' not found in {root_path}")
        tree = f[preferred_tree]
        if not isinstance(tree, uproot.behaviors.TBranch.TTree):
            raise TypeError(f"Object '{preferred_tree}' is not a TTree")
        return preferred_tree, tree

    preferred = ["Events", "events", "aodTree", "O2events", "tree"]
    classes = f.classnames()
    for name in preferred:
        for key, cls in classes.items():
            if cls == "TTree" and key.split(";")[0] == name:
                return key, f[key]

    for key, cls in classes.items():
        if cls == "TTree":
            return key, f[key]

    raise RuntimeError(f"No TTree found in {root_path}")


def load_numeric_arrays(tree: uproot.behaviors.TBranch.TTree, n_entries: int) -> Dict[str, ak.Array]:
    """Read only branches that uproot can interpret and that are numeric-like."""
    loaded: Dict[str, ak.Array] = {}
    branch_names = []
    for key in tree.keys():
        name = key.split(";")[0]
        if name not in loaded:
            branch_names.append(name)

    for name in branch_names:
        try:
            arr = tree[name].array(entry_stop=n_entries, library="ak")
            flat = ak.to_numpy(ak.ravel(arr))
            # Keep only plain numeric arrays; drop structured/object-like types.
            if getattr(flat.dtype, "fields", None):
                continue
            if flat.dtype.kind not in "biuf":
                continue
            loaded[name] = arr
        except Exception:
            # Expected for custom classes or unsupported streamer layouts.
            continue

    return loaded


def encode_to_bin(
    arrs: ak.Array,
    selected: List[str],
    max_list_len: int,
    show_skip_examples: bool = True,
) -> Tuple[np.ndarray, BinMeta]:
    n_events = len(arrs)
    cols: List[np.ndarray] = []
    branches: List[BranchMeta] = []
    lengths: Dict[str, List[int]] = {}
    skip_counts = {
        "nested_list": 0,
        "non_convertible": 0,
        "non_event_aligned": 0,
    }
    skip_examples: List[str] = []

    def note_skip(kind: str, msg: str) -> None:
        skip_counts[kind] += 1
        if len(skip_examples) < 20:
            skip_examples.append(msg)

    col_offset = 0
    for name in selected:
        a = arrs[name]
        is_list = isinstance(getattr(ak.type(a), "content", None), ak.types.ListType)

        try:
            if is_list:
                per_lengths = ak.to_numpy(ak.num(a, axis=-1)).astype(np.int32)
                if per_lengths.ndim != 1:
                    note_skip("nested_list", f"nested-list: {name}")
                    continue
                inferred = int(per_lengths.max()) if per_lengths.size > 0 else 0
                # Use fixed width for large multi-file corpus compatibility.
                width = max_list_len
                a_pad = ak.pad_none(a, width, axis=1, clip=True)
                a_fill = ak.fill_none(a_pad, 0)
                np_arr = ak.to_numpy(a_fill).astype(np.float32)
                lengths[name] = per_lengths.astype(int).tolist()
            else:
                width = 1
                np_arr = ak.to_numpy(a).astype(np.float32).reshape(-1, 1)
                lengths[name] = []
        except Exception:
            note_skip("non_convertible", f"non-convertible: {name}")
            continue

        # Normalize to 2D: (events, features)
        if np_arr.ndim == 1:
            np_arr = np_arr.reshape(-1, 1)
        elif np_arr.ndim > 2:
            np_arr = np_arr.reshape(np_arr.shape[0], -1)

        if np_arr.shape[0] != n_events:
            if n_events > 0 and np_arr.shape[0] % n_events == 0:
                # Recover fixed-size arrays flattened by uproot (e.g. 219*5 -> 1095 rows).
                np_arr = np_arr.reshape(n_events, -1)
            else:
                note_skip(
                    "non_event_aligned",
                    f"non-event-aligned: {name} (rows={np_arr.shape[0]}, expected={n_events})",
                )
                continue

        cols.append(np_arr)
        branches.append(
            BranchMeta(
                name=name,
                is_list=is_list,
                max_len=width,
                col_offset=col_offset,
            )
        )
        col_offset += np_arr.shape[1]

    if not cols:
        raise RuntimeError("No numeric branches were selected from the ROOT tree")

    kept = len(cols)
    skipped = sum(skip_counts.values())
    print(
        "[encode] summary: "
        f"kept={kept}, skipped={skipped} "
        f"(non-convertible={skip_counts['non_convertible']}, "
        f"nested-list={skip_counts['nested_list']}, "
        f"non-event-aligned={skip_counts['non_event_aligned']})"
    )
    if show_skip_examples and skip_examples:
        print("[encode] first skipped examples:")
        for msg in skip_examples:
            print(f"  - {msg}")

    data = np.concatenate(cols, axis=1).astype(np.float32)
    meta = BinMeta(
        n_events=int(n_events),
        tree_key="Events",
        branches=branches,
        lengths=lengths,
    )
    return data, meta


def write_event_splits(bin_path: Path, n_events: int, total_cols: int, force: bool = False) -> None:
    bytes_per_event = total_cols * 4
    sizes = [int(n_events * 0.8), int(n_events * 0.1)]
    sizes.append(n_events - sum(sizes))
    offsets = [0, sizes[0], sizes[0] + sizes[1], n_events]

    base, ext = os.path.splitext(str(bin_path))
    names = ["train", "val", "test"]

    for name, start_ev, end_ev in zip(names, offsets[:-1], offsets[1:]):
        out_path = Path(f"{base}_{name}{ext}")
        if out_path.exists() and not force:
            print(f"[split] Reusing {out_path.name}")
            continue

        start_byte = start_ev * bytes_per_event
        count = (end_ev - start_ev) * bytes_per_event
        with open(bin_path, "rb") as fin, open(out_path, "wb") as fout:
            fin.seek(start_byte)
            remaining = count
            while remaining > 0:
                chunk = fin.read(min(4 * 1024 * 1024, remaining))
                if not chunk:
                    break
                fout.write(chunk)
                remaining -= len(chunk)

        gb = out_path.stat().st_size / 1e9
        print(f"[split] {out_path.name}: {end_ev - start_ev:,} events, {gb:.2f} GB")


def write_byte_splits(src_path: Path, force: bool = False) -> None:
    """Split a raw byte stream into 80/10/10 train/val/test files."""
    total = src_path.stat().st_size
    sizes = [int(total * 0.8), int(total * 0.1)]
    sizes.append(total - sum(sizes))
    offsets = [0, sizes[0], sizes[0] + sizes[1], total]

    base, ext = os.path.splitext(str(src_path))
    names = ["train", "val", "test"]

    for name, start, end in zip(names, offsets[:-1], offsets[1:]):
        out = Path(f"{base}_{name}{ext}")
        if out.exists() and not force:
            print(f"[split] Reusing {out.name}")
            continue
        with open(src_path, "rb") as fin, open(out, "wb") as fout:
            fin.seek(start)
            remaining = end - start
            while remaining > 0:
                chunk = fin.read(min(4 * 1024 * 1024, remaining))
                if not chunk:
                    break
                fout.write(chunk)
                remaining -= len(chunk)
        print(f"[split] {out.name}: {(end - start) / 1e9:.2f} GB")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare ALICE ROOT data for Vortex training")
    parser.add_argument("--all-steps", action="store_true", help="Run full download+prepare pipeline")
    parser.add_argument(
        "--small-test",
        action="store_true",
        help=(
            "Local smoke-test mode: use a small number of events and compact list width "
            "for fast validation."
        ),
    )
    parser.add_argument("--url", default=None, help="HTTP(S) URL to ALICE ROOT file")
    parser.add_argument(
        "--record-id",
        default=DEFAULT_ALICE_RECORD,
        help=f"CERN OpenData ALICE record id used for auto-download (default: {DEFAULT_ALICE_RECORD})",
    )
    parser.add_argument("--input-root", default=None, help="Use an existing local ROOT file")
    parser.add_argument(
        "--target-gb",
        type=float,
        default=0.0,
        help=(
            "If >0, download multiple ROOT files from record until this many GB "
            "and build a large corpus"
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of ROOT files when using --target-gb",
    )
    parser.add_argument(
        "--large-format",
        choices=["float32", "raw"],
        default="float32",
        help="Output format for --target-gb mode: float32 feature corpus (default) or raw ROOT bytes",
    )
    parser.add_argument("--tree", default=None, help="Tree key to read (default: auto-detect)")
    parser.add_argument("--out-dir", default=str(here / "data"), help="Output directory")
    parser.add_argument("--root-name", default="alice.root", help="Local ROOT filename in out-dir")
    parser.add_argument("--bin-out", default="alice_events.bin", help="Output .bin filename")
    parser.add_argument("--meta-out", default="alice_events.meta.json", help="Output metadata filename")
    parser.add_argument("--nmax", type=int, default=1_000_000, help="Max events to load")
    parser.add_argument(
        "--max-list-len",
        type=int,
        default=32,
        help="Cap jagged/list branches to this many elements per event",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite bin/meta/splits if present")
    return parser.parse_args()


def apply_small_test_defaults(args: argparse.Namespace) -> None:
    """Adjust runtime args for quick local end-to-end validation."""
    if not args.small_test:
        return

    if args.nmax == 1_000_000:
        args.nmax = SMALL_TEST_EVENTS
    if args.max_list_len == 32:
        args.max_list_len = SMALL_TEST_MAX_LIST_LEN

    if args.bin_out == "alice_events.bin":
        args.bin_out = "alice_small.bin"
    if args.meta_out == "alice_events.meta.json":
        args.meta_out = "alice_small.meta.json"

    print(
        "[small-test] enabled: "
        f"nmax={args.nmax}, max_list_len={args.max_list_len}, "
        f"bin={args.bin_out}"
    )


def main() -> None:
    args = parse_args()
    apply_small_test_defaults(args)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.target_gb > 0:
        if args.input_root or args.url:
            raise SystemExit("--target-gb mode uses record files; do not pass --input-root/--url")

        candidates = list_record_root_files(args.record_id)
        if not candidates:
            raise RuntimeError(f"No ROOT files found for record {args.record_id}")

        # For large corpora, consider largest files first.
        candidates.sort(key=lambda x: x.get("size", 0), reverse=True)
        target_bytes = int(args.target_gb * 1e9)
        candidate_pool = candidates[: args.max_files] if args.max_files > 0 else candidates
        pool_raw_bytes = sum(int(x.get("size", 0)) for x in candidate_pool)

        picked: List[dict] = []
        total = 0
        if args.large_format == "raw":
            for item in candidate_pool:
                picked.append(item)
                total += int(item.get("size", 0))
                if total >= target_bytes:
                    break

        if args.bin_out == "alice_events.bin":
            args.bin_out = "alice_large.bin" if args.large_format == "float32" else "alice_raw.bin"
        if args.meta_out == "alice_events.meta.json":
            args.meta_out = "alice_large.meta.json" if args.large_format == "float32" else "alice_raw.meta.json"

        raw_dir = out_dir / "raw_roots"
        ensure_dir(raw_dir)

        bin_path = out_dir / args.bin_out
        meta_path = out_dir / args.meta_out
        if bin_path.exists() and args.force:
            bin_path.unlink()
        elif bin_path.exists() and not args.force:
            raise SystemExit(f"Output exists: {bin_path} (use --force)")

        if args.large_format == "raw":
            print(
                f"[large-raw] Download plan: {len(picked)} files, "
                f"~{total / 1e9:.2f} GB target={args.target_gb:.2f} GB"
            )

            downloaded_paths: List[Path] = []
            for i, item in enumerate(picked, start=1):
                uri = item["uri"]
                name = f"alice_{i:04d}.root"
                dest = raw_dir / name
                print(f"[large-raw] ({i}/{len(picked)}) {name}  size={item.get('size', 0) / 1e9:.3f} GB")
                download_file(uri, dest)
                downloaded_paths.append(dest)

            with open(bin_path, "wb") as fout:
                for p in downloaded_paths:
                    with open(p, "rb") as fin:
                        shutil.copyfileobj(fin, fout, length=4 * 1024 * 1024)

            meta_payload = {
                "mode": "raw_root_bytes",
                "record_id": args.record_id,
                "target_gb": args.target_gb,
                "total_bytes": bin_path.stat().st_size,
                "n_files": len(downloaded_paths),
                "files": [{"path": str(p), "size": p.stat().st_size} for p in downloaded_paths],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=2)

            print(f"[large] Wrote {bin_path} ({bin_path.stat().st_size / 1e9:.2f} GB)")
            print(f"[large] Wrote {meta_path}")
            write_byte_splits(bin_path, force=args.force)
            return

        # float32 large corpus mode: encode each file and append rows.
        print(
            f"[large-f32] Plan: up to {len(candidate_pool)} files "
            f"(~{pool_raw_bytes / 1e9:.2f} GB raw available), "
            f"target={args.target_gb:.2f} GB encoded"
        )

        target_bytes = int(args.target_gb * 1e9)
        total_bytes = 0
        total_events = 0
        ref_sig = None
        merged_meta: Optional[BinMeta] = None
        used_files: List[dict] = []

        with open(bin_path, "wb") as fout:
            for i, item in enumerate(candidate_pool, start=1):
                uri = item["uri"]
                p = raw_dir / f"alice_{i:04d}.root"
                print(
                    f"[large-f32] ({i}/{len(candidate_pool)}) {p.name} "
                    f"size={item.get('size', 0) / 1e9:.3f} GB"
                )
                download_file(uri, p)

                tree_key, tree = open_tree(p, args.tree)
                n_entries = min(args.nmax, tree.num_entries)
                numeric_arrays = load_numeric_arrays(tree, n_entries)
                selected = list(numeric_arrays.keys())
                if not selected:
                    print(f"[large-f32] Skipping file with no readable numeric branches: {p.name}")
                    continue

                arrs = ak.zip(numeric_arrays, depth_limit=1)
                data, meta = encode_to_bin(
                    arrs,
                    selected,
                    args.max_list_len,
                    show_skip_examples=(len(used_files) == 0),
                )
                sig = [(b.name, b.is_list, b.max_len) for b in meta.branches]

                if ref_sig is None:
                    ref_sig = sig
                    merged_meta = meta
                    merged_meta.lengths = {k: list(v) for k, v in meta.lengths.items()}
                    merged_meta.n_events = 0
                    merged_meta.tree_key = tree_key
                elif sig != ref_sig:
                    print(f"[large-f32] Skipping schema-mismatch file: {p.name}")
                    continue

                data.astype(np.float32).tofile(fout)
                total_bytes += int(data.nbytes)
                total_events += int(data.shape[0])
                used_files.append({"path": str(p), "events": int(data.shape[0]), "bytes": int(data.nbytes)})

                # Merge per-event list lengths.
                assert merged_meta is not None
                for key in merged_meta.lengths:
                    merged_meta.lengths[key].extend(meta.lengths.get(key, []))

                print(
                    f"[large-f32] appended {p.name}: events={data.shape[0]}, "
                    f"bytes={data.nbytes / 1e9:.3f} GB, total={total_bytes / 1e9:.3f} GB"
                )
                if total_bytes >= target_bytes:
                    break

        if total_bytes < target_bytes:
            print(
                f"[large-f32] warning: encoded target not reached "
                f"({total_bytes / 1e9:.2f}/{args.target_gb:.2f} GB). "
                "Increase --max-files (or remove it), use a higher-yield record, "
                "or use --large-format raw for exact raw-byte size targets."
            )

        if merged_meta is None or ref_sig is None or total_events == 0:
            raise RuntimeError("Failed to build float32 corpus: no compatible files were encoded")

        merged_meta.n_events = total_events
        with open(meta_path, "w", encoding="utf-8") as f:
            payload = json.loads(merged_meta.to_json())
            payload["mode"] = "float32_multi_root"
            payload["record_id"] = args.record_id
            payload["target_gb"] = args.target_gb
            payload["total_bytes"] = total_bytes
            payload["n_files"] = len(used_files)
            payload["files"] = used_files
            json.dump(payload, f, indent=2)

        print(f"[large-f32] Wrote {bin_path} ({total_bytes / 1e9:.2f} GB)")
        print(f"[large-f32] Wrote {meta_path}")
        total_cols = sum(b.max_len for b in merged_meta.branches)
        write_event_splits(bin_path, merged_meta.n_events, total_cols, force=args.force)
        return

    if not args.url and not args.input_root:
        # Atlas-style default: auto-resolve one ALICE file from OpenData and process end-to-end.
        args.url = resolve_alice_uri_from_record(args.record_id, prefer_small=args.small_test)
        print(f"[prep] Auto-selected ALICE source from record {args.record_id}")

    root_path = Path(args.input_root) if args.input_root else out_dir / args.root_name
    if args.url and not root_path.exists():
        print(f"[prep] Downloading ROOT from {args.url}")
        download_file(args.url, root_path)

    if not root_path.exists():
        raise SystemExit(f"ROOT file not found: {root_path}")

    tree_key, tree = open_tree(root_path, args.tree)
    n_entries = min(args.nmax, tree.num_entries)
    print(f"[prep] Tree: {tree_key} | events: {n_entries:,} / {tree.num_entries:,}")

    numeric_arrays = load_numeric_arrays(tree, n_entries)
    selected = list(numeric_arrays.keys())
    print(f"[prep] Selected {len(selected)} readable numeric branches")

    if not selected:
        raise RuntimeError(
            "No readable numeric branches found in this ROOT file. "
            "Try another ALICE source/record or pass --input-root with a simpler tree."
        )

    arrs = ak.zip(numeric_arrays, depth_limit=1)

    data, meta = encode_to_bin(arrs, selected, args.max_list_len)
    meta.tree_key = tree_key

    bin_path = out_dir / args.bin_out
    meta_path = out_dir / args.meta_out
    if not args.force and (bin_path.exists() or meta_path.exists()):
        raise SystemExit("Output exists. Re-run with --force to overwrite.")

    data.tofile(bin_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(meta.to_json())

    print(f"[prep] Wrote {bin_path} ({data.shape[0]:,}x{data.shape[1]:,})")
    print(f"[prep] Wrote {meta_path}")

    write_event_splits(bin_path, meta.n_events, data.shape[1], force=args.force)


if __name__ == "__main__":
    main()
