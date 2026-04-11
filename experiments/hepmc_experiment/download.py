#!/usr/bin/env python3
"""
Download multiple HEPMC tarballs from CERN EOS, extract and concatenate them,
then split into train / val / test:

  data/hepmc_full.hepmc        : concatenated byte stream of all downloaded files
  data/hepmc_full_train.hepmc  : 80% train split
  data/hepmc_full_val.hepmc    : 10% validation split
  data/hepmc_full_test.hepmc   : 10% test split

Source pattern:
  root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc16_13TeV/
    HEPMC.43646133._000001.tar.gz.1  (and _000002, _000003, ...)

Usage:
  # Download 10 files (default) — gives ~5-10 GB, test split ~500MB-1GB
  python download.py

  # Download a specific number of files
  python download.py --num-files 5

  # Use a custom base URL (files numbered _000001 through _000NNN)
  python download.py --num-files 10 --base-url root://eospublic.cern.ch//eos/...

Notes:
  - Splits are 80/10/10 by raw byte offsets.
  - Files that fail to download are skipped (warning printed); pipeline continues
    as long as at least one file succeeds.
  - Re-run with --force to overwrite existing outputs.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import requests
from tqdm import tqdm


# Base URL pattern — _NNNNNN suffix is appended per file number
BASE_URL = (
	"root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc16_13TeV/HEPMC.43646133"
)
DEFAULT_NUM_FILES = 10   # 10 files ≈ 5–10 GB extracted; test split ~500MB–1GB


def file_url(base: str, n: int) -> str:
	"""Build the numbered URL, e.g. base + '._000003.tar.gz.1'."""
	return f"{base}._{n:06d}.tar.gz.1"


def root_to_https(root_url: str) -> str:
	"""Convert an xrootd URL to an HTTPS URL for eospublic.cern.ch.

	Example:
	  root://eospublic.cern.ch//eos/opendata/... -> https://eospublic.cern.ch/eos/opendata/...
	"""
	prefix = "root://eospublic.cern.ch/"
	if not root_url.startswith(prefix):
		return root_url  # fallback: return as-is
	path = root_url[len(prefix) :]
	# Ensure single leading slash after host in https
	if not path.startswith("/"):
		path = "/" + path
	return "https://eospublic.cern.ch" + path


def has_aria2c() -> bool:
	return shutil.which("aria2c") is not None


def has_xrdcp() -> bool:
	try:
		subprocess.run(["xrdcp", "--version"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		return True
	except FileNotFoundError:
		return False


def download_file(url: str, dest: Path, force: bool = False) -> Path:
	"""Download a file using aria2c (preferred), xrdcp, or HTTPS fallback.

	Returns the path to the downloaded file.
	"""
	dest.parent.mkdir(parents=True, exist_ok=True)
	if dest.exists() and not force:
		print(f"File already exists: {dest} (use --force to re-download)")
		return dest

	https_url = root_to_https(url)

	# 1. aria2c — 16 parallel connections, fastest option
	if has_aria2c():
		print(f"[download] aria2c -> {dest}")
		try:
			subprocess.check_call([
				"aria2c",
				"-x16", "-s16", "-j16", "-k1M",
				"--check-certificate=false",
				"--summary-interval=10",
				"--allow-overwrite=true",
				"-o", dest.name,
				"-d", str(dest.parent),
				https_url,
			])
			return dest
		except subprocess.CalledProcessError as e:
			print(f"[download] aria2c failed ({e}), trying next method...")

	# 2. xrdcp fallback for root:// URLs
	if url.startswith("root://") and has_xrdcp():
		print(f"[download] xrdcp -> {dest}")
		try:
			subprocess.check_call(["xrdcp", "-f", "--insecure", url, str(dest)])
			return dest
		except subprocess.CalledProcessError as e:
			print(f"[download] xrdcp failed ({e}), trying HTTPS...")

	# 3. Plain HTTPS fallback
	print(f"[download] HTTPS -> {dest}")
	try:
		with requests.get(https_url, stream=True, timeout=60) as r:
			r.raise_for_status()
			total = int(r.headers.get("Content-Length", 0))
			with open(dest, "wb") as f, tqdm(
				total=total if total > 0 else None,
				unit="B", unit_scale=True,
				desc=f"Downloading {dest.name}",
			) as pbar:
				for chunk in r.iter_content(chunk_size=1024 * 1024):
					if chunk:
						f.write(chunk)
						if total > 0:
							pbar.update(len(chunk))
		return dest
	except Exception as e:
		raise RuntimeError(f"All download methods failed: {e}") from e


def safe_extract_tar(tar_path: Path, extract_dir: Path) -> None:
	"""Safely extract a tar(.gz) archive to extract_dir, preventing path traversal."""
	extract_dir.mkdir(parents=True, exist_ok=True)

	def is_within_directory(directory: Path, target: Path) -> bool:
		try:
			directory = directory.resolve(strict=False)
			target = target.resolve(strict=False)
		except Exception:
			# resolve(strict=False) can still raise in some edge cases; fallback to simple check
			pass
		return os.path.commonpath([str(directory), str(target)]) == str(directory)

	with tarfile.open(tar_path, mode="r:*") as tf:
		members = tf.getmembers()
		for m in members:
			target_path = extract_dir / m.name
			if not is_within_directory(extract_dir, target_path):
				raise RuntimeError(f"Unsafe path in tar archive: {m.name}")
		tf.extractall(path=extract_dir)


def find_hepmc_file(search_dir: Path) -> Path:
	"""Find a HEPMC payload file after extraction.

	Preference order:
	  1) First file with .hepmc (case-insensitive)
	  2) First file with .hepmc.gz (will be gunzipped on-the-fly)
	  3) If none found, choose the largest regular file
	"""
	hepmc_candidates = []
	gz_candidates = []
	other_files = []
	for root, _dirs, files in os.walk(search_dir):
		for fname in files:
			p = Path(root) / fname
			if p.is_file():
				lname = fname.lower()
				if lname.endswith(".hepmc"):
					hepmc_candidates.append(p)
				elif lname.endswith(".hepmc.gz"):
					gz_candidates.append(p)
				else:
					other_files.append(p)

	if hepmc_candidates:
		return hepmc_candidates[0]
	if gz_candidates:
		# On-the-fly gunzip into a temp file
		import gzip

		gz_path = gz_candidates[0]
		tmp_out = gz_path.with_suffix("")  # drop .gz
		with gzip.open(gz_path, "rb") as fin, open(tmp_out, "wb") as fout:
			shutil.copyfileobj(fin, fout)
		return tmp_out

	if other_files:
		# Choose the largest file as a fallback heuristic
		return max(other_files, key=lambda p: p.stat().st_size)

	raise FileNotFoundError("No files found in the extracted archive")


def write_truncated_copy(src: Path, dest: Path, limit_bytes: int) -> None:
	"""Write the first limit_bytes from src into dest."""
	with open(src, "rb") as fin, open(dest, "wb") as fout:
		remaining = limit_bytes
		bufsize = 4 * 1024 * 1024  # 4 MiB chunks
		total_written = 0
		while remaining > 0:
			to_read = min(bufsize, remaining)
			chunk = fin.read(to_read)
			if not chunk:
				break
			fout.write(chunk)
			total_written += len(chunk)
			remaining -= len(chunk)
	size = dest.stat().st_size if dest.exists() else 0
	if size < limit_bytes:
		print(
			f"Warning: source smaller than {limit_bytes} bytes. Wrote only {size} bytes to {dest.name}."
		)


def write_splits(src: Path, splits: tuple = (0.8, 0.1, 0.1), force: bool = False) -> None:
	"""Write 80/10/10 train/val/test byte-level splits alongside *src*."""
	total   = src.stat().st_size
	names   = ["train", "val", "test"]
	sizes   = [int(total * f) for f in splits]
	sizes[-1] = total - sum(sizes[:-1])   # last slice absorbs rounding remainder
	offsets = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
	bufsize = 4 * 1024 * 1024
	for name, start, end in zip(names, offsets[:-1], offsets[1:]):
		out = src.parent / f"{src.stem}_{name}{src.suffix}"
		if out.exists() and not force:
			print(f"  [split] {name}: {out.name} already exists (use --force)")
			continue
		with open(src, "rb") as fin, open(out, "wb") as fout:
			fin.seek(start)
			remaining = end - start
			while remaining > 0:
				chunk = fin.read(min(bufsize, remaining))
				if not chunk:
					break
				fout.write(chunk)
				remaining -= len(chunk)
		written = out.stat().st_size if out.exists() else 0
		print(f"  [split] {name}: {out.name}  ({written / 1e9:.2f} GB)")


def append_to_file(src: Path, dst: Path) -> int:
	"""Append contents of src to dst. Returns bytes appended."""
	bufsize = 4 * 1024 * 1024
	appended = 0
	with open(src, "rb") as fin, open(dst, "ab") as fout:
		while True:
			chunk = fin.read(bufsize)
			if not chunk:
				break
			fout.write(chunk)
			appended += len(chunk)
	return appended


def main(argv: Optional[list[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Download and prepare HEPMC files")
	parser.add_argument("--force", action="store_true", help="Re-download and overwrite all outputs")
	parser.add_argument(
		"--num-files", type=int, default=DEFAULT_NUM_FILES,
		help=f"Number of numbered HEPMC files to download (default: {DEFAULT_NUM_FILES})",
	)
	parser.add_argument(
		"--base-url", default=BASE_URL,
		help="Base URL prefix (file number + .tar.gz.1 appended automatically).",
	)
	args = parser.parse_args(argv)

	workdir  = Path(__file__).parent.resolve()
	data_dir = workdir / "data"
	data_dir.mkdir(parents=True, exist_ok=True)
	full_out = data_dir / "hepmc_full.hepmc"

	if full_out.exists() and not args.force:
		print(f"[info] {full_out} already exists ({full_out.stat().st_size / 1e9:.2f} GB). "
			  f"Use --force to rebuild.")
	else:
		# Remove partial/old full file before concatenating
		if full_out.exists():
			full_out.unlink()

		successful = 0
		for n in range(1, args.num_files + 1):
			url      = file_url(args.base_url, n)
			filename = url.split("/")[-1]
			tar_dest = workdir / filename

			print(f"\n[file {n}/{args.num_files}] {filename}")
			try:
				downloaded = download_file(url, tar_dest, force=args.force)
			except RuntimeError as e:
				print(f"  [warning] Skipping file {n}: {e}")
				continue

			with tempfile.TemporaryDirectory(prefix="hepmc_extract_") as tmpdir:
				tmpdir_p = Path(tmpdir)
				print(f"  Extracting...")
				try:
					safe_extract_tar(downloaded, tmpdir_p)
					hepmc_payload = find_hepmc_file(tmpdir_p)
					payload_mb = hepmc_payload.stat().st_size / 1e6
					print(f"  Payload: {hepmc_payload.name}  ({payload_mb:.0f} MB)")
					appended = append_to_file(hepmc_payload, full_out)
					print(f"  Appended {appended / 1e6:.0f} MB -> "
						  f"{full_out.stat().st_size / 1e9:.2f} GB total")
					successful += 1
				except Exception as e:
					print(f"  [warning] Extract/append failed for file {n}: {e}")
					continue

			# Clean up tar after successful extract to save disk space
			tar_dest.unlink(missing_ok=True)

		if successful == 0:
			print("\n[error] No HEPMC files were successfully downloaded and extracted.")
			return 1

		total_gb = full_out.stat().st_size / 1e9
		print(f"\n[concat] hepmc_full.hepmc: {total_gb:.2f} GB from {successful} file(s)")

		# Size check: warn if test split will be under 500 MB
		test_gb = total_gb * 0.1
		if test_gb < 0.5:
			print(f"  [warning] Test split will be {test_gb*1024:.0f} MB — under 500 MB.")
			print(f"  [warning] Re-run with --num-files {int(5 / total_gb * args.num_files) + 1} "
				  f"to reach ~500 MB test split.")
		else:
			print(f"  Test split will be ~{test_gb*1024:.0f} MB  ✓")

	# Generate / refresh splits
	print("\nGenerating 80/10/10 train/val/test byte splits...")
	write_splits(full_out, force=args.force)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

