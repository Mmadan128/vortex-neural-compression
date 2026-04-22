#!/usr/bin/env python3
"""Single-config end-to-end pipeline runner.

Runs train (optional/auto), roundtrip verification, and evaluation benchmarks
using one config file so model architecture and paths remain consistent.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _python_exe() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_py if venv_py.exists() else Path(sys.executable))


def _infer_record_bytes(config_path: str, exp_name: str) -> int:
    p = config_path.lower()
    e = exp_name.lower()
    if "camel" in p or "camel" in e:
        return 44
    if "atlas" in p or "atlas" in e:
        return 102
    return 1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment YAML config")
    ap.add_argument("--device", default="cuda", help="cuda|cpu|auto")
    ap.add_argument("--model", default=None, help="Override checkpoint path")
    ap.add_argument("--train", action="store_true", help="Force training stage")
    ap.add_argument("--roundtrip", action="store_true",
                    help="Run roundtrip verification stage (off by default)")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--roundtrip-mb", type=int, default=10)
    ap.add_argument("--sample-mb", type=float, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use-memory", action="store_true",
                    help="Use cross-chunk memory in compress/decompress")
    return ap.parse_args()


def main():
    args = parse_args()
    py = _python_exe()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment", {}).get("name", cfg_path.stem)
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    model = Path(args.model) if args.model else (ckpt_dir / "best.pt")
    test_data = Path(cfg.get("evaluation", {}).get("test_data", ""))
    if not test_data:
        raise RuntimeError("Config missing evaluation.test_data")

    # Auto-train only when checkpoint is missing, or force when --train provided.
    do_train = args.train or (not model.exists())

    if do_train:
        _run([py, "scripts/train.py", "--config", str(cfg_path), "--device", args.device])
        if not model.exists():
            latest = ckpt_dir / "latest.pt"
            if latest.exists():
                model = latest

    if not model.exists():
        raise RuntimeError(f"Checkpoint not found: {model}")

    if args.roundtrip:
        rec_bytes = _infer_record_bytes(str(cfg_path), exp_name)
        sample_bytes = args.roundtrip_mb * 1024 * 1024

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            sample = tdp / "sample.bin"
            comp = tdp / "sample.vxc"
            decomp = tdp / "sample_out.bin"

            raw = test_data.read_bytes()[:sample_bytes]
            n = (len(raw) // rec_bytes) * rec_bytes
            sample.write_bytes(raw[:n])
            print(f"[roundtrip] sample bytes: {n}")

            ccmd = [
                py, "scripts/compress.py",
                "--model", str(model),
                "--input", str(sample),
                "--output", str(comp),
                "--config", str(cfg_path),
                "--device", args.device,
            ]
            dcmd = [
                py, "scripts/decompress.py",
                "--model", str(model),
                "--input", str(comp),
                "--output", str(decomp),
                "--config", str(cfg_path),
                "--device", args.device,
            ]
            if args.use_memory:
                ccmd.append("--use-memory")
                dcmd.append("--use-memory")

            _run(ccmd)
            _run(dcmd)

            if sample.read_bytes() != decomp.read_bytes():
                raise RuntimeError("Roundtrip failed: decompressed bytes do not match original")
            print("[roundtrip] PASS (byte-identical)")

    if not args.skip_eval:
        out = REPO_ROOT / "results" / f"{exp_name}_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        _run([
            py, "scripts/evaluate.py",
            "--model", str(model),
            "--data", str(test_data),
            "--config", str(cfg_path),
            "--device", args.device,
            "--sample-mb", str(args.sample_mb),
            "--vortex-mb", str(args.sample_mb),
            "--batch-size", str(args.batch_size),
            "--out-json", str(out),
        ])

        with open(out, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print(f"[eval] saved: {out}")
        print(f"[eval] vortex_bpd={payload['vortex']['bpd']}")


if __name__ == "__main__":
    main()
