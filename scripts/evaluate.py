#!/usr/bin/env python3
# Usage: python scripts/evaluate.py --help
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, gzip, math, yaml
import torch, torch.nn as nn
from tqdm import tqdm

try:
    import zstandard as zstd
    ZSTD = True
except ImportError:
    ZSTD = False

from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
from vortex.data.dataset import make_loaders
from vortex.utils.training import load_checkpoint


def baseline(data: bytes, method="gzip"):
    comp = (gzip.compress(data, compresslevel=6) if method == "gzip"
            else zstd.ZstdCompressor(level=3).compress(data))
    bpd  = len(comp) * 8 / len(data)
    return bpd, len(comp) / len(data)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True)
    p.add_argument("--data",   required=True)
    p.add_argument("--config", default="experiments/atlas_experiment/config.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw = open(args.data, "rb").read()
    exp = cfg.get("experiment", {}).get("name", "experiment")
    print(f"
Experiment : {exp}")
    print(f"Test file  : {args.data}  ({len(raw)/1024/1024:.2f} MB)")
    print("-" * 56)

    gz_bpd,  gz_r  = baseline(raw, "gzip")
    print(f"  Gzip   | BPD={gz_bpd:.4f}  ratio={gz_r:.3f}  ({1/gz_r:.2f}x)")
    if ZSTD:
        zst_bpd, zst_r = baseline(raw, "zstd")
        print(f"  Zstd   | BPD={zst_bpd:.4f}  ratio={zst_r:.3f}  ({1/zst_r:.2f}x)")

    m, c = cfg["model"], cfg["compressive_memory"]
    model = OptimisedCompressiveTransformer(
        vocab_size=m["vocab_size"], d_model=m["d_model"],
        n_layers=m["n_layers"],    n_heads=m["n_heads"],
        d_ff=m["d_ff"],            window=c["window_size"],
        compression_rate=c["compression_rate"],
    ).to(args.device)
    load_checkpoint(model, args.model, device=args.device)
    model.eval()

    dl, _ = make_loaders(args.data, window=c["window_size"],
                         stride=c["window_size"], batch_size=16, num_workers=2)
    criterion  = nn.CrossEntropyLoss()
    total, cnt = 0.0, 0
    mems       = None
    with torch.no_grad():
        for batch in tqdm(dl, desc="  Vortex"):
            batch = batch.to(args.device)
            logits, mems, _ = model(batch, mems)
            loss = criterion(logits[:, :-1].reshape(-1, m["vocab_size"]),
                             batch[:, 1:].reshape(-1))
            total += loss.item(); cnt += 1
            mems = [mm.detach() if mm is not None else None for mm in mems]

    vx_bpd = total / cnt / math.log(2)
    vx_r   = vx_bpd / 8
    print(f"  Vortex | BPD={vx_bpd:.4f}  ratio={vx_r:.3f}  ({1/vx_r:.2f}x)")
    print("-" * 56)
    print(f"  Improvement vs Gzip : {(gz_bpd - vx_bpd)/gz_bpd*100:.1f}%")
    if ZSTD:
        print(f"  Improvement vs Zstd : {(zst_bpd - vx_bpd)/zst_bpd*100:.1f}%")


if __name__ == "__main__":
    main()
