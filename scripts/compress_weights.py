#!/usr/bin/env python3
# Usage: python scripts/compress_weights.py --checkpoint CKPT --output OUT --config CONFIG
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, yaml
import torch

from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
from vortex.utils.zipnn import (
    save_compressed, load_compressed, weight_size_report,
    compress_model_weights,
)


def parse_args():
    p = argparse.ArgumentParser(description="ZipNN post-training weight compression")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a standard vortex .pt checkpoint (output of train.py)")
    p.add_argument("--output",     required=True,
                   help="Output path for the compressed .zipnn.pt file")
    p.add_argument("--config",     required=True,
                   help="Experiment config.yaml used during training")
    p.add_argument("--report",     action="store_true",
                   help="Print a per-tensor size breakdown after compression")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    m, c = cfg["model"], cfg["compressive_memory"]

    device = "cpu"

    model = OptimisedCompressiveTransformer(
        vocab_size=m["vocab_size"],
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        d_ff=m["d_ff"],
        window=c["window_size"],
        compression_rate=c["compression_rate"],
        dropout=m.get("dropout", 0.1),
        use_tdt=m.get("use_tdt", False),
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)   # handle both bare state-dict and wrapped
    model.load_state_dict(state, strict=False)
    print(f"\n[zipnn] Loaded checkpoint: {args.checkpoint}")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[zipnn] Model parameters : {params:.1f}M")

    compressed = compress_model_weights(model)

    if args.report:
        weight_size_report(compressed)

    torch.save(compressed, args.output)
    orig_mb = os.path.getsize(args.checkpoint) / 1e6
    comp_mb = os.path.getsize(args.output)     / 1e6
    print(f"\n[zipnn] {args.checkpoint}  ({orig_mb:.1f} MB)")
    print(f"[zipnn] → {args.output}  ({comp_mb:.1f} MB)  "
          f"({orig_mb/max(comp_mb,0.01):.2f}× smaller)")

    print("\n[zipnn] Verifying round-trip fidelity...")
    model2 = OptimisedCompressiveTransformer(
        vocab_size=m["vocab_size"], d_model=m["d_model"],
        n_layers=m["n_layers"],    n_heads=m["n_heads"],
        d_ff=m["d_ff"],            window=c["window_size"],
        compression_rate=c["compression_rate"],
        dropout=m.get("dropout", 0.1),
        use_tdt=m.get("use_tdt", False),
    )
    load_compressed(model2, args.output, device=device)

    # Compare a few weight tensors for exact bit equality
    errors = []
    for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                   model2.named_parameters()):
        if not torch.equal(p1.data.float(), p2.data.float()):
            errors.append(n1)
    if errors:
        print(f"[zipnn] MISMATCH in {len(errors)} tensors: {errors[:3]}")
    else:
        print("[zipnn] ✓ All weights restored bit-exactly.\n")


if __name__ == "__main__":
    main()
