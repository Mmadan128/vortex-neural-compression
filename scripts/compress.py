#!/usr/bin/env python3
# Usage: python scripts/compress.py --model MODEL --input INPUT --output OUTPUT --config CONFIG
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, struct, time, yaml
import torch, numpy as np
from tqdm import tqdm

from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
from vortex.compression.arithmetic_coding import encode, theoretical_bpd, TORCHAC_AVAILABLE
from vortex.utils.training import load_checkpoint

MAGIC = b"VXC1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True)
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="experiments/atlas_experiment/config.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if not TORCHAC_AVAILABLE:
        sys.exit("pip install torchac  (required for arithmetic coding)")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    m, c = cfg["model"], cfg["compressive_memory"]

    model = OptimisedCompressiveTransformer(
        vocab_size=m["vocab_size"], d_model=m["d_model"],
        n_layers=m["n_layers"],    n_heads=m["n_heads"],
        d_ff=m["d_ff"],            window=c["window_size"],
        compression_rate=c["compression_rate"],
    ).to(args.device)
    load_checkpoint(model, args.model, device=args.device)
    model.eval()

    data          = np.fromfile(args.input, dtype=np.uint8)
    original_size = len(data)
    chunk_size    = c["window_size"]
    pad           = (-len(data)) % chunk_size
    if pad:
        data = np.pad(data, (0, pad))
    chunks = data.reshape(-1, chunk_size)
    print(f"Input : {args.input}  ({original_size/1024/1024:.2f} MB,  {len(chunks)} chunks)")

    t0 = time.time()
    compressed_chunks, total_bpd, memories = [], 0.0, None
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Compressing"):
            x = torch.from_numpy(chunk.astype(np.int64)).unsqueeze(0).to(args.device)
            probs, memories, _ = model.get_probs(x, memories)
            memories = [mm.detach() if mm is not None else None for mm in memories]
            compressed_chunks.append(encode(probs, x.short()))
            total_bpd += theoretical_bpd(probs, x)

    mean_bpd = total_bpd / len(chunks)
    with open(args.output, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">Q", original_size))
        f.write(struct.pack(">I", len(compressed_chunks)))
        for cb in compressed_chunks:
            f.write(struct.pack(">I", len(cb)))
            f.write(cb)

    elapsed  = time.time() - t0
    csize    = os.path.getsize(args.output)
    ratio    = csize / original_size
    print(f"\nOutput : {args.output}  ({csize/1024/1024:.2f} MB)")
    print(f"BPD    : {mean_bpd:.4f}")
    print(f"Ratio  : {ratio:.3f}  ({1/ratio:.2f}x compression)")
    print(f"Speed  : {original_size/1e6/elapsed:.3f} MB/s")


if __name__ == "__main__":
    main()
