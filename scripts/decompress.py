#!/usr/bin/env python3
# Usage: python scripts/decompress.py --help
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, struct, time, yaml
import torch, numpy as np
from tqdm import tqdm

from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
from vortex.compression.arithmetic_coding import TORCHAC_AVAILABLE
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

    with open(args.input, "rb") as f:
        assert f.read(4) == MAGIC, f"{args.input} is not a valid .vxc file"
        original_size = struct.unpack(">Q", f.read(8))[0]
        n_chunks      = struct.unpack(">I", f.read(4))[0]
        blobs         = [f.read(struct.unpack(">I", f.read(4))[0]) for _ in range(n_chunks)]

    print(f"Decompressing {n_chunks} chunks -> {original_size/1024/1024:.2f} MB")
    chunk_size    = c["window_size"]
    output, memories = [], None
    t0 = time.time()

    with torch.no_grad():
        for blob in tqdm(blobs, desc="Decompressing"):
            decoded   = []
            x         = torch.zeros(1, 1, dtype=torch.long, device=args.device)
            kv_caches = None
            for _ in range(chunk_size):
                # KV cache: only recompute the new token each step (O(1) per step)
                probs, memories, kv_caches = model.get_probs(x, memories, kv_caches)
                byte_val = int(probs[0, -1].argmax().item())
                decoded.append(byte_val)
                x = torch.tensor([[byte_val]], device=args.device)
            memories  = [mm.detach() if mm is not None else None for mm in memories]
            kv_caches = None  # reset KV cache between chunks
            output.extend(decoded)

    elapsed = time.time() - t0
    arr = np.array(output[:original_size], dtype=np.uint8)
    arr.tofile(args.output)
    print(f"
Saved : {args.output}  ({original_size/1024/1024:.2f} MB)")
    print(f"Speed : {original_size/1e6/elapsed:.4f} MB/s")
    print(f"Verify: python -c \"")
    print(f"  import hashlib, sys")
    print(f"  h = lambda f: hashlib.md5(open(f,\'rb\').read()).hexdigest()")
    print(f"  print(\'OK\' if h(sys.argv[1])==h(sys.argv[2]) else \'MISMATCH\')")
    print(f'\" {args.output.replace("_recovered","").replace(".vxc",".bin")} {args.output}')


if __name__ == "__main__":
    main()
