# Usage: python scripts/decompress.py --model MODEL --input INPUT --output OUTPUT --config CONFIG
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, struct, time, yaml
import torch, numpy as np
from tqdm import tqdm

from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
from vortex.compression.range_coder_gpu import (
    StreamDecoder, make_gpu_cdf, RANGE_CODER_AVAILABLE,
)
from vortex.utils.training import load_checkpoint

MAGIC = b"VXC2"   # must match compressor


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
        magic = f.read(4)
        assert magic == MAGIC, (
            f"{args.input}: expected magic {MAGIC} (VXC2 lossless format), "
            f"got {magic}. Re-compress with the updated compress.py."
        )
        original_size = struct.unpack(">Q", f.read(8))[0]
        n_chunks      = struct.unpack(">I", f.read(4))[0]
        blobs         = [f.read(struct.unpack(">I", f.read(4))[0]) for _ in range(n_chunks)]

    chunk_size = c["window_size"]
    print(f"Decompressing {n_chunks} chunks -> {original_size/1024/1024:.2f} MB")

    amp_device = "cuda" if args.device == "cuda" else "cpu"
    use_amp    = (args.device == "cuda")

    output   = []
    memories = None          # cross-chunk infini-attention state
    SOS      = torch.zeros(1, 1, dtype=torch.long, device=args.device)  # start-of-sequence
    t0       = time.time()

    with torch.no_grad():
        for blob in tqdm(blobs, desc="Decompressing"):
            # ── Autoregressive decode ──────────────────────────────────────
            # Mirror of compress.py's shifted-input logic:
            #   encode used probs from model([SOS, x_0, …, x_{T-2}], memories)
            #   so decode must reconstruct exactly those probs, token by token.
            #
            # With KV cache: each step is a single-token forward pass O(1)
            # attention instead of O(t) re-processing from scratch.
            dec       = StreamDecoder(blob)   # stateful range-coder reader
            kv_caches = None
            x_in      = SOS                   # first input is SOS (= 0)
            decoded   = []

            with torch.amp.autocast(amp_device, enabled=use_amp):
                for _ in range(chunk_size):
                    # Single-token forward — uses memories as infini-attention
                    # context (fixed for whole chunk) and grows kv_caches.
                    probs_step, _, kv_caches = model.get_probs(
                        x_in, memories, kv_caches
                    )

                # probs_step[:, -1, :] = P(next_token | all seen so far, memories)
                    cdf_row = make_gpu_cdf(probs_step[:, -1:, :].float())[0]  # (257,)
                    sym     = dec.decode_symbol(cdf_row)   # recovers x[t] exactly
                    decoded.append(sym)

                    # Feed decoded symbol back as next input
                    x_in = torch.tensor([[sym]], dtype=torch.long, device=args.device)

            # ── Memory update (mirrors compress.py exactly) ───────────────
            # Run the full recovered chunk through the model to produce the
            # same cross-chunk memory state that compress.py computed.
            x_full = torch.tensor(decoded, dtype=torch.long,
                                  device=args.device).unsqueeze(0)  # (1, T)
            with torch.amp.autocast(amp_device, enabled=use_amp):
                _, memories, _ = model(x_full, memories)
            memories = [mm.detach() if mm is not None else None for mm in memories]

            output.extend(decoded)

    elapsed = time.time() - t0
    arr = np.array(output[:original_size], dtype=np.uint8)
    arr.tofile(args.output)

    print(f"\nSaved : {args.output}  ({original_size/1024/1024:.2f} MB)")
    print(f"Speed : {original_size/1e6/elapsed:.4f} MB/s")

    orig_bin = args.input.replace(".vxc", ".bin")
    print(f"\nVerify:")
    print(f"  python -c \"import hashlib; h=lambda f:hashlib.md5(open(f,'rb').read()).hexdigest(); "
          f"print('OK' if h('{orig_bin}')==h('{args.output}') else 'MISMATCH')\"")


if __name__ == "__main__":
    main()