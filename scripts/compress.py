# Usage: python scripts/compress.py --model MODEL --input INPUT --output OUTPUT --config CONFIG
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, struct, time, yaml
import torch, numpy as np
from tqdm import tqdm

from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
from vortex.compression.range_coder_gpu import gpu_encode, theoretical_bpd
from vortex.utils.training import load_checkpoint

MAGIC = b"VXC3"   # first-byte-raw + next-token range-coded format


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True)
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="experiments/atlas_experiment/config.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use-memory", action="store_true",
                   help="Enable cross-chunk compressed memory during encode (must match decode)")
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
        use_tdt=m.get("use_tdt", False),
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

    # Mixed-precision dtype (fp16 on CUDA, bf16 on CPU if available)
    amp_device = "cuda" if args.device == "cuda" else "cpu"
    use_amp    = (args.device == "cuda")

    t0                    = time.time()
    compressed_chunks     = []
    total_bpd             = 0.0
    memories              = None          # cross-chunk infini-attention state (optional)

    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Compressing"):
            x = torch.from_numpy(chunk.astype(np.int64)).unsqueeze(0).to(args.device)
            # (1, T) — values in [0, 255]

            with torch.amp.autocast(amp_device, enabled=use_amp):
                probs, _, _ = model.get_probs(
                    x,
                    memories if args.use_memory else None,
                )

            # Training/eval objective predicts x[t+1] from position t.
            # Encode the first byte raw, then arithmetic-code x[1:] using probs[:, :-1].
            probs_f32 = probs[:, :-1, :].float()         # (1, T-1, 256)
            syms      = x[:, 1:]                          # (1, T-1)
            first_raw = bytes([int(x[0, 0].item())])
            encoded   = gpu_encode(probs_f32, syms) if syms.numel() > 0 else b""
            compressed_chunks.append(first_raw + encoded)

            tail_bpd = theoretical_bpd(probs_f32, syms) if syms.numel() > 0 else 0.0
            chunk_bpd = (8.0 + tail_bpd * max(0, x.size(1) - 1)) / x.size(1)
            total_bpd += chunk_bpd

            # Keep memory update optional; decode must use the same setting.
            if args.use_memory:
                with torch.amp.autocast(amp_device, enabled=use_amp):
                    _, memories, _ = model(x, memories)
                memories = [mm.detach() if mm is not None else None for mm in memories]

    mean_bpd = total_bpd / len(chunks)
    with open(args.output, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">Q", original_size))
        f.write(struct.pack(">I", len(compressed_chunks)))
        for cb in compressed_chunks:
            f.write(struct.pack(">I", len(cb)))
            f.write(cb)

    elapsed = time.time() - t0
    csize   = os.path.getsize(args.output)
    ratio   = csize / original_size
    print(f"\nOutput : {args.output}  ({csize/1024/1024:.2f} MB)")
    print(f"BPD    : {mean_bpd:.4f}")
    print(f"Ratio  : {ratio:.3f}  ({1/ratio:.2f}x compression)")
    print(f"Speed  : {original_size/1e6/elapsed:.3f} MB/s")


if __name__ == "__main__":
    main()