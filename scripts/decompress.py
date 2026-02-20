# Usage: python scripts/decompress.py --model MODEL --input INPUT --output OUTPUT --config CONFIG
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, struct, time, yaml
import torch, numpy as np
from tqdm import tqdm

from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
from vortex.compression.arithmetic_coding import decode, TORCHAC_AVAILABLE
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

    chunk_size = c["window_size"]
    print(f"Decompressing {n_chunks} chunks -> {original_size/1024/1024:.2f} MB")

    fp16     = args.device == "cuda"
    output   = []
    memories = None
    t0       = time.time()

    with torch.no_grad():
        for blob in tqdm(blobs, desc="Decompressing"):


            all_probs = []
            x         = torch.zeros(1, 1, dtype=torch.long, device=args.device)
            kv_caches = None
            mem_snapshot = memories

            with torch.amp.autocast("cuda", enabled=fp16):
                for pos in range(chunk_size):
                    probs_step, _, kv_caches = model.get_probs(x, mem_snapshot, kv_caches)
                    step_p = probs_step[:, -1:, :].float()
                    all_probs.append(step_p)

                    x = torch.zeros(1, 1, dtype=torch.long, device=args.device)


            dummy = torch.zeros(1, chunk_size, dtype=torch.long, device=args.device)
            with torch.amp.autocast("cuda", enabled=fp16):
                probs, _, _ = model.get_probs(dummy, mem_snapshot)
            probs = probs.float()

            decoded_tensor = decode(blob, probs)
            decoded = decoded_tensor[0].tolist()

            recovered = torch.tensor(decoded, dtype=torch.long,
                                     device=args.device).unsqueeze(0)
            with torch.amp.autocast("cuda", enabled=fp16):
                _, memories, _ = model(recovered, memories)
            memories = [mm.detach() if mm is not None else None for mm in memories]

            output.extend(decoded)

    elapsed = time.time() - t0
    arr = np.array(output[:original_size], dtype=np.uint8)
    arr.tofile(args.output)

    print(f"\nSaved : {args.output}  ({original_size/1024/1024:.2f} MB)")
    print(f"Speed : {original_size/1e6/elapsed:.4f} MB/s")

    orig_bin = args.input.replace(".vxc", ".bin")
    print(f"\nVerify:")
    print(f"  python -c \"import hashlib; h=lambda f:hashlib.md5(open(f,'rb').read()).hexdigest(); print('OK' if h('{orig_bin}')==h('{args.output}') else 'MISMATCH')\"")



if __name__ == "__main__":
    main()