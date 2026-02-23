# Usage: python scripts/evaluate.py --model MODEL --data DATA --config CONFIG --device cuda
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, gzip, zlib, lzma, bz2, math, time, yaml
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import zstandard as zstd
    ZSTD = True
except ImportError:
    ZSTD = False

try:
    import brotli
    BROTLI = True
except ImportError:
    BROTLI = False

try:
    import lz4.frame as lz4
    LZ4 = True
except ImportError:
    LZ4 = False

from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
from vortex.data.dataset import MemmapWindowDataset
from vortex.utils.training import load_checkpoint, get_amp_dtype



SAMPLE_MB = 50

def _read_sample(path: str, max_mb: float = SAMPLE_MB) -> bytes:
    max_bytes = int(max_mb * 1024 * 1024)
    with open(path, "rb") as f:
        return f.read(max_bytes)


def _make_eval_loader(path: str, window: int, batch_size: int) -> DataLoader:
    ds = MemmapWindowDataset(path, window=window, stride=window)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def _stats(original: bytes, compressed: bytes, elapsed: float) -> dict:
    n = len(original)
    c = len(compressed)
    return {
        "bpd":       c * 8 / n,
        "ratio_x":   n / c,
        "size_mb":   c / 1024 / 1024,
        "speed_mbs": n / 1e6 / elapsed if elapsed > 0 else 0.0,
    }


def run_baseline(name: str, data: bytes, compress_fn) -> dict:
    t0      = time.time()
    comp    = compress_fn(data)
    elapsed = time.time() - t0
    return {"name": name, **_stats(data, comp, elapsed)}


def all_baselines(data: bytes) -> list:
    results = []

    for lvl in [1, 6, 9]:
        results.append(run_baseline(
            f"Gzip  (L{lvl})", data,
            lambda d, l=lvl: gzip.compress(d, compresslevel=l)))

    for lvl in [1, 6, 9]:
        results.append(run_baseline(
            f"Zlib  (L{lvl})", data,
            lambda d, l=lvl: zlib.compress(d, level=l)))

    for lvl in [1, 9]:
        results.append(run_baseline(
            f"Bz2   (L{lvl})", data,
            lambda d, l=lvl: bz2.compress(d, compresslevel=l)))

    for preset in [6, 9]:
        results.append(run_baseline(
            f"LZMA  (P{preset})", data,
            lambda d, p=preset: lzma.compress(d, preset=p)))

    if ZSTD:
        for lvl in [1, 3, 9, 19]:
            results.append(run_baseline(
                f"Zstd  (L{lvl})", data,
                lambda d, l=lvl: zstd.ZstdCompressor(level=l).compress(d)))

    if BROTLI:
        for q in [1, 6, 11]:
            results.append(run_baseline(
                f"Brotli(Q{q:2d})", data,
                lambda d, qq=q: brotli.compress(d, quality=qq)))

    if LZ4:
        results.append(run_baseline(
            "LZ4   (default)", data,
            lambda d: lz4.compress(d)))

    return results



def print_table(rows: list, vortex: dict = None):
    gzip6_bpd = next((r["bpd"] for r in rows if "Gzip" in r["name"] and "L6" in r["name"]), None)

    W = [18, 8, 8, 9, 11, 10]
    sep  = "+" + "+".join("-" * (w + 2) for w in W) + "+"
    head = ["Codec", "BPD", "Ratio", "Size MB", "Speed MB/s", "vs Gzip-6"]

    def fmt_row(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, W)) + " |"

    print(sep)
    print(fmt_row(head))
    print(sep)

    for r in rows:
        if gzip6_bpd:
            delta = f"{(gzip6_bpd - r['bpd']) / gzip6_bpd * 100:+.1f}%"
        else:
            delta = "n/a"
        print(fmt_row([
            r["name"],
            f"{r['bpd']:.4f}",
            f"{r['ratio_x']:.2f}x",
            f"{r['size_mb']:.2f}",
            f"{r['speed_mbs']:.1f}",
            delta,
        ]))

    if vortex is not None and vortex["bpd"] != float("inf"):
        print(sep)
        v_delta = f"{(gzip6_bpd - vortex['bpd']) / gzip6_bpd * 100:+.1f}%" if gzip6_bpd else "n/a"
        print(fmt_row([
            "* Vortex-Codec",
            f"{vortex['bpd']:.4f}",
            f"{vortex['ratio_x']:.2f}x",
            "(theoretical)",
            "—",
            v_delta,
        ]))
    print(sep)



def eval_vortex(model, dl, vocab_size, device, amp_dtype, max_tokens=None) -> dict:
    criterion = nn.CrossEntropyLoss()
    total_nats, total_tokens = 0.0, 0
    fp16 = device == "cuda"
    dev_type = device.split(":")[0]  # "cuda:0" -> "cuda"

    with torch.no_grad():
        for batch in tqdm(dl, desc="  Vortex"):
            batch = batch.to(device)
            # Each window is independent — no meaningful past context across batches.
            with torch.amp.autocast(dev_type, enabled=fp16, dtype=amp_dtype):
                logits, _, _ = model(batch, None)
                loss = criterion(
                    logits[:, :-1].reshape(-1, vocab_size),
                    batch[:, 1:].reshape(-1),
                )
            n_tok = batch.size(0) * (batch.size(1) - 1)
            total_nats   += loss.item() * n_tok
            total_tokens += n_tok
            if max_tokens and total_tokens >= max_tokens:
                break

    bpd = total_nats / total_tokens / math.log(2)
    return {
        "bpd":     bpd,
        "ratio_x": 8 / bpd,
    }



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         required=False, default=None)
    p.add_argument("--data",          required=True)
    p.add_argument("--config",       default="experiments/atlas_experiment/config.yaml")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-baselines", action="store_true", help="Skip baseline codecs")
    p.add_argument("--sample-mb",    type=float, default=SAMPLE_MB,
                   help=f"MB of data to use for baselines (default: {SAMPLE_MB})")
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--full-vortex",   action="store_true",
                   help="Evaluate Vortex on full file; baselines still use --sample-mb")
    p.add_argument("--baselines-only", action="store_true",
                   help="Run baseline codecs only, skip Vortex model")
    return p.parse_args()



def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_size_gb = os.path.getsize(args.data) / 1e9
    exp = cfg.get("experiment", {}).get("name", "experiment")

    amp_dtype = get_amp_dtype(args.device)

    print(f"\n{'='*62}")
    print(f"  Vortex-Codec Compression Benchmark")
    print(f"  Experiment : {exp}")
    print(f"  Test file  : {args.data}  ({data_size_gb:.2f} GB)")
    print(f"  Device     : {args.device}  |  AMP dtype: {amp_dtype}")
    print(f"{'='*62}\n")

    baseline_results = []
    if not args.no_baselines:
        print(f"  Running baselines on {args.sample_mb:.0f} MB sample "
              f"(not full {data_size_gb:.1f} GB — use --sample-mb to adjust)...")
        sample = _read_sample(args.data, args.sample_mb)
        baseline_results = all_baselines(sample)
        del sample

        missing = []
        if not ZSTD:   missing.append("zstandard")
        if not BROTLI: missing.append("brotli")
        if not LZ4:    missing.append("lz4")
        if missing:
            print(f"  [INFO] Optional codecs not installed: pip install {' '.join(missing)}\n")

    if args.baselines_only:
        if baseline_results:
            print()
            print_table(baseline_results)
        else:
            print("  No baselines run (--no-baselines set).")
        return

    if not args.model:
        print("[ERROR] --model is required unless --baselines-only is set.")
        sys.exit(1)

    m, c = cfg["model"], cfg["compressive_memory"]
    model = OptimisedCompressiveTransformer(
        vocab_size=m["vocab_size"],       d_model=m["d_model"],
        n_layers=m["n_layers"],           n_heads=m["n_heads"],
        d_ff=m["d_ff"],                   window=c["window_size"],
        compression_rate=c["compression_rate"],
        dropout=m.get("dropout", 0.1),
        use_tdt=m.get("use_tdt", False),
    ).to(args.device)
    load_checkpoint(model, args.model, device=args.device)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model : {params:.1f}M parameters\n")

    dl = _make_eval_loader(args.data, c["window_size"], args.batch_size)

    max_tok = None if args.full_vortex else int(args.sample_mb * 1024 * 1024)
    vortex = eval_vortex(model, dl, m["vocab_size"], args.device, amp_dtype, max_tok)

    print()
    if baseline_results:
        print_table(baseline_results, vortex)
    else:
        print(f"  Vortex BPD : {vortex['bpd']:.4f}  ({vortex['ratio_x']:.2f}x)")

    print(f"\n{'='*62}")
    print(f"  Summary")
    print(f"{'='*62}")
    print(f"  Vortex BPD : {vortex['bpd']:.4f}  ({vortex['ratio_x']:.2f}x)")
    print(f"  Evaluated on full test set ({data_size_gb:.2f} GB)")
    print(f"  Baselines on {args.sample_mb:.0f} MB sample")
    if baseline_results:
        for tag, key in [("Gzip-6",  "Gzip  (L6)"),
                         ("Zlib-9",  "Zlib  (L9)"),
                         ("LZMA-9",  "LZMA  (P9)"),
                         ("Zstd-3",  "Zstd  (L3)"),
                         ("Zstd-19", "Zstd  (L19)"),
                         ("LZMA-6",  "LZMA  (P6)")]:
            row = next((r for r in baseline_results if r["name"] == key), None)
            if row:
                delta = (row["bpd"] - vortex["bpd"]) / row["bpd"] * 100
                print(f"  vs {tag:10s}: {row['bpd']:.4f} BPD  -> Vortex {delta:+.1f}%")

        best = min(baseline_results, key=lambda r: r["bpd"])
        delta = (best["bpd"] - vortex["bpd"]) / best["bpd"] * 100
        print(f"\n  vs best baseline ({best['name'].strip()}: {best['bpd']:.4f} BPD)"
              f"  -> Vortex {delta:+.1f}%")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()