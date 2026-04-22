# Usage: python scripts/evaluate.py --model MODEL --data DATA --config CONFIG --device cuda
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, gzip, zlib, lzma, bz2, math, time, yaml
import concurrent.futures, os
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

from vortex.models.optimized_transformer import OptimisedCompressiveTransformer, CATWrapper
from vortex.data.dataset import MemmapWindowDataset
from vortex.utils.training import load_checkpoint, get_amp_dtype



SAMPLE_MB  = 1024  # default MB for both baselines and Vortex (1 GB)
DEFAULT_DATA = "experiments/atlas_experiment/data/mc-flavtag-ttbar-medium.bin"

def _read_sample(path: str, max_mb: float = SAMPLE_MB) -> bytes:
    file_size = os.path.getsize(path)
    max_bytes = min(int(max_mb * 1024 * 1024), file_size)
    with open(path, "rb") as f:
        return f.read(max_bytes)


def _make_eval_loader(path: str, window: int, batch_size: int) -> DataLoader:
    ds = MemmapWindowDataset(path, window=window, stride=window)
    # Use multiple workers so the CPU prepares batches ahead of the GPU.
    # 4 workers saturate prefetch on both small and large machines without
    # excessive memory duplication of the memmap.
    n_workers = min(4, os.cpu_count() or 1)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )


def _stats(original: bytes, compressed: bytes, elapsed: float) -> dict:
    n = len(original)
    c = len(compressed)
    return {
        "bpd":       c * 8 / n,
        "ratio_x":   n / c,
        "size_mb":   c / 1024 / 1024,
        "elapsed_s": elapsed,
        "speed_mbs": n / 1e6 / elapsed if elapsed > 0 else 0.0,
    }


def run_baseline(name: str, data: bytes, compress_fn) -> dict:
    t0      = time.time()
    comp    = compress_fn(data)
    elapsed = time.time() - t0
    return {"name": name, **_stats(data, comp, elapsed)}


def all_baselines(data: bytes) -> list:
    jobs = []

    for lvl in [1, 6, 9]:
        jobs.append((f"Gzip  (L{lvl})", data, lambda d, l=lvl: gzip.compress(d, compresslevel=l)))

    for lvl in [1, 6, 9]:
        jobs.append((f"Zlib  (L{lvl})", data, lambda d, l=lvl: zlib.compress(d, level=l)))

    for lvl in [1, 9]:
        jobs.append((f"Bz2   (L{lvl})", data, lambda d, l=lvl: bz2.compress(d, compresslevel=l)))

    for preset in [6, 9]:
        jobs.append((f"LZMA  (P{preset})", data, lambda d, p=preset: lzma.compress(d, preset=p)))

    if ZSTD:
        for lvl in [1, 3, 9, 19]:
            jobs.append((f"Zstd  (L{lvl})", data, lambda d, l=lvl: zstd.ZstdCompressor(level=l).compress(d)))

    if BROTLI:
        for q in [1, 6, 11]:
            jobs.append((f"Brotli(Q{q:2d})", data, lambda d, qq=q: brotli.compress(d, quality=qq)))

    if LZ4:
        jobs.append(("LZ4   (default)", data, lambda d: lz4.compress(d)))

    # Run all baselines in parallel — compression C-extensions release the GIL,
    # so threads achieve true parallelism.  LZMA-P9 and Brotli-Q11 on 1 GB can
    # each take several minutes single-threaded; running them concurrently
    # reduces wall-clock time to roughly the slowest single codec.
    n_workers = min(len(jobs), os.cpu_count() or 1)
    results_map: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_baseline, name, d, fn): name
                   for name, d, fn in jobs}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results_map[r["name"]] = r

    # Preserve original ordering
    return [results_map[name] for name, _, _ in jobs]



def print_table(rows: list, vortex: dict = None):
    gzip6_bpd = next((r["bpd"] for r in rows if "Gzip" in r["name"] and "L6" in r["name"]), None)

    #                  Codec   BPD   Ratio  Size MB  Time(s)  Speed MB/s  vs Gzip-6
    W = [18, 8, 8, 9, 9, 11, 10]
    sep  = "+" + "+".join("-" * (w + 2) for w in W) + "+"
    head = ["Codec", "BPD", "Ratio", "Size MB", "Time (s)", "Speed MB/s", "vs Gzip-6"]

    def fmt_row(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, W)) + " |"

    print(sep)
    print(fmt_row(head))
    print(sep)

    for r in rows:
        delta = f"{(gzip6_bpd - r['bpd']) / gzip6_bpd * 100:+.1f}%" if gzip6_bpd else "n/a"
        print(fmt_row([
            r["name"],
            f"{r['bpd']:.4f}",
            f"{r['ratio_x']:.2f}x",
            f"{r['size_mb']:.2f}",
            f"{r['elapsed_s']:.2f}s",
            f"{r['speed_mbs']:.1f}",
            delta,
        ]))

    if vortex is not None and vortex["bpd"] != float("inf"):
        print(sep)
        v_delta = f"{(gzip6_bpd - vortex['bpd']) / gzip6_bpd * 100:+.1f}%" if gzip6_bpd else "n/a"
        v_time  = f"{vortex['elapsed_s']:.2f}s" if vortex.get('elapsed_s') is not None else "—"
        print(fmt_row([
            "* Vortex-Codec",
            f"{vortex['bpd']:.4f}",
            f"{vortex['ratio_x']:.2f}x",
            "(theoretical)",
            v_time,
            f"{vortex.get('speed_mbs', 0):.1f}" if vortex.get('speed_mbs') else "—",
            v_delta,
        ]))
    print(sep)



def eval_vortex(model, dl, vocab_size, device, amp_dtype, max_tokens=None) -> dict:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_nats   = torch.zeros(1, device=device)
    total_tokens = 0
    fp16     = device == "cuda"
    dev_type = device.split(":")[0]  # "cuda:0" -> "cuda"

    # Accurate tqdm total: only count the batches we'll actually process
    if max_tokens is not None:
        try:
            bs = dl.batch_size
            w  = dl.dataset.window
            max_batches = math.ceil(max_tokens / (bs * (w - 1)))
        except Exception:
            max_batches = None
    else:
        max_batches = len(dl)

    # Warmup: one dummy batch so torch.compile JIT doesn't skew timing
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(dl, desc="  Vortex", total=max_batches):
            batch = batch.to(device)
            with torch.amp.autocast(dev_type, enabled=fp16, dtype=amp_dtype):
                logits, _, _ = model(batch, None)
                # Use sum reduction so we can accumulate on-GPU without .item()
                loss = criterion(
                    logits[:, :-1].reshape(-1, vocab_size),
                    batch[:, 1:].reshape(-1),
                )
            n_tok = batch.size(0) * (batch.size(1) - 1)
            total_nats   += loss.detach()   # stays on GPU — no sync per batch
            total_tokens += n_tok
            if max_tokens and total_tokens >= max_tokens:
                break

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed   = time.time() - t0
    total_nats = total_nats.item()          # single GPU→CPU transfer at the end

    data_mb  = total_tokens / 1e6
    bpd      = total_nats / total_tokens / math.log(2)
    return {
        "bpd":       bpd,
        "ratio_x":   8 / bpd,
        "elapsed_s": elapsed,
        "speed_mbs": data_mb / elapsed if elapsed > 0 else 0.0,
        "data_mb":   data_mb,
    }



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=False, default=None)
    p.add_argument("--data",     default=DEFAULT_DATA,
                   help=f"Binary test file (default: {DEFAULT_DATA})")
    p.add_argument("--config",   default="experiments/atlas_experiment/config.yaml")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-baselines", action="store_true", help="Skip baseline codecs")
    p.add_argument("--sample-mb",   type=float, default=SAMPLE_MB,
                   help=f"MB of data sample for baselines (default: {SAMPLE_MB})")
    p.add_argument("--vortex-mb",   type=float, default=None,
                   help="MB of data for Vortex eval (default: same as --sample-mb).\n"
                        "Pass a larger value to get a more representative BPD.")
    p.add_argument("--no-compile",  action="store_true",
                   help="Disable torch.compile (use if compile causes issues)")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--full-vortex", action="store_true",
                   help="Evaluate Vortex on the full file; baselines still use --sample-mb")
    p.add_argument("--baselines-only", action="store_true",
                   help="Run baseline codecs only, skip Vortex model")
    p.add_argument("--out-json", default=None, metavar="PATH",
                   help="Save all results to a JSON file (for paper tables)")
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
        print(f"  Running baselines on {args.sample_mb:.0f} MB sample ...")
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
    _base = OptimisedCompressiveTransformer(
        vocab_size=m["vocab_size"],       d_model=m["d_model"],
        n_layers=m["n_layers"],           n_heads=m["n_heads"],
        d_ff=m["d_ff"],                   window=c["window_size"],
        compression_rate=c["compression_rate"],
        dropout=m.get("dropout", 0.1),
        use_tdt=m.get("use_tdt", False),
    ).to(args.device)

    cat_cfg = cfg.get("cat", {})
    if cat_cfg.get("enabled", False):
        chunk_sizes = tuple(cat_cfg.get("chunk_sizes", [128, 256, 512]))
        model = CATWrapper(_base, chunk_sizes=chunk_sizes).to(args.device)
        print(f"  CAT wrapper: chunk_sizes={chunk_sizes}")
    else:
        model = _base

    load_checkpoint(model, args.model, device=args.device)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model : {params:.1f}M parameters")

    if not getattr(args, 'no_compile', False) and hasattr(torch, 'compile'):
        print("  Compiling model (torch.compile reduce-overhead) ...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  Compile OK\n")
        except Exception as e:
            print(f"  Compile skipped ({e})\n")
    else:
        print()

    dl = _make_eval_loader(args.data, c["window_size"], args.batch_size)

    vortex_mb = args.vortex_mb if args.vortex_mb else args.sample_mb
    max_tok   = None if args.full_vortex else int(vortex_mb * 1024 * 1024)
    print(f"  Evaluating Vortex on {'full file' if args.full_vortex else f'{vortex_mb:.0f} MB'} ..."
          f"  (--vortex-mb to change, --full-vortex for entire file)")
    vortex = eval_vortex(model, dl, m["vocab_size"], args.device, amp_dtype, max_tok)

    print()
    if baseline_results:
        print_table(baseline_results, vortex)
    else:
        print(f"  Vortex BPD : {vortex['bpd']:.4f}  ({vortex['ratio_x']:.2f}x)")

    print(f"\n{'='*62}")
    print(f"  Summary")
    print(f"{'='*62}")
    vortex_mb_used = args.sample_mb if not args.full_vortex and not args.vortex_mb else (args.vortex_mb or data_size_gb*1024)
    print(f"  Vortex BPD   : {vortex['bpd']:.4f}  ({vortex['ratio_x']:.2f}x)")
    print(f"  Vortex speed : {vortex.get('speed_mbs', 0):.1f} MB/s  ({vortex['elapsed_s']:.1f}s on {vortex['data_mb']:.0f} MB)")
    print(f"  Baselines on : {args.sample_mb:.0f} MB sample  |  Vortex on: {vortex['data_mb']:.0f} MB")
    print(f"  Data file    : {args.data}  ({data_size_gb:.2f} GB total)")
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

    # ── Save JSON for paper tables ─────────────────────────────────────────
    if args.out_json:
        import json, datetime
        record = {
            "experiment": exp,
            "test_file":  args.data,
            "test_size_gb": data_size_gb,
            "model": args.model,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "vortex": {
                "bpd":       round(vortex["bpd"], 6),
                "ratio_x":   round(vortex["ratio_x"], 4),
                "elapsed_s": round(vortex.get("elapsed_s", 0), 2),
                "speed_mbs": round(vortex.get("speed_mbs", 0), 2),
                "data_mb":   round(vortex.get("data_mb", 0), 1),
            },
            "baselines": [
                {
                    "name":          r["name"].strip(),
                    "bpd":           round(r["bpd"], 6),
                    "ratio_x":       round(r["ratio_x"], 4),
                    "elapsed_s":     round(r.get("elapsed_s", 0), 3),
                    "speed_mbs":     round(r["speed_mbs"], 2),
                    "vs_vortex_pct": round(
                        (r["bpd"] - vortex["bpd"]) / r["bpd"] * 100, 2
                    ),
                }
                for r in baseline_results
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as jf:
            json.dump(record, jf, indent=2)
        print(f"  [json] Results saved -> {args.out_json}")


if __name__ == "__main__":
    main()