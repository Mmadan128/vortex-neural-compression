# Usage: python scripts/train.py --config experiments/atlas_experiment/config.yaml
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, math, random, time, yaml
import torch, torch.nn as nn
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vortex.models.optimized_transformer import (
    OptimisedCompressiveTransformer,
    CATWrapper,
)
from vortex.data.dataset import make_loaders
from vortex.utils.training import (cosine_with_warmup, set_lr,
                                   save_checkpoint, load_checkpoint,
                                   EarlyStopping, get_amp_dtype)

try:
    import gzip as _gz, zlib as _zl, lzma as _lz
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False

try:
    import psutil
    PSUTIL = True
except ImportError:
    PSUTIL = False


def quick_baselines(data: bytes) -> dict:
    """Fast baselines on a small sample for live display."""
    sample = data[:min(len(data), 2 * 1024 * 1024)]
    n = len(sample)
    results = {}
    for name, fn in [
        ("gzip-6",  lambda d: _gz.compress(d, compresslevel=6)),
        ("zlib-9",  lambda d: _zl.compress(d, level=9)),
        ("lzma-6",  lambda d: _lz.compress(d, preset=6)),
    ]:
        try:
            c = fn(sample)
            results[name] = len(c) * 8 / n
        except Exception:
            pass
    return results


def print_scoreboard(step: int, train_bpd: float, val_bpd: float,
                     best_bpd: float, baselines: dict,
                     lr: float, elapsed_h: float, eta_h: float,
                     recent_bpds: list):
    """Live ASCII scoreboard printed during training."""
    bar_width = 30
    if len(recent_bpds) >= 2:
        trend = "▼" if recent_bpds[-1] < recent_bpds[0] else "▲"
    else:
        trend = "~"

    lines = [
        f"\n{'─'*54}",
        f"  Vortex-Codec  │  Step {step:>7,}  │  LR {lr:.2e}",
        f"{'─'*54}",
        f"  Train BPD : {train_bpd:7.4f}  {trend}",
        f"  Val   BPD : {val_bpd:7.4f}  {'★ best' if abs(val_bpd - best_bpd) < 1e-6 else ''}",
        f"  Best  BPD : {best_bpd:7.4f}",
        f"{'─'*54}",
    ]
    if baselines:
        lines.append(f"  Baselines (2MB sample):")
        for name, bpd in baselines.items():
            delta = (bpd - val_bpd) / bpd * 100
            bar_filled = int(bar_width * min(1.0, val_bpd / bpd))
            bar = "█" * bar_filled + "░" * (bar_width - bar_filled)
            lines.append(f"    {name:10s}: {bpd:.4f}  Vortex {delta:+.1f}%")
    lines += [
        f"{'─'*54}",
        f"  Elapsed : {elapsed_h:.1f}h  │  ETA : {eta_h:.1f}h",
        f"{'─'*54}\n",
    ]
    print("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/atlas_experiment/config.yaml")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"].get("seed", 42))
    random.seed(cfg["training"].get("seed", 42))

    try:
        torch.backends.cuda.matmul.fp32_precision  = "tf32"
        torch.backends.cudnn.conv.fp32_precision   = "tf32"
    except AttributeError:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark = True

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    dev_type = device.split(":")[0]  # "cuda:0" -> "cuda"

    amp_dtype = get_amp_dtype(device)
    if device == "cuda" and torch.version.hip is not None:
        os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",
                              "max_split_size_mb:512,garbage_collection_threshold:0.8")
        print(f"  [ROCm] {torch.cuda.get_device_name(0)}  amp={amp_dtype}")

    fp16 = cfg["training"].get("mixed_precision", True) and device == "cuda"
    grad_accum = cfg["training"].get("grad_accumulation_steps", 1)
    exp        = cfg.get("experiment", {}).get("name", "experiment")

    print(f"\n{'='*54}")
    print(f"  Experiment       : {exp}")
    print(f"  Device           : {device}  |  FP16: {fp16}")
    print(f"  Grad accumulation: {grad_accum}  "
          f"(effective batch = {cfg['training']['batch_size'] * grad_accum})")
    print(f"{'='*54}\n")

    m, c, t = cfg["model"], cfg["compressive_memory"], cfg["training"]
    t["learning_rate"] = float(t["learning_rate"])

    model = OptimisedCompressiveTransformer(
        vocab_size=m["vocab_size"], d_model=m["d_model"],
        n_layers=m["n_layers"],    n_heads=m["n_heads"],
        d_ff=m["d_ff"],            window=c["window_size"],
        compression_rate=c["compression_rate"], dropout=m["dropout"],
        use_tdt=m.get("use_tdt", False),
    ).to(device)

    cat_cfg = cfg.get("cat", {})
    if cat_cfg.get("enabled", False):
        chunk_sizes = tuple(cat_cfg.get("chunk_sizes", [128, 256, 512]))
        model = CATWrapper(model, chunk_sizes=chunk_sizes)
        print(f"  CAT wrapper      : chunk_sizes={chunk_sizes}")

    if cfg["training"].get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
    if cfg["training"].get("compile_model", False):
        print("Compiling model...")
        model = torch.compile(model)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters       : {params:.1f}M\n")

    if device == "cuda":
        print("  [bench] Warming up — 5 forward passes...")
        _dummy = torch.randint(0, 256, (cfg["training"]["batch_size"], cfg["data"]["window_size"]),
                               device=device)
        _mems = None
        with torch.no_grad(), torch.amp.autocast(dev_type, enabled=fp16, dtype=amp_dtype):
            for _ in range(2):
                _, _mems, _ = model(_dummy, _mems)
                _mems = [m.detach() if m is not None else None for m in _mems]
        torch.cuda.synchronize()
        _t0 = time.time()
        _mems = None
        with torch.no_grad(), torch.amp.autocast(dev_type, enabled=fp16, dtype=amp_dtype):
            for _ in range(5):
                _, _mems, _ = model(_dummy, _mems)
                _mems = [m.detach() if m is not None else None for m in _mems]
        torch.cuda.synchronize()
        _elapsed = time.time() - _t0
        _tok_per_sec = (5 * cfg["training"]["batch_size"] * cfg["data"]["window_size"]) / _elapsed
        print(f"  [bench] {_tok_per_sec:,.0f} tokens/sec  "
              f"({_elapsed/5*1000:.1f} ms/step forward-only)\n")
        del _dummy, _mems

    d, p_cfg = cfg["data"], cfg["paths"]
    if not os.path.exists(p_cfg["train_data"]):
        print(f"\n[ERROR] Training data not found: {p_cfg['train_data']}")
        print("  Run: python experiments/atlas_experiment/download.py --all-steps\n")
        sys.exit(1)

    train_bytes = os.path.getsize(p_cfg["train_data"])
    if PSUTIL:
        avail_gb  = psutil.virtual_memory().available / 1e9
        train_gb  = train_bytes / 1e9
        streaming = d.get("streaming", False)
        print(f"  RAM available    : {avail_gb:.1f} GB")
        print(f"  Train file size  : {train_gb:.1f} GB")
        print(f"  Streaming        : {streaming}")
        if not streaming and train_gb > avail_gb * 0.5:
            print(f"\n  [WARNING] Train file ({train_gb:.1f} GB) is large relative to "
                  f"available RAM ({avail_gb:.1f} GB).")
            print(f"  [WARNING] Forcing streaming=True to avoid OOM kill.\n")
            d["streaming"] = True
        print()

    train_dl, val_dl = make_loaders(
        p_cfg["train_data"], p_cfg.get("val_data"),
        window=d["window_size"], stride=d["stride"],
        batch_size=t["batch_size"], num_workers=d["num_workers"],
        streaming=d.get("streaming", False),
        preload=d.get("preload", False),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["learning_rate"],
        weight_decay=t["weight_decay"], betas=(0.9, 0.999),
    )
    scaler    = torch.amp.GradScaler(dev_type, enabled=(fp16 and amp_dtype == torch.float16))
    stopper   = EarlyStopping(patience=8)
    writer    = SummaryWriter(p_cfg["log_dir"])
    criterion = nn.CrossEntropyLoss()

    step, best_bpd = 0, float("inf")

    # Auto-resume from latest.pt if it exists and --resume not explicitly given
    ckpt_dir = p_cfg.get("checkpoint_dir", "")
    _auto_latest = os.path.join(ckpt_dir, "latest.pt") if ckpt_dir else ""
    if not args.resume and os.path.exists(_auto_latest):
        args.resume = _auto_latest
        print(f"  [auto-resume] Found {_auto_latest}")

    if args.resume:
        step, best_bpd = load_checkpoint(model, args.resume, optimizer, device)
        print(f"  Resumed from {args.resume}")
        print(f"  Restored step={step:,}  best_bpd={best_bpd:.4f}")

    baselines = {}
    if BASELINES_AVAILABLE and val_dl and p_cfg.get("val_data"):
        try:
            val_sample = open(p_cfg["val_data"], "rb").read(2 * 1024 * 1024)
            baselines = quick_baselines(val_sample)
            print("  Baseline BPDs (2MB sample):")
            for k, v in baselines.items():
                print(f"    {k}: {v:.4f}")
            print()
        except Exception as e:
            print(f"  [baseline] skipped: {e}")

    eval_interval = cfg.get("evaluation", {}).get("eval_interval", 5000)
    recent_bpds   = deque(maxlen=20)
    train_bpd_ema = None
    t0            = time.time()
    epoch         = 0

    memories = None
    while step < t["max_steps"]:
        epoch += 1
        # Advance ChunkShuffleSampler so each epoch uses a different chunk order
        if hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(epoch)
        model.train()
        steps_left = t["max_steps"] - step
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}",
                    total=min(len(train_dl), steps_left),
                    dynamic_ncols=True, leave=True)
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar):
            batch = batch.to(device, non_blocking=True)  # async H2D; GPU pipeline stays full
            lr    = cosine_with_warmup(step, t["warmup_steps"],
                                       t["max_steps"], max_lr=t["learning_rate"])
            set_lr(optimizer, lr)

            with torch.amp.autocast(dev_type, enabled=fp16, dtype=amp_dtype):
                logits, memories, _ = model(batch, memories)
                loss = criterion(
                    logits[:, :-1].reshape(-1, m["vocab_size"]),
                    batch[:, 1:].reshape(-1),
                ) / grad_accum

            scaler.scale(loss).backward()
            memories = [mm.detach() if mm is not None else None
                        for mm in memories]

            if (i + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                raw_bpd = loss.item() * grad_accum / math.log(2)
                alpha = 0.02
                train_bpd_ema = (raw_bpd if train_bpd_ema is None
                                 else (1 - alpha) * train_bpd_ema + alpha * raw_bpd)
                recent_bpds.append(raw_bpd)

                writer.add_scalar("train/bpd", raw_bpd, step)
                writer.add_scalar("train/bpd_ema", train_bpd_ema, step)
                writer.add_scalar("train/lr", lr, step)

                # Only update tqdm and tensorboard every 50 steps
                # — eliminates ~300ms/step of file I/O overhead from tb writes
                if step % 50 == 0:
                    writer.flush()

                pbar.set_postfix(
                    bpd=f"{raw_bpd:.4f}",
                    ema=f"{train_bpd_ema:.4f}",
                    lr=f"{lr:.2e}",
                    step=step,
                    ordered=True,
                )
                step += 1

            if step >= t["max_steps"]:
                break

            if step > 0 and step % eval_interval == 0 and val_dl:
                model.eval()
                val_nats, val_tokens, val_mems = 0.0, 0, None
                eval_batches = cfg.get("evaluation", {}).get("eval_batches", 50)
                with torch.no_grad():
                    for _eval_i, vbatch in enumerate(val_dl):
                        if _eval_i >= eval_batches:
                            break
                        vbatch = vbatch.to(device)
                        with torch.amp.autocast(dev_type, enabled=fp16, dtype=amp_dtype):
                            vlogits, val_mems, _ = model(vbatch, val_mems)
                            vloss = criterion(
                                vlogits[:, :-1].reshape(-1, m["vocab_size"]),
                                vbatch[:, 1:].reshape(-1),
                            )
                        n_tok = vbatch.size(0) * (vbatch.size(1) - 1)
                        val_nats   += vloss.item() * n_tok
                        val_tokens += n_tok
                        val_mems = [mm.detach() if mm is not None else None
                                    for mm in val_mems]
                val_bpd = val_nats / val_tokens / math.log(2)
                writer.add_scalar("val/bpd", val_bpd, step)

                elapsed_h = (time.time() - t0) / 3600
                eta_h     = elapsed_h / max(step, 1) * (t["max_steps"] - step)

                print_scoreboard(
                    step        = step,
                    train_bpd   = train_bpd_ema or raw_bpd,
                    val_bpd     = val_bpd,
                    best_bpd    = best_bpd,
                    baselines   = baselines,
                    lr          = lr,
                    elapsed_h   = elapsed_h,
                    eta_h       = eta_h,
                    recent_bpds = list(recent_bpds),
                )

                if val_bpd < best_bpd:
                    best_bpd = val_bpd
                    os.makedirs(p_cfg["checkpoint_dir"], exist_ok=True)
                    save_checkpoint(
                        model, optimizer, step, best_bpd,
                        os.path.join(p_cfg["checkpoint_dir"], "best.pt"),
                    )
                    print(f"  ★ New best checkpoint saved: {best_bpd:.4f} BPD\n")

                ckpt_name = f"step_{step:07d}.pt"
                save_checkpoint(
                    model, optimizer, step, val_bpd,
                    os.path.join(p_cfg["checkpoint_dir"], ckpt_name),
                )

                if stopper.step(val_bpd):
                    print("Early stopping triggered.")
                    writer.close()
                    return

                model.train()

        if val_dl and step % eval_interval != 0:
            model.eval()
            val_nats, val_tokens, val_mems = 0.0, 0, None
            with torch.no_grad():
                for vbatch in val_dl:
                    vbatch = vbatch.to(device)
                    with torch.amp.autocast(dev_type, enabled=fp16, dtype=amp_dtype):
                        vlogits, val_mems, _ = model(vbatch, val_mems)
                        vloss = criterion(
                            vlogits[:, :-1].reshape(-1, m["vocab_size"]),
                            vbatch[:, 1:].reshape(-1),
                        )
                    n_tok = vbatch.size(0) * (vbatch.size(1) - 1)
                    val_nats   += vloss.item() * n_tok
                    val_tokens += n_tok
                    val_mems = [mm.detach() if mm is not None else None
                                for mm in val_mems]
            val_bpd = val_nats / val_tokens / math.log(2)
            writer.add_scalar("val/bpd", val_bpd, step)

            elapsed_h = (time.time() - t0) / 3600
            eta_h     = elapsed_h / max(step, 1) * (t["max_steps"] - step)
            print_scoreboard(
                step=step, train_bpd=train_bpd_ema or 0,
                val_bpd=val_bpd, best_bpd=best_bpd,
                baselines=baselines, lr=lr,
                elapsed_h=elapsed_h, eta_h=eta_h,
                recent_bpds=list(recent_bpds),
            )

            if val_bpd < best_bpd:
                best_bpd = val_bpd
                os.makedirs(p_cfg["checkpoint_dir"], exist_ok=True)
                save_checkpoint(
                    model, optimizer, step, best_bpd,
                    os.path.join(p_cfg["checkpoint_dir"], "best.pt"),
                )
                print(f"  ★ New best: {best_bpd:.4f} BPD\n")

            if stopper.step(val_bpd):
                print("Early stopping triggered.")
                break

    writer.close()
    print(f"\n{'='*54}")
    print(f"  Training complete.  Best Val BPD: {best_bpd:.4f}")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()