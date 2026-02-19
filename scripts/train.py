#!/usr/bin/env python3
# Usage: python scripts/train.py --config experiments/atlas_experiment/config.yaml
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, math, random, yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
from vortex.data.dataset import make_loaders
from vortex.utils.training import cosine_with_warmup, set_lr, save_checkpoint, load_checkpoint, EarlyStopping


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16   = cfg["training"].get("mixed_precision", True) and device == "cuda"
    exp    = cfg.get("experiment", {}).get("name", "experiment")
    print(f"Experiment : {exp}")
    print(f"Device     : {device}  |  FP16: {fp16}")

    m, c, t = cfg["model"], cfg["compressive_memory"], cfg["training"]
    t["learning_rate"] = float(t["learning_rate"])
    model = OptimisedCompressiveTransformer(
        vocab_size=m["vocab_size"], d_model=m["d_model"],
        n_layers=m["n_layers"],    n_heads=m["n_heads"],
        d_ff=m["d_ff"],            window=c["window_size"],
        compression_rate=c["compression_rate"], dropout=m["dropout"],
    ).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters : {params:.1f}M")

    d, p_cfg = cfg["data"], cfg["paths"]

    if not os.path.exists(p_cfg["train_data"]):
        print(f"\n[ERROR] Training data not found: {p_cfg['train_data']}")
        print("  Run: python experiments/atlas_experiment/download.py --all-steps")
        print("  Then: python experiments/atlas_experiment/prepare.py\n")
        sys.exit(1)

    train_dl, val_dl = make_loaders(
        p_cfg["train_data"], p_cfg.get("val_data"),
        window=d["window_size"], stride=d["stride"],
        batch_size=t["batch_size"], num_workers=d["num_workers"],
        streaming=d.get("streaming", False),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["learning_rate"],
        weight_decay=t["weight_decay"], betas=(0.9, 0.999),
    )
    scaler    = torch.cuda.amp.GradScaler(enabled=fp16)
    stopper   = EarlyStopping(patience=5)
    writer    = SummaryWriter(p_cfg["log_dir"])
    criterion = nn.CrossEntropyLoss()

    step, best_bpd = 0, float("inf")
    if args.resume:
        _, best_bpd = load_checkpoint(model, args.resume, optimizer, device)

    epoch = 0
    while step < t["max_steps"]:
        epoch += 1
        model.train()
        memories = None
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = batch.to(device)
            lr = cosine_with_warmup(step, t["warmup_steps"], t["max_steps"],
                                    max_lr=t["learning_rate"])
            set_lr(optimizer, lr)
            with torch.cuda.amp.autocast(enabled=fp16):
                logits, memories, _ = model(batch, memories)
                loss = criterion(logits[:, :-1].reshape(-1, m["vocab_size"]),
                                 batch[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            memories = [mm.detach() if mm is not None else None for mm in memories]
            bpd = loss.item() / math.log(2)
            pbar.set_postfix(bpd=f"{bpd:.4f}", lr=f"{lr:.2e}", step=step)
            writer.add_scalar("train/bpd", bpd, step)
            step += 1
            if step >= t["max_steps"]:
                break

        if val_dl:
            model.eval()
            total, count, mems = 0.0, 0, None
            with torch.no_grad():
                for batch in val_dl:
                    batch = batch.to(device)
                    logits, mems, _ = model(batch, mems)
                    loss = criterion(logits[:, :-1].reshape(-1, m["vocab_size"]),
                                     batch[:, 1:].reshape(-1))
                    total += loss.item(); count += 1
                    mems = [mm.detach() if mm is not None else None for mm in mems]
            val_bpd = total / count / math.log(2)
            writer.add_scalar("val/bpd", val_bpd, step)
            print(f"\n  Val BPD: {val_bpd:.4f}")
            if val_bpd < best_bpd:
                best_bpd = val_bpd
                save_checkpoint(model, optimizer, epoch, best_bpd,
                                os.path.join(p_cfg["checkpoint_dir"], "best.pt"))
            if stopper.step(val_bpd):
                print("Early stopping.")
                break

    writer.close()
    print(f"\nDone. Best Val BPD: {best_bpd:.4f}")


if __name__ == "__main__":
    main()
