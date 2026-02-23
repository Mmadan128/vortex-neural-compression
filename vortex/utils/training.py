# Usage: from vortex.utils.training import cosine_with_warmup, save_checkpoint, get_amp_dtype
import os, math
import torch


def cosine_with_warmup(step, warmup, total, min_lr=1e-6, max_lr=3e-4):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


def save_checkpoint(model, optimizer, epoch, bpd, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "bpd": bpd}, path)
    print(f"  [ckpt] Saved -> {path}  (BPD={bpd:.4f})")


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("bpd", float("inf"))


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    def step(self, val_bpd) -> bool:
        if val_bpd < self.best - self.min_delta:
            self.best    = val_bpd
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def get_amp_dtype(device: str) -> "torch.dtype":
    if device == "cuda" and torch.version.hip is not None:
        try:
            if tuple(int(x) for x in torch.version.hip.split(".")[:2]) >= (5, 7):
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16
