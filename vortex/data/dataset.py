# Usage: from vortex.data.dataset import make_loaders
import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MemmapWindowDataset(Dataset):
    """Memory-mapped binary dataset. Never loads the full file into RAM."""

    def __init__(self, path: str, window: int = 512, stride: int = 256):
        self.path      = path
        self.window    = window
        self.stride    = stride
        self.data      = np.memmap(path, dtype=np.uint8, mode="r")
        n              = len(self.data)
        self.n_windows = max(0, math.ceil((n - window) / stride) + 1) if n >= window else 0

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end   = start + self.window
        chunk = self.data[start:end]
        if len(chunk) < self.window:
            padded = np.zeros(self.window, dtype=np.uint8)
            padded[:len(chunk)] = chunk
            chunk = padded
        return torch.from_numpy(chunk.copy()).long()

    def __del__(self):
        if hasattr(self, "data"):
            del self.data


def make_loaders(train_path: str, val_path: str = None,
                 window: int = 512, stride: int = 256,
                 batch_size: int = 32, num_workers: int = 4,
                 streaming: bool = False):
    """Returns (train_dataloader, val_dataloader | None)."""

    if streaming is False and num_workers > 0:
        pass

    train_ds = MemmapWindowDataset(train_path, window, stride)

    print(f"  [dataset] train  : {len(train_ds):,} windows  "
          f"({os.path.getsize(train_path)/1e9:.2f} GB  mmap)")

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_dl = None
    if val_path and os.path.exists(val_path):
        val_ds = MemmapWindowDataset(val_path, window, window)
        print(f"  [dataset] val    : {len(val_ds):,} windows  "
              f"({os.path.getsize(val_path)/1e9:.2f} GB  mmap)")
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
            persistent_workers=(num_workers > 0),
        )

    return train_dl, val_dl