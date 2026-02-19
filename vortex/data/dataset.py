# Usage: from vortex.data.dataset import make_loaders
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BinaryWindowDataset(Dataset):
    """Loads entire file into RAM; creates overlapping windows."""

    def __init__(self, path: str, window: int = 512, stride: int = 256):
        data = np.fromfile(path, dtype=np.uint8)
        self.windows = []
        for start in range(0, len(data) - window + 1, stride):
            self.windows.append(data[start: start + window].copy())
        # Last partial window (zero-padded)
        if len(data) > 0 and len(data) % window != 0:
            tail = data[len(data) - (len(data) % window):]
            if len(tail) < window:
                padded = np.zeros(window, dtype=np.uint8)
                padded[: len(tail)] = tail
                self.windows.append(padded)

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx): return torch.from_numpy(self.windows[idx]).long()


class StreamingBinaryDataset(Dataset):
    """Reads windows from disk on demand — memory-efficient for large files."""

    def __init__(self, path: str, window: int = 512, stride: int = 256):
        self.path    = path
        self.window  = window
        size         = os.path.getsize(path)
        self.offsets = list(range(0, size - window + 1, stride))

    def __len__(self): return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            raw = f.read(self.window)
        arr = np.frombuffer(raw, dtype=np.uint8)
        if len(arr) < self.window:
            arr = np.pad(arr, (0, self.window - len(arr)))
        return torch.from_numpy(arr.copy()).long()


def make_loaders(train_path: str, val_path: str = None,
                 window: int = 512, stride: int = 256,
                 batch_size: int = 32, num_workers: int = 4,
                 streaming: bool = False):
    DS       = StreamingBinaryDataset if streaming else BinaryWindowDataset
    train_ds = DS(train_path, window, stride)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = None
    if val_path and os.path.exists(val_path):
        val_ds = DS(val_path, window, window)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl
