# Usage: from vortex.data.dataset import make_loaders
import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

# pin_memory is beneficial for CUDA (PCIe DMA) but can cause shared-memory
# allocation failures on some ROCm configurations.  Disable it on HIP/ROCm.
_ON_ROCM = torch.version.hip is not None


class ChunkShuffleSampler(Sampler):
    """Reads windows in sequential chunks then shuffles within each chunk.

    Benefits over DataLoader(shuffle=True):
    - Sequential I/O: each chunk = contiguous run of windows in the mmap file
      -> OS read-ahead + page-cache hit rate goes from ~0% to ~99%
    - Still provides good shuffle: chunk_size=4096 windows shuffled per chunk
      -> training signal quality is equivalent to global shuffle for large datasets
    """

    def __init__(self, n: int, chunk_size: int = 4096, seed: int = 0):
        self.n          = n
        self.chunk_size = chunk_size
        self.seed       = seed
        self._epoch     = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        rng    = np.random.default_rng(self.seed + self._epoch)
        idx    = np.arange(self.n, dtype=np.int64)
        # shuffle chunk start positions so chunks themselves come in random order
        n_chunks = math.ceil(self.n / self.chunk_size)
        chunk_order = rng.permutation(n_chunks)
        out = []
        for ci in chunk_order:
            lo = ci * self.chunk_size
            hi = min(lo + self.chunk_size, self.n)
            chunk = idx[lo:hi].copy()
            rng.shuffle(chunk)
            out.append(chunk)
        return iter(np.concatenate(out).tolist())


class MemmapWindowDataset(Dataset):
    """Memory-mapped binary dataset. Never loads the full file into RAM.

    If preload=True, the entire file is loaded into a RAM numpy array at init
    (takes ~3-5 sec for 14 GB). All subsequent reads are pure in-RAM — zero
    storage I/O during training. Recommended on machines with >=32 GB free RAM.
    """

    def __init__(self, path: str, window: int = 512, stride: int = 256,
                 preload: bool = False):
        self.path      = path
        self.window    = window
        self.stride    = stride
        mm             = np.memmap(path, dtype=np.uint8, mode="r")
        if preload:
            print(f"  [dataset] preloading {os.path.getsize(path)/1e9:.2f} GB into RAM ...",
                  end=" ", flush=True)
            self.data = np.array(mm)  # copies entire file into RAM
            del mm
            print("done")
        else:
            self.data = mm
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
                 streaming: bool = False, preload: bool = False):
    """Returns (train_dataloader, val_dataloader | None)."""

    train_ds = MemmapWindowDataset(train_path, window, stride, preload=preload)

    print(f"  [dataset] train  : {len(train_ds):,} windows  "
          f"({os.path.getsize(train_path)/1e9:.2f} GB  mmap)")

    sampler = ChunkShuffleSampler(len(train_ds), chunk_size=4096, seed=42)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,          # replaces shuffle=True
        num_workers=num_workers,
        pin_memory=False,         # ROCm: keep False; non_blocking used on .to(device)
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )

    val_dl = None
    if val_path and os.path.exists(val_path):
        val_ds = MemmapWindowDataset(val_path, window, window, preload=preload)
        print(f"  [dataset] val    : {len(val_ds):,} windows  "
              f"({os.path.getsize(val_path)/1e9:.2f} GB  mmap)")
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(4 if num_workers > 0 else None),
        )

    return train_dl, val_dl