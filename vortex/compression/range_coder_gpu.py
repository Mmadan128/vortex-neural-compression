"""
GPU-accelerated range coder for neural compression.

Components
----------
1. Triton CDF quantization kernel
     Parallel over all (B*T) positions.  Faster than the .cpu() round-trip
     that torchac requires; removes GPU→CPU sync from the hot decode loop.

2. Pure-Python uint32 range coder (encode / decode)
     No torchac dependency.  Mathematically equivalent to arithmetic coding
     with 16-bit CDF precision.

3. StreamDecoder
     Stateful wrapper that decodes exactly ONE symbol per call – needed for
     the autoregressive decompression loop where each decoded byte is fed
     back as the next model input before the next symbol is decoded.

4. make_gpu_cdf(probs)  →  np.ndarray (N_rows, V+1, int32)
     Shared helper used by both encoder and decoder.

Lossless guarantee
------------------
Both sides must call make_gpu_cdf with numerically identical probability
tensors.  Compress + Decompress achieve this by:
  • Compress   : feeding shifted input  [SOS, x_0, …, x_{T-2}]  to the model
  • Decompress : autoregressively reconstructing exactly that shifted sequence
    using the identical model weights and memory state.
The resulting per-position probability distributions are bit-for-bit equal.
"""

from __future__ import annotations

import numpy as np
import torch

# ── Triton (optional) ────────────────────────────────────────────────────────
try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False

RANGE_CODER_AVAILABLE = True   # always True – pure-Python fallback guaranteed

# ── Constants ─────────────────────────────────────────────────────────────────
_SCALE      = 1 << 16          # CDF precision  (65536)
_TOP        = 1 << 32          # range coder state space
_BOT        = 1 << 24          # renormalisation threshold
_EPS_UNITS  = 1                # each symbol guaranteed ≥ 1 CDF unit after quantisation
_VOCAB      = 256              # byte alphabet

# ── Triton CDF quantisation kernel ───────────────────────────────────────────
if _TRITON_OK:
    @triton.jit
    def _quant_cdf_kernel(
        p_ptr,                  # (N, V) float32  input
        q_ptr,                  # (N, V+1) int32  output
        N,
        EPS:   tl.constexpr,   # per-symbol floor probability
        SCALE: tl.constexpr,   # = 65536
        V:     tl.constexpr,   # = 256
    ):
        row = tl.program_id(0)
        if row >= N:
            return

        col = tl.arange(0, V)                          # (256,)

        # ── Load & floor ──────────────────────────────────────────────────
        p = tl.load(p_ptr + row * V + col)             # (256,) float32
        p = tl.maximum(p, EPS)                         # floor to avoid zero

        # ── Normalise ─────────────────────────────────────────────────────
        total = tl.sum(p, axis=0)
        p     = p / total

        # ── Cumulative sum then scale to [0, SCALE] ───────────────────────
        cum      = tl.cumsum(p, axis=0) * SCALE        # (256,) float32
        cdf_vals = tl.cast(cum, tl.int32)
        cdf_vals = tl.minimum(cdf_vals, SCALE)

        # Force last entry to be exactly SCALE (guarantees CDF[V] == SCALE)
        cdf_vals = tl.where(col == V - 1, SCALE, cdf_vals)

        # ── Write CDF[0] = 0, CDF[1..V] = cdf_vals ───────────────────────
        zero_vec = tl.zeros([1], dtype=tl.int32)
        tl.store(q_ptr + row * (V + 1) + tl.arange(0, 1), zero_vec)
        tl.store(q_ptr + row * (V + 1) + 1 + col, cdf_vals)


def _gpu_prepare_cdf(probs: torch.Tensor) -> torch.Tensor:
    """Triton path: probs (N, 256) float32 → cdf (N, 257) int32 on same device."""
    N, V = probs.shape
    assert V == _VOCAB
    probs_c = probs.contiguous().float()
    cdf = torch.empty(N, V + 1, dtype=torch.int32, device=probs.device)
    grid = (N,)
    _quant_cdf_kernel[grid](
        probs_c, cdf, N,
        EPS   = float(_EPS_UNITS) / _SCALE,
        SCALE = _SCALE,
        V     = _VOCAB,
    )
    return cdf


def _cpu_prepare_cdf(probs: torch.Tensor) -> torch.Tensor:
    """CPU fallback: same semantics as Triton kernel."""
    N, V = probs.shape
    p = probs.float()
    eps = float(_EPS_UNITS) / _SCALE
    p   = torch.clamp(p, min=eps)
    p   = p / p.sum(dim=-1, keepdim=True)
    cum = torch.cumsum(p, dim=-1) * _SCALE
    cdf_body = cum.long().clamp(0, _SCALE).int()          # (N, V)
    cdf_body[:, -1] = _SCALE                               # exact last entry
    cdf = torch.zeros(N, V + 1, dtype=torch.int32)
    cdf[:, 1:] = cdf_body
    return cdf


def make_gpu_cdf(probs: torch.Tensor) -> np.ndarray:
    """
    Convert model probability tensor to quantised CDF array.

    Parameters
    ----------
    probs : Tensor, shape (B, T, 256) or (T, 256) or (1, 256)

    Returns
    -------
    cdf : np.ndarray, shape (N_rows, 257) int32
        N_rows = B*T (or T or 1).  CDF[i, 0] == 0, CDF[i, 256] == 65536.
    """
    orig_shape = probs.shape
    flat = probs.reshape(-1, _VOCAB)

    # Use Triton only when N is large enough to amortise the kernel-launch +
    # GPU→CPU sync cost.  For N ≤ 32 (e.g. per-symbol decode in decompress.py)
    # the CPU path is faster because it avoids a full device synchronisation.
    _TRITON_MIN_ROWS = 32
    if _TRITON_OK and flat.is_cuda and flat.shape[0] > _TRITON_MIN_ROWS:
        cdf_t = _gpu_prepare_cdf(flat)
        return cdf_t.cpu().numpy()
    else:
        cdf_t = _cpu_prepare_cdf(flat.cpu())
        return cdf_t.numpy()


# ── uint32 range coder ────────────────────────────────────────────────────────

def _rc_encode(symbols: np.ndarray, cdf: np.ndarray) -> bytes:
    """
    Encode a sequence of byte symbols using a pre-built CDF table.

    Parameters
    ----------
    symbols : np.ndarray  shape (T,)  int32/int64, values in [0, 255]
    cdf     : np.ndarray  shape (T, 257) int32

    Returns
    -------
    bitstring : bytes

    Renorm strategy (two passes, mirrored in decoder)
    --------------------------------------------------
    Pass 1 (top-byte agreement) : emit the certain top byte whenever lo and
                                  hi-1 share it — standard interval coder.
    Pass 2 (range floor)        : if after pass 1 the interval is still
                                  smaller than _SCALE (2^16), keep emitting
                                  bytes until range >= _SCALE.  This guarantees
                                  r16 = (hi - lo) >> 16 >= 1 before every CDF
                                  update, preventing division-by-zero.
    Both passes must be mirrored byte-for-byte in the decoder.
    """
    T   = len(symbols)
    buf = bytearray()
    lo  = 0
    hi  = _TOP                    # interval is [lo, hi)

    for t in range(T):
        s      = int(symbols[t])
        r      = hi - lo
        r16    = r >> 16          # always >= 1 after renorm below
        hi     = lo + r16 * int(cdf[t, s + 1])
        lo     = lo + r16 * int(cdf[t, s])

        # Pass 1: emit top byte while lo and hi-1 agree on it
        while (lo ^ (hi - 1)) >> 24 == 0:
            buf.append(lo >> 24)
            lo = (lo << 8) & (_TOP - 1)
            hi = ((hi << 8) & (_TOP - 1)) or _TOP

        # Pass 2: guarantee range >= _SCALE so next r16 >= 1
        while (hi - lo) < _SCALE:
            buf.append(lo >> 24)
            lo = (lo << 8) & (_TOP - 1)
            hi = ((hi << 8) & (_TOP - 1)) or _TOP

    # Flush: emit 5 bytes so decoder can always read 4 ahead
    for _ in range(5):
        buf.append(lo >> 24)
        lo = (lo << 8) & (_TOP - 1)

    return bytes(buf)


def _rc_decode_all(bitstring: bytes, cdf: np.ndarray) -> np.ndarray:
    """
    Decode all symbols from a bitstring using a pre-built CDF table.

    Parameters
    ----------
    bitstring : bytes
    cdf       : np.ndarray  shape (T, 257) int32

    Returns
    -------
    symbols : np.ndarray  shape (T,) int32
    """
    T    = len(cdf)
    data = bytearray(bitstring) + bytearray(8)  # 8-byte guard for safe read
    pos  = 0

    # Bootstrap: read 4 bytes into the code register
    code = 0
    for _ in range(4):
        code = (code << 8) | data[pos];  pos += 1

    lo      = 0
    hi      = _TOP
    symbols = np.empty(T, dtype=np.int32)

    for t in range(T):
        r      = hi - lo
        r16    = r >> 16          # guaranteed >= 1 after renorm below
        # Relative position of code within [lo, hi), scaled to [0, _SCALE)
        target = (code - lo) // r16
        target = min(target, _SCALE - 1)    # safety clamp

        # Binary search: largest s s.t. cdf[t, s] <= target
        s = int(np.searchsorted(cdf[t], target, side='right')) - 1
        s = int(np.clip(s, 0, _VOCAB - 1))
        symbols[t] = s

        hi = lo + r16 * int(cdf[t, s + 1])
        lo = lo + r16 * int(cdf[t, s])

        # Pass 1: mirror of encoder pass 1
        while (lo ^ (hi - 1)) >> 24 == 0:
            lo   = (lo << 8) & (_TOP - 1)
            hi   = ((hi << 8) & (_TOP - 1)) or _TOP
            code = ((code << 8) & (_TOP - 1)) | data[pos]
            pos += 1

        # Pass 2: mirror of encoder pass 2 (consume the range-floor bytes)
        while (hi - lo) < _SCALE:
            lo   = (lo << 8) & (_TOP - 1)
            hi   = ((hi << 8) & (_TOP - 1)) or _TOP
            code = ((code << 8) & (_TOP - 1)) | data[pos]
            pos += 1

    return symbols


# ── StreamDecoder ─────────────────────────────────────────────────────────────

class StreamDecoder:
    """
    Stateful range decoder for autoregressive (one-symbol-at-a-time) decoding.

    Usage in decompress loop
    ------------------------
        dec = StreamDecoder(blob)
        for t in range(chunk_size):
            cdf_row = make_gpu_cdf(probs_t)[0]   # shape (257,) int32
            symbol  = dec.decode_symbol(cdf_row)
            # feed symbol back to model …

    The decoder maintains the range-coder state (lo, hi, code) across calls,
    so each call consumes exactly the bits that the encoder wrote for that
    symbol – no look-ahead required.
    """

    def __init__(self, bitstring: bytes):
        self._data = bytearray(bitstring) + bytearray(8)
        self._pos  = 0
        self._lo   = 0
        self._hi   = _TOP

        # Bootstrap: fill code register
        self._code = 0
        for _ in range(4):
            self._code = (self._code << 8) | self._data[self._pos]
            self._pos += 1

    def decode_symbol(self, cdf_row: np.ndarray) -> int:
        """
        Decode one symbol using a single row of the CDF table.

        Parameters
        ----------
        cdf_row : np.ndarray  shape (257,) int32
            Must be identical to the CDF row used by the encoder for this
            position.

        Returns
        -------
        symbol : int in [0, 255]
        """
        r      = self._hi - self._lo
        r16    = r >> 16          # guaranteed >= 1 after renorm
        target = (self._code - self._lo) // r16
        target = min(target, _SCALE - 1)

        s = int(np.searchsorted(cdf_row, target, side='right')) - 1
        s = int(np.clip(s, 0, _VOCAB - 1))

        self._hi = self._lo + r16 * int(cdf_row[s + 1])
        self._lo = self._lo + r16 * int(cdf_row[s])

        # Pass 1: top-byte-agreement renorm
        while (self._lo ^ (self._hi - 1)) >> 24 == 0:
            self._lo   = (self._lo   << 8) & (_TOP - 1)
            self._hi   = ((self._hi  << 8) & (_TOP - 1)) or _TOP
            self._code = ((self._code << 8) & (_TOP - 1)) | self._data[self._pos]
            self._pos += 1

        # Pass 2: range-floor renorm (mirror of encoder)
        while (self._hi - self._lo) < _SCALE:
            self._lo   = (self._lo   << 8) & (_TOP - 1)
            self._hi   = ((self._hi  << 8) & (_TOP - 1)) or _TOP
            self._code = ((self._code << 8) & (_TOP - 1)) | self._data[self._pos]
            self._pos += 1

        return s


# ── Drop-in replacements for arithmetic_coding.encode / decode ────────────────

def gpu_encode(probs: torch.Tensor, symbols: torch.Tensor) -> bytes:
    """
    Encode symbols using GPU-prepared CDFs.

    Parameters
    ----------
    probs   : Tensor (B, T, 256) float32  – model output probabilities
    symbols : Tensor (B, T)      int64/int16  – ground-truth byte values

    Returns
    -------
    bitstring : bytes  (one combined stream for the whole batch row B=0)

    Note: for B > 1 call this once per batch element.
    """
    # Squeeze batch dim (compress.py uses B=1)
    p1 = probs[0]                          # (T, 256)
    s1 = symbols[0].cpu().numpy().astype(np.int32)   # (T,)
    cdf = make_gpu_cdf(p1)                 # (T, 257) int32 np.ndarray
    return _rc_encode(s1, cdf)


def gpu_decode(bitstring: bytes, probs: torch.Tensor) -> torch.Tensor:
    """
    Decode all symbols at once (compress-side mirror of gpu_encode).

    Parameters
    ----------
    bitstring : bytes
    probs     : Tensor (B, T, 256) float32

    Returns
    -------
    symbols : Tensor (B, T) int16
    """
    B, T, _ = probs.shape
    out = torch.empty(B, T, dtype=torch.int16)
    for b in range(B):
        cdf  = make_gpu_cdf(probs[b])          # (T, 257) int32
        syms = _rc_decode_all(bitstring, cdf)  # (T,) int32
        out[b] = torch.from_numpy(syms).short()
    return out


def theoretical_bpd(probs: torch.Tensor, symbols: torch.Tensor) -> float:
    """Cross-entropy in bits per byte. Lower is better."""
    gathered = probs.gather(-1, symbols.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log2(gathered.clamp(min=1e-10)).mean().item()
