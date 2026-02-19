# Usage: from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
    print("[vortex] Flash Attention 2 available")
except ImportError:
    FLASH_AVAILABLE = False

from .compressive_transformer import PositionalEncoding, MemoryManager


class SwiGLU(nn.Module):
    """SwiGLU gated feed-forward (Shazeer 2020)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff,   d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class OptimisedCompressiveAttention(nn.Module):
    """Flash Attention 2 when available; otherwise PyTorch fused SDPA.
    Supports KV cache for O(1) per-step cost during decoding."""

    def __init__(self, d_model: int, n_heads: int,
                 window: int = 512, rate: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.window  = window
        self.dropout = dropout
        self.mem_mgr = MemoryManager(d_model, rate)
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out     = nn.Linear(d_model, d_model,     bias=False)
        self.drop    = nn.Dropout(dropout)

    def _project(self, x: torch.Tensor):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return (t.view(B, T, self.n_heads, self.d_k) for t in (q, k, v))

    def forward(self, x: torch.Tensor, comp_mem=None, kv_cache=None):
        B, T, D = x.shape
        Q, K, V = self._project(x)

        # KV Cache accumulation (for autoregressive decode)
        if kv_cache is not None:
            K = torch.cat([kv_cache["k"], K], dim=1)
            V = torch.cat([kv_cache["v"], V], dim=1)
        new_cache = {"k": K.detach(), "v": V.detach()}

        # Prepend compressed memory as extra K/V context
        if comp_mem is not None:
            _, Km, Vm = self._project(comp_mem)
            K = torch.cat([Km, K], dim=1)
            V = torch.cat([Vm, V], dim=1)

        # Attention (Flash or Fused SDPA)
        if FLASH_AVAILABLE and x.is_cuda:
            out = flash_attn_func(
                Q.half(), K.half(), V.half(),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            ).to(x.dtype)
        else:
            out = F.scaled_dot_product_attention(
                Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            ).transpose(1, 2)

        out = out.contiguous().view(B, T, D)

        # Update compressed memory
        new_comp = self.mem_mgr.compress(x)
        if comp_mem is not None:
            new_comp = torch.cat([comp_mem, new_comp], dim=1)
            max_c = self.window // 2
            if new_comp.size(1) > max_c:
                new_comp = new_comp[:, -max_c:]

        return self.out(out), new_comp, new_cache


class OptimisedBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, window, rate, dropout):
        super().__init__()
        self.attn = OptimisedCompressiveAttention(d_model, n_heads, window, rate, dropout)
        self.ff   = SwiGLU(d_model, d_ff)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x, comp_mem=None, kv_cache=None):
        attn_out, new_comp, new_cache = self.attn(self.ln1(x), comp_mem, kv_cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_comp, new_cache


class OptimisedCompressiveTransformer(nn.Module):
    """Drop-in replacement for CompressiveTransformer with:
      Flash Attention 2 | KV Cache | SwiGLU | Pre-LayerNorm"""

    def __init__(self, vocab_size=256, d_model=512, n_layers=8, n_heads=8,
                 d_ff=2048, window=512, compression_rate=4, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.pe     = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            OptimisedBlock(d_model, n_heads, d_ff, window, compression_rate, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, memories=None, kv_caches=None):
        if memories  is None: memories  = [None] * len(self.layers)
        if kv_caches is None: kv_caches = [None] * len(self.layers)
        h = self.pe(self.embed(x))
        new_mems, new_caches = [], []
        for layer, mem, cache in zip(self.layers, memories, kv_caches):
            h, new_mem, new_cache = layer(h, mem, cache)
            new_mems.append(new_mem)
            new_caches.append(new_cache)
        return self.head(self.ln_f(h)), new_mems, new_caches

    @torch.no_grad()
    def get_probs(self, x, memories=None, kv_caches=None):
        logits, mems, caches = self(x, memories, kv_caches)
        return torch.softmax(logits, dim=-1), mems, caches
