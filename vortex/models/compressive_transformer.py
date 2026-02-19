# Usage: from vortex.models.compressive_transformer import CompressiveTransformer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MemoryManager(nn.Module):
    """Compresses old activations 4:1 via strided Conv1d (learned compression)."""

    def __init__(self, d_model: int, rate: int = 4):
        super().__init__()
        self.rate = rate
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=rate, stride=rate)
        self.norm = nn.LayerNorm(d_model)

    def compress(self, acts: torch.Tensor) -> torch.Tensor:
        """acts: (B, T, D) -> (B, T/rate, D)"""
        x = acts.transpose(1, 2)            # (B, D, T)
        c = self.conv(x).transpose(1, 2)    # (B, T/rate, D)
        return self.norm(c)


class CompressiveAttention(nn.Module):
    """
    Multi-head attention with two-tier memory:
      - recent:     last `window` tokens at full resolution
      - compressed: older tokens compressed 4:1

    Effective context = window + rate * c_max  with O(window + c_max) memory.
    """

    def __init__(self, d_model: int, n_heads: int,
                 window: int = 512, rate: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.window  = window
        self.mem_mgr = MemoryManager(d_model, rate)
        self.q       = nn.Linear(d_model, d_model)
        self.k       = nn.Linear(d_model, d_model)
        self.v       = nn.Linear(d_model, d_model)
        self.out     = nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, comp_mem=None):
        B, T, D = x.shape
        ctx = torch.cat([comp_mem, x], dim=1) if comp_mem is not None else x

        Q = self._split(self.q(x))
        K = self._split(self.k(ctx))
        V = self._split(self.v(ctx))

        scale = math.sqrt(self.d_k)
        attn  = torch.softmax(Q @ K.transpose(-2, -1) / scale, dim=-1)
        attn  = self.drop(attn)
        out   = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)

        # Update compressed memory
        new_comp = self.mem_mgr.compress(x)
        if comp_mem is not None:
            new_comp = torch.cat([comp_mem, new_comp], dim=1)
            max_comp = self.window // 2
            if new_comp.size(1) > max_comp:
                new_comp = new_comp[:, -max_comp:]

        return self.out(out), new_comp


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, window, rate, dropout):
        super().__init__()
        self.attn = CompressiveAttention(d_model, n_heads, window, rate, dropout)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, comp_mem=None):
        attn_out, new_comp = self.attn(self.ln1(x), comp_mem)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, new_comp


class CompressiveTransformer(nn.Module):
    """
    Byte-level compressive transformer.
    Memory stays O(window + c_max) regardless of total sequence length.
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 512,
                 n_layers: int = 8, n_heads: int = 8, d_ff: int = 2048,
                 window: int = 512, compression_rate: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.pe     = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, window, compression_rate, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, memories=None):
        if memories is None:
            memories = [None] * len(self.layers)
        h = self.pe(self.embed(x))
        new_mems = []
        for layer, mem in zip(self.layers, memories):
            h, new_mem = layer(h, mem)
            new_mems.append(new_mem)
        return self.head(self.ln_f(h)), new_mems

    @torch.no_grad()
    def get_probs(self, x: torch.Tensor, memories=None):
        logits, new_mems = self(x, memories)
        return torch.softmax(logits, dim=-1), new_mems
