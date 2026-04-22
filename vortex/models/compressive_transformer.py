# Usage: from vortex.models.compressive_transformer import CompressiveTransformer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        end = position_offset + x.size(1)
        return x + self.pe[:, position_offset:end]


class TDTEmbedding(nn.Module):
    """Per-type embedding for IEEE-754 float32 byte streams.

    Each of the 4 byte positions within a float32 gets its own lookup table,
    since they have very different entropy profiles.
    """

    def __init__(self, d_model: int, vocab_size: int = 256):
        super().__init__()
        self.d_model = d_model
        self.embeds = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(4)
        ])
        self.type_scale = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """x : (B, T) -> (B, T, d_model)

        Vectorized single-lookup: eliminates the 4×(bool-mask + scatter) graph
        breaks so torch.compile can capture the whole model as one CUDA graph.

        Each embedding table i is placed at rows [i*256 … (i+1)*256) of a
        combined [4*256, d_model] weight matrix built by torch.cat — the
        individual nn.Embedding parameters are untouched, so checkpoint keys
        stay identical.  No data is duplicated at init; the cat happens once
        per forward (fused by the compiler into a no-copy view on most paths).
        """
        B, T = x.shape
        pos_type  = (torch.arange(T, device=x.device) + position_offset) % 4  # [T]
        # Combined lookup: row = byte_value + table_index * 256
        combined  = torch.cat([emb.weight for emb in self.embeds], dim=0)  # [1024, D]
        x_shifted = x + pos_type.unsqueeze(0).mul(256)               # [B, T]
        out       = F.embedding(x_shifted, combined)                  # [B, T, D]
        # Per-position scale gate (static shape → compile-friendly)
        scale     = torch.softmax(self.type_scale, dim=0)             # [4]
        out       = out * scale[pos_type].view(1, T, 1)               # [B, T, D]
        return out


class LearnableTokenEviction(nn.Module):
    """Content-adaptive token selection via a lightweight importance scorer.

    Keeps the top-k highest-scoring tokens (k = ceil(T / rate)) in original
    temporal order, replacing strided conv downsampling.
    """

    def __init__(self, d_model: int, rate: int = 4, kernel_size: int = 7):
        super().__init__()
        self.rate    = rate
        self.d_model = d_model
        self.scorer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=d_model, bias=False),
            nn.Conv1d(d_model, 1, kernel_size=1, bias=True),
        )
        # Value-preserving linear projection after token selection.
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def compress(self, acts: torch.Tensor) -> torch.Tensor:
        """acts: (B, T, D) -> (B, k, D)"""
        B, T, D = acts.shape
        k = max(1, min(math.ceil(T / self.rate), T))

        x_t = acts.transpose(1, 2)
        scores = self.scorer(x_t).squeeze(1)  # (B, T)

        _, topk_idx = scores.topk(k, dim=1)
        topk_idx, _ = topk_idx.sort(dim=1)
        idx_exp  = topk_idx.unsqueeze(-1).expand(B, k, D)
        selected = acts.gather(1, idx_exp)  # (B, k, D)

        # Straight-through soft gating for end-to-end training.
        soft_w   = torch.sigmoid(scores).gather(1, topk_idx)
        selected = selected * soft_w.unsqueeze(-1)

        # Project and normalise.
        out = self.proj(selected.transpose(1, 2)).transpose(1, 2)
        return self.norm(out)


class MemoryManager(nn.Module):
    """LTE-backed memory compressor. Keeps the most informative tokens."""

    def __init__(self, d_model: int, rate: int = 4, deep: bool = True):
        super().__init__()
        self.rate = rate
        self.lte  = LearnableTokenEviction(d_model, rate)

    def compress(self, acts: torch.Tensor) -> torch.Tensor:
        """acts: (B, T, D) -> (B, ceil(T/rate), D)"""
        return self.lte.compress(acts)


class CompressiveAttention(nn.Module):
    """Multi-head attention with two-tier memory (recent + compressed past)."""

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
        self.mem_kv  = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out     = nn.Linear(d_model, d_model, bias=False)
        self.infini_beta = nn.Parameter(torch.zeros(n_heads, 1, 1))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, n_heads, T, d_k)"""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, comp_mem=None):
        B, T, D = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        Q = self._split_heads(q)
        K = self._split_heads(k)
        V = self._split_heads(v)

        # Local causal attention on the current window.
        out_local = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        if comp_mem is not None:
            km, vm = self.mem_kv(comp_mem).chunk(2, dim=-1)
            Km = self._split_heads(km)
            Vm = self._split_heads(vm)
            out_mem = F.scaled_dot_product_attention(
                Q, Km, Vm,
                dropout_p=0.0,
                is_causal=False,
            )
            beta = torch.sigmoid(self.infini_beta)
            out  = beta * out_mem + (1.0 - beta) * out_local
        else:
            out = out_local

        out = out.transpose(1, 2).contiguous().view(B, T, D)

        new_comp = self.mem_mgr.compress(x)
        if comp_mem is not None:
            new_comp = torch.cat([comp_mem, new_comp], dim=1)
            max_comp = self.window // 2
            if new_comp.size(1) > max_comp:
                new_comp = new_comp[:, -max_comp:]

        return self.out(out), new_comp


class SwiGLU(nn.Module):
    """SwiGLU feed-forward (Shazeer 2020). No bias, no dropout."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, window, rate, dropout):
        super().__init__()
        self.attn = CompressiveAttention(d_model, n_heads, window, rate, dropout)
        self.ff   = SwiGLU(d_model, d_ff)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x, comp_mem=None):
        attn_out, new_comp = self.attn(self.ln1(x), comp_mem)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_comp


class CompressiveTransformer(nn.Module):
    """Byte-level compressive transformer with LTE memory, Infini-Attention, SwiGLU.

    Optional: use_tdt=True enables TDT embedding for IEEE-754 float32 data.
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 512,
                 n_layers: int = 8, n_heads: int = 8, d_ff: int = 2048,
                 window: int = 512, compression_rate: int = 4,
                 dropout: float = 0.1, use_tdt: bool = False):
        super().__init__()
        self.use_tdt = use_tdt
        if use_tdt:
            self.embed = TDTEmbedding(d_model, vocab_size)
        else:
            self.embed = nn.Embedding(vocab_size, d_model)
        self.pe     = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, window, compression_rate, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._use_grad_ckpt = False
        self._init_weights()

    def _init_weights(self):
        n = sum(1 for _ in self.layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def enable_gradient_checkpointing(self):
        self._use_grad_ckpt = True

    def _forward_layer(self, layer, x, mem):
        if self._use_grad_ckpt and self.training:
            def fn(x, mem):
                return layer(x, mem if mem is not None else None)
            return grad_checkpoint(fn, x, mem, use_reentrant=False)
        return layer(x, mem)

    def forward(self, x: torch.Tensor, memories=None):
        if memories is None:
            memories = [None] * len(self.layers)
        h = self.pe(self.embed(x))
        new_mems = []
        for layer, mem in zip(self.layers, memories):
            h, new_mem = self._forward_layer(layer, h, mem)
            new_mems.append(new_mem)
        return self.head(self.ln_f(h)), new_mems

    @torch.no_grad()
    def get_probs(self, x: torch.Tensor, memories=None):
        logits, new_mems = self(x, memories)
        return torch.softmax(logits, dim=-1), new_mems


