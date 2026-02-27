# Usage: from vortex.models.optimized_transformer import OptimisedCompressiveTransformer
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

# LTE-backed MemoryManager, TDTEmbedding, and SwiGLU from compressive_transformer.
from .compressive_transformer import (
    PositionalEncoding,
    MemoryManager,
    TDTEmbedding,
    SwiGLU,
)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.scale


class OptimisedCompressiveAttention(nn.Module):
    """Multi-head attention with Flash Attention 2 / PyTorch SDPA, KV cache,
    LTE memory compression, and Infini-Attention beta gating.
    """

    def __init__(self, d_model: int, n_heads: int,
                 window: int = 512, rate: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.window  = window
        self.dropout = dropout
        self.mem_mgr  = MemoryManager(d_model, rate)
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.mem_kv   = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out      = nn.Linear(d_model, d_model, bias=False)
        # Beta init at -3 so training starts near pure local attention.
        self.infini_beta = nn.Parameter(torch.full((n_heads, 1, 1), -3.0))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, n_heads, T, d_k)"""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _project(self, x: torch.Tensor):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return q, k, v

    def _sdpa(self, Q, K, V, causal: bool) -> torch.Tensor:
        """Flash-Attn 2 for causal; PyTorch SDPA otherwise."""
        if FLASH_AVAILABLE and Q.is_cuda and causal:
            # flash_attn_func expects (B, seqlen, nheads, headdim) in fp16/bf16.
            # Preserve original dtype (e.g. bfloat16 on MI300X/ROCm) instead of
            # hardcoding .half() which silently downcasts bf16 -> fp16.
            dtype = Q.dtype
            fa_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            out = flash_attn_func(
                Q.transpose(1, 2).to(fa_dtype),
                K.transpose(1, 2).to(fa_dtype),
                V.transpose(1, 2).to(fa_dtype),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            ).to(dtype).transpose(1, 2)
        else:
            out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            )
        return out

    def forward(self, x: torch.Tensor, comp_mem=None, kv_cache=None):
        B, T, D = x.shape
        q, k, v = self._project(x)
        Q = self._split(q)
        K = self._split(k)
        V = self._split(v)

        if kv_cache is not None:
            K = torch.cat([kv_cache["k"], K], dim=2)
            V = torch.cat([kv_cache["v"], V], dim=2)

        out_local = self._sdpa(Q, K, V, causal=True)

        if comp_mem is not None:
            km, vm = self.mem_kv(comp_mem).chunk(2, dim=-1)
            Km = self._split(km)
            Vm = self._split(vm)
            out_mem  = self._sdpa(Q, Km, Vm, causal=False)
            beta     = torch.sigmoid(self.infini_beta)
            out      = beta * out_mem + (1.0 - beta) * out_local
        else:
            out = out_local

        new_cache = {"k": K.detach(), "v": V.detach()}

        out = out.transpose(1, 2).contiguous().view(B, T, D)

        new_comp = self.mem_mgr.compress(x)
        if comp_mem is not None:
            new_comp = torch.cat([comp_mem, new_comp], dim=1)
            max_c = self.window // 2
            if new_comp.size(1) > max_c:
                new_comp = new_comp[:, -max_c:]

        return self.out(out), new_comp, new_cache


class OptimisedBlock(nn.Module):
    """Transformer block with RMSNorm and SwiGLU."""
    def __init__(self, d_model, n_heads, d_ff, window, rate, dropout):
        super().__init__()
        self.attn = OptimisedCompressiveAttention(d_model, n_heads, window, rate, dropout)
        self.ff   = SwiGLU(d_model, d_ff)
        self.ln1  = RMSNorm(d_model)
        self.ln2  = RMSNorm(d_model)

    def forward(self, x, comp_mem=None, kv_cache=None):
        attn_out, new_comp, new_cache = self.attn(self.ln1(x), comp_mem, kv_cache)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_comp, new_cache


class OptimisedCompressiveTransformer(nn.Module):
    """Byte-level compressive transformer: RMSNorm, Flash Attention 2,
    LTE memory, Infini-Attention, TDT embedding (opt-in), KV cache.
    """

    def __init__(self, vocab_size=256, d_model=512, n_layers=8, n_heads=8,
                 d_ff=2048, window=512, compression_rate=4, dropout=0.1,
                 use_tdt: bool = False):
        super().__init__()
        self.use_tdt = use_tdt
        if use_tdt:
            self.embed = TDTEmbedding(d_model, vocab_size)
        else:
            self.embed = nn.Embedding(vocab_size, d_model)
        self.pe     = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            OptimisedBlock(d_model, n_heads, d_ff, window, compression_rate, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f   = RMSNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
        self._use_grad_ckpt = False
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
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

    def _run_layer(self, layer, x, mem, cache):
        if self._use_grad_ckpt and self.training:
            def fn(x, mem, cache):
                return layer(x, mem, cache)
            return grad_checkpoint(fn, x, mem, cache, use_reentrant=False)
        return layer(x, mem, cache)

    def forward(self, x: torch.Tensor, memories=None, kv_caches=None):
        if memories  is None: memories  = [None] * len(self.layers)
        if kv_caches is None: kv_caches = [None] * len(self.layers)
        h = self.pe(self.embed(x))
        new_mems, new_caches = [], []
        for layer, mem, cache in zip(self.layers, memories, kv_caches):
            h, new_mem, new_cache = self._run_layer(layer, h, mem, cache)
            new_mems.append(new_mem)
            new_caches.append(new_cache)
        return self.head(self.ln_f(h)), new_mems, new_caches

    @torch.no_grad()
    def get_probs(self, x, memories=None, kv_caches=None):
        logits, mems, caches = self(x, memories, kv_caches)
        return torch.softmax(logits, dim=-1), mems, caches

    def vram_estimate_gb(self, batch_size: int, seq_len: int) -> dict:
        """Rough VRAM breakdown in GB."""
        params   = sum(p.numel() for p in self.parameters())
        p_fp16   = params * 2 / 1e9
        p_fp32   = params * 4 / 1e9
        d_model  = self.embed.embedding_dim if hasattr(self.embed, "embedding_dim") else self.embed.d_model
        n_layers = len(self.layers)
        acts     = 2 * batch_size * seq_len * d_model * n_layers * 12 / 1e9
        opt      = params * 8 / 1e9
        return {
            "params_fp16_GB":      round(p_fp16, 2),
            "params_fp32_GB":      round(p_fp32, 2),
            "activations_GB":      round(acts,   2),
            "optimizer_states_GB": round(opt,    2),
            "total_training_GB":   round(p_fp32 + acts + opt, 2),
            "inference_fp16_GB":   round(p_fp16, 2),
        }


class CATWrapper(nn.Module):
    """Dynamic chunk scheduler for OptimisedCompressiveTransformer.

    During training samples chunk sizes from chunk_sizes uniformly, enabling
    test-time quality/compute control without retraining.
    During inference, pass chunk_size= to override.
    """

    def __init__(self, model: nn.Module,
                 chunk_sizes: tuple = (128, 256, 512)):
        super().__init__()
        self.model       = model
        self.chunk_sizes = chunk_sizes

    def forward(self, x: torch.Tensor,
                memories=None, kv_caches=None,
                chunk_size: int = None):
        if chunk_size is None:
            if self.training:
                chunk_size = random.choice(self.chunk_sizes)
            else:
                chunk_size = self.chunk_sizes[-1]

        B, T = x.shape

        if T <= chunk_size:
            return self.model(x, memories, kv_caches)

        all_logits = []
        for start in range(0, T, chunk_size):
            end   = min(start + chunk_size, T)
            chunk = x[:, start:end]
            logits, memories, kv_caches = self.model(chunk, memories, kv_caches)
            memories = [m.detach() if m is not None else None for m in memories]
            all_logits.append(logits)

        return torch.cat(all_logits, dim=1), memories, kv_caches

    # Convenience proxies so the wrapper is transparent to training code.
    def enable_gradient_checkpointing(self):
        self.model.enable_gradient_checkpointing()

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def vram_estimate_gb(self, batch_size: int, seq_len: int) -> dict:
        return self.model.vram_estimate_gb(batch_size, seq_len)

    def state_dict(self, *args, **kwargs):
        """Delegate to inner model so checkpoints are portable without the wrapper."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Delegate to inner model."""
        return self.model.load_state_dict(state_dict, strict=strict)



