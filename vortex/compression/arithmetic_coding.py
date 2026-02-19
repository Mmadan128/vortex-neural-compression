# Usage: import vortex.compression.arithmetic_coding
import torch

try:
    import torchac
    TORCHAC_AVAILABLE = True
except ImportError:
    TORCHAC_AVAILABLE = False
    print("[WARNING] torchac not found — install with: pip install torchac")


def probs_to_cdf(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """probs: (..., vocab_size) -> CDF (..., vocab_size+1) in [0, 1]."""
    probs = (probs + eps) / (probs + eps).sum(dim=-1, keepdim=True)
    cdf   = torch.zeros(*probs.shape[:-1], probs.size(-1) + 1, device=probs.device)
    cdf[..., 1:] = probs.cumsum(dim=-1)
    cdf[..., -1] = 1.0
    return cdf


def encode(probs: torch.Tensor, symbols: torch.Tensor) -> bytes:
    """Lossless arithmetic encode. probs:(B,T,256), symbols:(B,T) -> bytes."""
    if not TORCHAC_AVAILABLE:
        raise RuntimeError("torchac required. pip install torchac")
    cdf     = probs_to_cdf(probs)
    cdf_int = (cdf * 65536).clamp(0, 65536).short().cpu()
    return torchac.encode_float_cdf(cdf_int, symbols.short().cpu())


def decode(bitstring: bytes, probs: torch.Tensor) -> torch.Tensor:
    """Lossless arithmetic decode. Returns (B,T) int16 symbols."""
    if not TORCHAC_AVAILABLE:
        raise RuntimeError("torchac required. pip install torchac")
    cdf     = probs_to_cdf(probs)
    cdf_int = (cdf * 65536).clamp(0, 65536).short().cpu()
    return torchac.decode_float_cdf(bitstring, cdf_int)


def theoretical_bpd(probs: torch.Tensor, symbols: torch.Tensor) -> float:
    """Cross-entropy in bits per byte. Lower is better."""
    gathered = probs.gather(-1, symbols.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log2(gathered.clamp(min=1e-10)).mean().item()
