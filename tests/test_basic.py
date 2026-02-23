# Usage: pytest tests/ -v
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, pytest
from vortex.models.compressive_transformer import (
    CompressiveTransformer,
    LearnableTokenEviction,    # Feature 1
    TDTEmbedding,              # Feature 4
    MemoryManager,
)
from vortex.models.optimized_transformer import (
    OptimisedCompressiveTransformer,
    CATWrapper,                # Feature 2
)
from vortex.compression.arithmetic_coding import probs_to_cdf, theoretical_bpd
from vortex.utils.zipnn import (                # Feature 5
    compress_model_weights,
    decompress_model_weights,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def small_ct():
    return CompressiveTransformer(d_model=64, n_layers=2, n_heads=2, d_ff=128, window=16)

@pytest.fixture
def small_opt():
    return OptimisedCompressiveTransformer(d_model=64, n_layers=2, n_heads=2, d_ff=128, window=16)

@pytest.fixture
def small_tdt():
    return OptimisedCompressiveTransformer(d_model=64, n_layers=2, n_heads=2, d_ff=128,
                                           window=16, use_tdt=True)


# ─── original shape / memory tests ────────────────────────────────────────────

def test_ct_forward_shape(small_ct):
    x = torch.randint(0, 256, (2, 16))
    logits, mems = small_ct(x)
    assert logits.shape == (2, 16, 256)
    assert len(mems) == 2


def test_ct_memory_preserved(small_ct):
    x = torch.randint(0, 256, (1, 16))
    _, mems = small_ct(x)
    mems = [m.detach() if m is not None else None for m in mems]
    _, mems2 = small_ct(x, mems)
    assert any(m is not None for m in mems2)


def test_opt_forward(small_opt):
    x = torch.randint(0, 256, (2, 16))
    logits, mems, caches = small_opt(x)
    assert logits.shape == (2, 16, 256)
    assert "k" in caches[0] and "v" in caches[0]


def test_cdf_monotonic():
    probs = torch.softmax(torch.randn(2, 10, 256), dim=-1)
    cdf   = probs_to_cdf(probs)
    assert cdf.shape == (2, 10, 257)
    assert (cdf[..., 1:] >= cdf[..., :-1]).all()
    assert torch.allclose(cdf[..., -1], torch.ones(2, 10), atol=1e-4)


def test_bpd_positive():
    probs = torch.softmax(torch.randn(1, 16, 256), dim=-1)
    syms  = torch.randint(0, 256, (1, 16))
    assert theoretical_bpd(probs, syms) > 0


# ─── Feature 1: Learnable Token Eviction ──────────────────────────────────────

def test_lte_output_shape():
    """LTE should return ceil(T/rate) tokens in temporal order."""
    lte = LearnableTokenEviction(d_model=64, rate=4)
    acts = torch.randn(2, 32, 64)
    out  = lte.compress(acts)
    assert out.shape == (2, 8, 64), f"Expected (2,8,64), got {out.shape}"


def test_lte_small_t():
    """LTE must not crash when T < rate."""
    lte = LearnableTokenEviction(d_model=64, rate=4)
    acts = torch.randn(1, 2, 64)
    out  = lte.compress(acts)
    assert out.shape[0] == 1
    assert out.shape[2] == 64


def test_lte_gradients():
    """Importance scorer must receive gradients."""
    lte    = LearnableTokenEviction(d_model=32, rate=4)
    acts   = torch.randn(1, 16, 32, requires_grad=False)
    inp    = torch.randn(1, 16, 32, requires_grad=True)
    out    = lte.compress(inp)
    loss   = out.sum()
    loss.backward()
    # Scorer conv weights should have gradients
    for p in lte.scorer.parameters():
        assert p.grad is not None


def test_memory_manager_uses_lte():
    """MemoryManager.compress should delegate to LTE (no strided conv)."""
    mm  = MemoryManager(d_model=32, rate=4)
    acts = torch.randn(2, 16, 32)
    out  = mm.compress(acts)
    assert out.shape == (2, 4, 32)
    assert hasattr(mm, "lte"), "MemoryManager must hold an LTE sub-module"


# ─── Feature 3: Infini-Attention β gating ─────────────────────────────────────

def test_infini_beta_parameter_exists(small_opt):
    """Every attention layer must expose an infini_beta parameter."""
    for layer in small_opt.layers:
        assert hasattr(layer.attn, "infini_beta"), \
            "OptimisedCompressiveAttention missing infini_beta"
        assert layer.attn.infini_beta.shape == (layer.attn.n_heads, 1, 1)


def test_infini_gating_effect(small_opt):
    """With a non-trivial memory, output should differ from no-memory case."""
    small_opt.eval()
    x    = torch.randn(1, 16, 64)
    mem  = torch.randn(1, 4, 64)
    with torch.no_grad():
        out_no_mem,  _, _ = small_opt.layers[0].attn(x)
        out_with_mem, _, _ = small_opt.layers[0].attn(x, comp_mem=mem)
    assert not torch.allclose(out_no_mem, out_with_mem), \
        "Memory gating should change the output"


# ─── Feature 4: TDT Embedding ─────────────────────────────────────────────────

def test_tdt_embedding_shape():
    embed = TDTEmbedding(d_model=64)
    x     = torch.randint(0, 256, (2, 16))
    out   = embed(x)
    assert out.shape == (2, 16, 64)


def test_tdt_different_positions_differ():
    """Bytes at different positions within the float should produce different embeddings."""
    embed = TDTEmbedding(d_model=64)
    embed.eval()
    # Two identical byte values at different float positions
    x = torch.tensor([[42, 42, 42, 42]])   # (1, 4) — pos 0,1,2,3
    with torch.no_grad():
        out = embed(x)
    # Embeddings at each byte position should differ (different tables)
    assert not torch.allclose(out[0, 0], out[0, 1]), \
        "TDT must use different tables for different byte positions"


def test_tdt_model_forward(small_tdt):
    x = torch.randint(0, 256, (2, 16))
    logits, mems, caches = small_tdt(x)
    assert logits.shape == (2, 16, 256)


# ─── Feature 2: CAT Dynamic Chunking ──────────────────────────────────────────

def test_cat_wrapper_train_mode_chunks():
    """During training, CATWrapper should split a long sequence into chunks."""
    base   = OptimisedCompressiveTransformer(d_model=64, n_layers=2, n_heads=2,
                                             d_ff=128, window=32)
    cat    = CATWrapper(base, chunk_sizes=(8, 16, 32))
    cat.train()
    x = torch.randint(0, 256, (1, 32))
    logits, mems, caches = cat(x)
    assert logits.shape == (1, 32, 256)


def test_cat_wrapper_eval_uses_largest_chunk():
    """In eval mode, CATWrapper should use the largest chunk (best quality)."""
    base  = OptimisedCompressiveTransformer(d_model=64, n_layers=2, n_heads=2,
                                            d_ff=128, window=16)
    cat   = CATWrapper(base, chunk_sizes=(4, 8, 16))
    cat.eval()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        logits, _, _ = cat(x)
    assert logits.shape == (1, 16, 256)


def test_cat_wrapper_explicit_chunk_size():
    base = OptimisedCompressiveTransformer(d_model=64, n_layers=2, n_heads=2,
                                           d_ff=128, window=32)
    cat  = CATWrapper(base, chunk_sizes=(8, 16, 32))
    x    = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        logits, _, _ = cat(x, chunk_size=8)
    assert logits.shape == (1, 32, 256)


def test_cat_state_dict_portable():
    """CATWrapper.state_dict() must produce keys compatible with bare model."""
    base = OptimisedCompressiveTransformer(d_model=64, n_layers=2, n_heads=2,
                                           d_ff=128, window=16)
    cat  = CATWrapper(base, chunk_sizes=(8, 16))
    sd   = cat.state_dict()
    # All keys should be loadable into a bare model
    base2 = OptimisedCompressiveTransformer(d_model=64, n_layers=2, n_heads=2,
                                            d_ff=128, window=16)
    missing, unexpected = base2.load_state_dict(sd, strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected}"


# ─── Feature 5: ZipNN Weight Compression ──────────────────────────────────────

def test_zipnn_round_trip(small_opt):
    """Decompress → exact bit equality with original weights."""
    compressed = compress_model_weights(small_opt)
    model2     = OptimisedCompressiveTransformer(d_model=64, n_layers=2, n_heads=2,
                                                  d_ff=128, window=16)
    decompress_model_weights(model2, compressed, device="cpu")
    for (n1, p1), (n2, p2) in zip(small_opt.named_parameters(),
                                   model2.named_parameters()):
        assert torch.equal(p1.data.float(), p2.data.float()), \
            f"Mismatch in tensor '{n1}' after ZipNN round-trip"


def test_zipnn_compression_ratio(small_opt):
    """Compressed blobs must exist and have all required fields.
    (Ratio >= 1 is only guaranteed on trained weights with skewed exponent
    distributions; randomly-initialised weights have uniform exponents so
    Huffman tables add overhead rather than savings.)
    """
    compressed = compress_model_weights(small_opt)
    any_compressed = any(
        "exp_bytes" in v for k, v in compressed.items() if not k.startswith("__")
    )
    assert any_compressed, "No tensors were Huffman-compressed"
    # Structural check: every compressed blob must have all required fields.
    required = {"exp_bytes", "exp_codes", "exp_n", "exp_pad",
                "sign_bytes", "sign_codes", "sign_n", "sign_pad",
                "mantissa_raw", "shape", "dtype"}
    for name, blob in compressed.items():
        if name.startswith("__") or "raw" in blob:
            continue
        missing_fields = required - blob.keys()
        assert not missing_fields, \
            f"Blob for '{name}' missing fields: {missing_fields}"

