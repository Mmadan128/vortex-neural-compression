# Usage: pytest tests/ -v
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, pytest
from vortex.models.compressive_transformer import CompressiveTransformer
from vortex.models.optimized_transformer  import OptimisedCompressiveTransformer
from vortex.compression.arithmetic_coding import probs_to_cdf, theoretical_bpd


@pytest.fixture
def small_ct():
    return CompressiveTransformer(d_model=64, n_layers=2, n_heads=2, d_ff=128, window=16)

@pytest.fixture
def small_opt():
    return OptimisedCompressiveTransformer(d_model=64, n_layers=2, n_heads=2, d_ff=128, window=16)


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
