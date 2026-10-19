"""Phase 1 engine: the batch-1 contiguous-cache engine follows the Engine contract and reproduces greedy_generate.

Loads the 0.5B model once in bf16 (1 GB of GPU memory); a few seconds.
"""

import pytest
import torch

from bench.engines.base import Engine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="module")
def engine():
    """One loaded ContiguousEngine with a small cache."""
    from engine.engine import ContiguousEngine
    try:
        return ContiguousEngine(max_len=512)
    except OSError:
        pytest.skip("Qwen2.5-0.5B-Instruct not in the local HF cache")


def drain(engine):
    """Step until done; return the (req_id, token_id, finished) triples."""
    out = []
    while engine.has_unfinished():
        out.extend(engine.step())
    return out


def test_conforms_to_protocol(engine):
    """ContiguousEngine satisfies the Engine protocol."""
    assert isinstance(engine, Engine)


def test_fifo_and_matches_greedy_generate(engine):
    """Two requests run FIFO; each one's tokens equal greedy_generate on a fresh cache (so the shared cache is reset
    correctly between requests), with exactly one finished flag at the end."""
    from engine.generate import greedy_generate
    prompts = [[9707, 11, 1879, 0], list(range(1000, 1040))]
    n = [12, 30]
    for i, (p, k) in enumerate(zip(prompts, n)):
        engine.add_request(i, p, k)
    out = drain(engine)
    assert [rid for rid, _, _ in out] == [0] * n[0] + [1] * n[1]
    for i, (p, k) in enumerate(zip(prompts, n)):
        assert [tok for rid, tok, _ in out if rid == i] == greedy_generate(engine.model, p, k)
        assert [fin for rid, _, fin in out if rid == i] == [False] * (k - 1) + [True]
    assert engine.step() == [] and not engine.has_unfinished()


def test_rejects_invalid_requests(engine):
    """Empty prompts, output_len < 2, and requests longer than the cache are refused up front."""
    with pytest.raises(ValueError):
        engine.add_request(0, [], 5)
    with pytest.raises(ValueError):
        engine.add_request(0, [1, 2], 1)
    with pytest.raises(ValueError, match="max_len"):
        engine.add_request(0, [1] * 500, 13)
