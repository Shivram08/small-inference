"""Phase 2 paged cache: metadata builders, write/gather, and invariant 4 (paged outputs equal Phase 1's exactly).

The CPU tests use the miniature fp64 model. The GPU tests load Qwen2.5-0.5B (bf16 1 GB, then fp32 2 GB) and run
in well under a minute. Pools are fragmented on purpose, so block tables are shuffled and non-contiguous.
"""

import json

import pytest
import torch

from engine.block_manager import BlockManager
from engine.config import ModelConfig
from engine.generate import greedy_generate, greedy_generate_paged
from engine.kv_cache import ContiguousKVCache
from engine.model import Qwen2Model
from engine.paged_cache import PagedKVCache, build_metadata
from tests.tiny_model import TINY, random_weights

LENGTHS = [1, 15, 16, 17, 33]  # around the 16-token block boundary, and a single token


def fragmented_pool(cfg: ModelConfig, num_blocks: int, block_size: int, device, dtype, seed: int = 0):
    """A manager and cache whose free list is shuffled: fill the pool with 1-block sequences, free them in random
    order. Every later allocation gets a non-monotonic block table."""
    m = BlockManager(num_blocks, block_size)
    for s in range(num_blocks):
        m.allocate(1000 + s, block_size)
    order = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(seed)).tolist()
    for s in order:
        m.free(1000 + s)
    return m, PagedKVCache(cfg, num_blocks, block_size, device, dtype)


@pytest.fixture(scope="module")
def tiny():
    """The miniature model in fp64 on the CPU."""
    return Qwen2Model(TINY, random_weights(TINY))


def test_metadata_for_a_packed_batch():
    """Two sequences in one forward: a 5-token prefill chunk at positions 3..7 and a single decode token at 9."""
    m = BlockManager(num_blocks=8, block_size=4)
    m.allocate(0, 3)  # block 0
    m.allocate(1, 9)  # blocks 1, 2, 3
    m.append_slots(0, 5)  # positions 3..7: finishes block 0, takes block 4
    m.append_slot(1)  # position 9: block 3, offset 1
    meta = build_metadata(m, [(0, 3, 5), (1, 9, 1)], "cpu")
    assert meta.num_tokens == 6 and meta.num_seqs == 2
    assert meta.query_starts == [0, 5, 6] and meta.context_lens == [8, 10]
    assert meta.positions.tolist() == [3, 4, 5, 6, 7, 9]
    assert meta.slot_mapping.tolist() == [3, 16, 17, 18, 19, 13]
    assert meta.slot_blocks.tolist() == [0, 4, 4, 4, 4, 3] and meta.slot_offsets.tolist() == [3, 0, 1, 2, 3, 1]
    assert [b.tolist() for b in meta.block_ids] == [[0, 4], [1, 2, 3]]
    assert meta.block_tables.tolist() == [[0, 4, -1], [1, 2, 3]] and meta.block_tables.dtype == torch.int32
    assert meta.context_lens_t.tolist() == [8, 10] and meta.last_token_indices.tolist() == [4, 5]
    with pytest.raises(ValueError):
        build_metadata(m, [(0, 8, 1)], "cpu")  # position 8 has no slot yet


def test_write_then_gather_roundtrip():
    """K/V written at shuffled slots come back from gather() in position order, for a length not a multiple of B."""
    m, cache = fragmented_pool(TINY, 10, 4, "cpu", torch.float64)
    m.allocate(7, 11)
    meta = build_metadata(m, [(7, 0, 11)], "cpu")
    cache.begin(11, meta)
    k = torch.randn(11, TINY.num_kv_heads, TINY.head_dim, dtype=torch.float64)
    v = torch.randn_like(k)
    cache.write(1, k, v)
    gk, gv = cache.gather(1, 0)
    assert torch.equal(gk, k.transpose(0, 1)) and torch.equal(gv, v.transpose(0, 1))
    t = m.block_table(7)
    assert any(b != a + 1 for a, b in zip(t, t[1:])), f"table {t} is contiguous; the test needs scattered blocks"


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_paged_equals_contiguous_tiny(tiny, length, block_size):
    """Invariant 4 on the miniature model: prefill logits and 20 decode steps' logits are bit-identical."""
    prompt = torch.randint(0, TINY.vocab_size, (length,), generator=torch.Generator().manual_seed(length)).tolist()
    m, pcache = fragmented_pool(TINY, 16 * 64 // block_size, block_size, "cpu", torch.float64)
    ccache = ContiguousKVCache(TINY, length + 20, "cpu", torch.float64)
    m.allocate(0, length)
    meta = build_metadata(m, [(0, 0, length)], "cpu")
    ids = torch.tensor(prompt)
    paged = tiny.forward(ids, meta.positions, pcache, meta=meta)
    contiguous = tiny.forward(ids, ccache.next_positions(length), ccache)
    assert torch.equal(paged, contiguous)
    tok = paged[-1].argmax()[None]
    for pos in range(length, length + 20):
        m.append_slot(0)
        meta = build_metadata(m, [(0, pos, 1)], "cpu")
        paged = tiny.forward(tok, meta.positions, pcache, meta=meta)
        contiguous = tiny.forward(tok, ccache.next_positions(1), ccache)
        assert torch.equal(paged, contiguous), f"decode position {pos}"
        tok = paged[-1].argmax()[None]


def test_chunked_prefill_through_paged_cache(tiny):
    """Prefilling in 5-token chunks across 4-token blocks generates the same tokens as one contiguous prefill."""
    prompt = list(range(3, 26))
    m, cache = fragmented_pool(TINY, 32, 4, "cpu", torch.float64)
    assert greedy_generate_paged(tiny, prompt, 15, m, cache, prefill_chunk=5) == greedy_generate(tiny, prompt, 15)
    assert m.num_free_blocks() == 32  # the sequence's blocks were returned


def test_packed_batch_matches_individual_tiny(tiny):
    """Two sequences in one forward (a prefill and a decode) get the same logits as run alone. The packed linear
    layers see a different M, so this is fp64-close rather than bitwise (bitwise batch invariance is Phase 4's)."""
    a, b = list(range(10, 23)), list(range(40, 49))
    alone = {}
    for name, prompt, extra in (("a", a, 0), ("b", b, 1)):
        m, cache = fragmented_pool(TINY, 16, 4, "cpu", torch.float64)
        m.allocate(0, len(prompt) + extra)
        meta = build_metadata(m, [(0, 0, len(prompt) + extra)], "cpu")
        alone[name] = tiny.forward(torch.tensor(prompt + [5] * extra), meta.positions, cache, meta=meta)
    m, cache = fragmented_pool(TINY, 16, 4, "cpu", torch.float64)
    m.allocate(1, len(b))
    meta = build_metadata(m, [(1, 0, len(b))], "cpu")
    tiny.forward(torch.tensor(b), meta.positions, cache, meta=meta)  # b's prompt is already cached
    m.allocate(0, len(a))
    m.append_slot(1)
    meta = build_metadata(m, [(0, 0, len(a)), (1, len(b), 1)], "cpu")
    packed = tiny.forward(torch.tensor(a + [5]), meta.positions, cache, meta=meta)
    torch.testing.assert_close(packed[:len(a)], alone["a"], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(packed[len(a):], alone["b"][-1:], rtol=1e-10, atol=1e-10)
    # last_only gives each sequence's final row (rerunning rewrites the same K/V into the same slots)
    last = tiny.forward(torch.tensor(a + [5]), meta.positions, cache, meta=meta, last_only=True)
    assert last.shape[0] == 2 and torch.equal(last, packed[[len(a) - 1, len(a)]])


def test_blocks_for_budget():
    """2 GiB of bf16 KV for the 0.5B model is 10,922 blocks of 16 tokens (192 KiB each)."""
    cfg = ModelConfig(vocab_size=151936, hidden_size=896, intermediate_size=4864, num_layers=24, num_q_heads=14,
                      num_kv_heads=2, head_dim=64, rms_norm_eps=1e-6, rope_theta=1e6, max_position_embeddings=32768,
                      tie_word_embeddings=True)
    assert PagedKVCache.blocks_for_budget(cfg, 2 * 2**30, 16) == 10_922


# --- Real model on the GPU ---------------------------------------------------------------------------------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="module", params=[torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def real(request):
    """Qwen2.5-0.5B in one dtype at a time (the previous one is released first)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    try:
        model = Qwen2Model.from_pretrained(dtype=request.param)
    except OSError:
        pytest.skip("Qwen2.5-0.5B-Instruct not in the local HF cache")
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def trace_prompt():
    """The first ShareGPT trace prompt with at least 33 tokens."""
    with open("traces/sharegpt_n100_seed0_burst.jsonl") as f:
        return next(r["prompt_ids"] for r in map(json.loads, f) if len(r["prompt_ids"]) >= 33)


@gpu
@pytest.mark.parametrize("length", LENGTHS)
def test_paged_equals_contiguous_real(real, trace_prompt, length):
    """Invariant 4 on Qwen2.5-0.5B: prefill logits bit-identical, and 50 greedy tokens identical, in a shuffled pool
    of 16-token blocks."""
    prompt = trace_prompt[:length]
    m, pcache = fragmented_pool(real.cfg, 64, 16, "cuda", real.dtype)
    ccache = ContiguousKVCache(real.cfg, length, "cuda", real.dtype)
    m.allocate(0, length)
    meta = build_metadata(m, [(0, 0, length)], "cuda")
    ids = torch.tensor(prompt, device="cuda")
    assert torch.equal(real.forward(ids, meta.positions, pcache, meta=meta),
                       real.forward(ids, ccache.next_positions(length), ccache))
    m.free(0)
    assert greedy_generate_paged(real, prompt, 50, m, pcache) == greedy_generate(real, prompt, 50)
    assert m.num_free_blocks() == 64
