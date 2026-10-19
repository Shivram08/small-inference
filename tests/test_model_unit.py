"""Phase 1 unit tests on the CPU with random weights: attention, RoPE properties, KV cache, cached == uncached.

These need neither the GPU nor the checkpoint, so they run in about a second. The HF comparisons are in
test_model_hf.py.
"""

import pytest
import torch
import torch.nn.functional as F

from engine.config import ModelConfig
from engine.generate import greedy_generate
from engine.kv_cache import ContiguousKVCache
from engine.model import Qwen2Model, Rope, attention

# A miniature Qwen2: 2 layers, 6 query heads over 2 KV heads (g = 3), head_dim 16.
TINY = ModelConfig(vocab_size=97, hidden_size=96, intermediate_size=160, num_layers=2, num_q_heads=6,
                   num_kv_heads=2, head_dim=16, rms_norm_eps=1e-6, rope_theta=10_000.0,
                   max_position_embeddings=256, tie_word_embeddings=True)


def random_weights(cfg: ModelConfig, dtype=torch.float64, seed=0) -> dict[str, torch.Tensor]:
    """HF-named random tensors with the real shapes and small scales, so activations stay O(1)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: (torch.randn(*shape, generator=g, dtype=dtype) * 0.1)  # noqa: E731
    w = {"model.embed_tokens.weight": r(cfg.vocab_size, cfg.hidden_size),
         "model.norm.weight": 1 + r(cfg.hidden_size)}
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        w |= {p + "input_layernorm.weight": 1 + r(cfg.hidden_size),
              p + "post_attention_layernorm.weight": 1 + r(cfg.hidden_size),
              p + "self_attn.q_proj.weight": r(cfg.q_size, cfg.hidden_size), p + "self_attn.q_proj.bias": r(cfg.q_size),
              p + "self_attn.k_proj.weight": r(cfg.kv_size, cfg.hidden_size),
              p + "self_attn.k_proj.bias": r(cfg.kv_size),
              p + "self_attn.v_proj.weight": r(cfg.kv_size, cfg.hidden_size),
              p + "self_attn.v_proj.bias": r(cfg.kv_size),
              p + "self_attn.o_proj.weight": r(cfg.hidden_size, cfg.q_size),
              p + "mlp.gate_proj.weight": r(cfg.intermediate_size, cfg.hidden_size),
              p + "mlp.up_proj.weight": r(cfg.intermediate_size, cfg.hidden_size),
              p + "mlp.down_proj.weight": r(cfg.hidden_size, cfg.intermediate_size)}
    w["lm_head.weight"] = w["model.embed_tokens.weight"]
    return w


@pytest.fixture(scope="module")
def tiny():
    """The miniature model in fp64 on the CPU."""
    return Qwen2Model(TINY, random_weights(TINY))


def sdpa_reference(q, k, v, q_pos, k_pos, scale):
    """Attention by PyTorch SDPA with K/V explicitly repeated per query head and an explicit boolean mask."""
    T, Hq, d = q.shape
    g = Hq // k.shape[0]
    kr, vr = k.repeat_interleave(g, dim=0), v.repeat_interleave(g, dim=0)  # [Hq, S, d], head h -> KV head h // g
    allowed = k_pos[None, :] <= q_pos[:, None]
    out = F.scaled_dot_product_attention(q.transpose(0, 1), kr, vr, attn_mask=allowed, scale=scale)
    return out.transpose(0, 1).reshape(T, Hq * d)


@pytest.mark.parametrize("T,S,offset", [(1, 1, 0), (1, 37, 36), (5, 5, 0), (7, 20, 13), (16, 16, 0)])
def test_attention_matches_sdpa(T, S, offset):
    """GQA broadcast attention equals SDPA with repeated KV: single token, decode, prefill, and a mid-sequence chunk."""
    torch.manual_seed(0)
    q = torch.randn(T, 6, 16, dtype=torch.float64)
    k, v = torch.randn(2, S, 16, dtype=torch.float64), torch.randn(2, S, 16, dtype=torch.float64)
    q_pos, k_pos = torch.arange(offset, offset + T), torch.arange(S)
    out = attention(q, k, v, q_pos, k_pos, 16 ** -0.5)
    torch.testing.assert_close(out, sdpa_reference(q, k, v, q_pos, k_pos, 16 ** -0.5), rtol=1e-12, atol=1e-12)


def test_rope_is_relative():
    """<R(m) q, R(n) k> depends only on m - n. fp32 tables limit this: at angle ~3000 rad the fp32 spacing is
    2.4e-4 rad, so positions stay small and the tolerance is 1e-4 relative to |q||k|."""
    rope = Rope(TINY, "cpu", max_positions=4096)
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 16), torch.randn(1, 1, 16)
    scale = (q.norm() * k.norm()).item()
    dot = lambda m, n: (rope(q, torch.tensor([m])) * rope(k, torch.tensor([n]))).sum().item()  # noqa: E731
    for m, n in [(0, 0), (5, 2), (100, 37), (1000, 1999)]:
        for shift in (1, 17, 1000):
            assert abs(dot(m, n) - dot(m + shift, n + shift)) < 1e-4 * scale
    assert abs(dot(0, 0) - (q * k).sum().item()) < 1e-6 * scale  # position 0 is the identity


def test_rope_pairs_i_with_i_plus_half():
    """Only dimensions i and i + d/2 mix: a one-hot input at i rotates into {i, i + d/2} and nowhere else."""
    rope = Rope(TINY, "cpu", max_positions=16)
    for i in (0, 3, 7):
        x = torch.zeros(1, 1, 16)
        x[0, 0, i] = 1.0
        y = rope(x, torch.tensor([5]))[0, 0]
        assert set(torch.nonzero(y.abs() > 1e-7).flatten().tolist()) <= {i, i + 8}
        assert y.norm().item() == pytest.approx(1.0, abs=1e-6)  # a rotation


def test_kv_cache_bookkeeping():
    """Positions continue the sequence, writes land at the right slots, reads are views of the valid prefix."""
    c = ContiguousKVCache(TINY, max_len=8, device="cpu", dtype=torch.float64)
    assert c.next_positions(3).tolist() == [0, 1, 2]
    c.advance(3)
    k = torch.arange(3 * 2 * 16, dtype=torch.float64).view(3, 2, 16)
    c.write(1, k, -k)
    kr, vr = c.read(1)
    assert kr.shape == (2, 3, 16) and torch.equal(kr, k.transpose(0, 1)) and torch.equal(vr, -k.transpose(0, 1))
    assert kr.data_ptr() == c.k[1].data_ptr()  # a view, not a copy
    assert c.next_positions(2).tolist() == [3, 4] and c.positions().tolist() == [0, 1, 2]
    c.advance(5)
    with pytest.raises(ValueError, match="capacity"):
        c.next_positions(1)
    c.reset()
    assert c.length == 0 and c.next_positions(1).tolist() == [0]


def test_chunked_prefill_matches_one_shot(tiny):
    """Prefilling in chunks of 1, 3, 7 through the cache gives the same logits as one uncached forward."""
    ids = torch.randint(0, TINY.vocab_size, (23,), generator=torch.Generator().manual_seed(1))
    full = tiny.forward(ids, torch.arange(23))
    for chunk in (1, 3, 7):
        cache = ContiguousKVCache(TINY, 32, "cpu", torch.float64)
        parts = [tiny.forward(ids[s:s + chunk], cache.next_positions(len(ids[s:s + chunk])), cache)
                 for s in range(0, 23, chunk)]
        torch.testing.assert_close(torch.cat(parts), full, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("prompt_len", [1, 2, 17])
def test_cached_equals_uncached_tiny(tiny, prompt_len):
    """Invariant 2 on the miniature model, including a 1-token prompt."""
    prompt = torch.randint(0, TINY.vocab_size, (prompt_len,), generator=torch.Generator().manual_seed(2)).tolist()
    assert greedy_generate(tiny, prompt, 30, use_cache=True) == greedy_generate(tiny, prompt, 30, use_cache=False)


def test_fusion_releases_originals():
    """Building the model consumes the per-projection tensors, so fused copies don't double the memory."""
    w = random_weights(TINY)
    Qwen2Model(TINY, w)
    assert set(w) == {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}


def test_config_rejects_unsupported(tmp_path):
    """Sliding windows and RoPE scaling are refused rather than silently ignored."""
    import json
    base = {"model_type": "qwen2", "vocab_size": 10, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
            "rope_theta": 1e4, "max_position_embeddings": 32}
    for extra in ({"use_sliding_window": True}, {"rope_scaling": {"type": "yarn", "factor": 4}}):
        (tmp_path / "config.json").write_text(json.dumps(base | extra))
        with pytest.raises(ValueError):
            ModelConfig.from_json(tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps(base))
    assert ModelConfig.from_json(tmp_path / "config.json").head_dim == 4
