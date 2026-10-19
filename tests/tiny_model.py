"""A miniature random-weight Qwen2 for fast CPU tests that need neither the GPU nor the checkpoint."""

import torch

from engine.config import ModelConfig

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
