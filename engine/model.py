"""Qwen2 forward pass in plain PyTorch: one sequence, tokens laid out flat as [T, ...], explicit positions.

Shapes below are for Qwen2.5-0.5B (hidden 896, 14 query heads and 2 KV heads of dim 64, so g = 7; MLP 4864).

Per layer, for T tokens:
  x [T, 896] -> RMSNorm -> fused QKV matmul [T, 896] x [896, 1152] (+ bias) -> q [T, 14, 64], k, v [T, 2, 64]
  -> RoPE on q and k at `positions` -> (cache write) -> attention over S keys -> o_proj [T, 896] -> residual
  -> RMSNorm -> fused gate/up matmul [T, 896] x [896, 9728] -> silu(gate) * up [T, 4864] -> down [T, 896] -> residual

Kernel economy matters because batch-1 decode is CPU-bound (HF: 1006 launches per step at ~11 µs each):
- Q/K/V and gate/up are fused matmuls: the same weight bytes, 3 fewer launches per layer.
- RoPE's cos/sin rows are gathered and cast once per forward, not per layer, and q and k are rotated together as one
  strided [T, 16, 64] view of the QKV output, in 4 elementwise kernels.
- Attention never copies K/V per query head. The g = 7 query heads sharing a KV head go into the matmul's M
  dimension: q becomes [2, 7T, 64] and multiplies K as [2, 64, S], so each KV head's cache is read once.
Keys are stored and attended to after RoPE, so positions are absolute.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from engine.config import ModelConfig
from engine.kv_cache import ContiguousKVCache


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x / rms(x) * weight, with the statistics in fp32 and the cast back before the weight multiply (as HF does)."""
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


class Rope:
    """Rotary embedding tables with HF's rotate_half pairing: dimension i rotates with i + d/2.

    Frequencies: inv_freq[i] = theta^(-2i/d) for i < d/2 (32 of them for d = 64). With theta = 1e6 they range from 1
    rad/token down to about 1.5e-6 rad/token. Tables are precomputed for every position with the same fp32 ops HF runs
    per forward (inv_freq on the CPU, the position product and cos/sin on the device), so they match it exactly.
    The sign of rotate_half is folded into the sin table: HF computes x * cos + cat(-x2, x1) * sin, this computes
    x * cos + cat(x2, x1) * cat(-sin, sin), and (-a) * b == a * (-b) exactly in IEEE arithmetic.
    """

    def __init__(self, cfg: ModelConfig, device: str | torch.device, max_positions: int | None = None):
        """fp32 tables [max_positions, d]: cos duplicated over both halves, and sin as (-sin, sin)."""
        d = cfg.head_dim
        n = max_positions or cfg.max_position_embeddings
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float) / d))
        freqs = torch.arange(n, device=device).float()[:, None] * inv_freq.to(device)[None, :]
        cos, sin = freqs.cos(), freqs.sin()
        self.cos, self.sin_signed = torch.cat((cos, cos), dim=-1), torch.cat((-sin, sin), dim=-1)

    def tables(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and signed sin rows for `positions` [T], as [T, 1, d] in `dtype`. Once per forward."""
        return self.cos[positions].to(dtype)[:, None, :], self.sin_signed[positions].to(dtype)[:, None, :]

    def __call__(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate x [T, heads, d] by its tokens' positions [T]."""
        return apply_rope(x, *self.tables(positions, x.dtype))


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin_signed: torch.Tensor) -> torch.Tensor:
    """x [T, heads, d] (may be a strided view) rotated by precomputed rows [T, 1, d]. 4 kernels: cat, mul, mul, add."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((x2, x1), dim=-1) * sin_signed


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor,
              scale: float) -> torch.Tensor:
    """Causal GQA attention for one sequence.

    q: [T, Hq, d]; k, v: [Hkv, S, d] (strided views of the cache are fine); q_pos [T] and k_pos [S] are absolute
    positions. Query t sees key s iff k_pos[s] <= q_pos[t]; a single query (T = 1) must be the last position, seeing
    all S keys. Query head h uses KV head h // g, matching HF's repeat_kv. Softmax in at least fp32.

    The g heads of a group are stacked along M: q [Hkv, g*T, d] x K^T [Hkv, d, S] -> scores [Hkv, g*T, S], which
    reads each KV head's cache once. A broadcast matmul against K as [Hkv, 1, d, S] looks equivalent but makes
    torch.matmul expand and copy K and V g times. Returns [T, Hq * d].
    """
    T, Hq, d = q.shape
    Hkv = k.shape[0]
    g = Hq // Hkv
    qg = q.reshape(T, Hkv, g, d).permute(1, 2, 0, 3).reshape(Hkv, g * T, d)  # copies q only: T x 896 elements
    scores = torch.bmm(qg, k.transpose(1, 2)) * scale  # [Hkv, g*T, S]
    if T > 1:  # a single query is always the newest token, which sees every key: skip 2 kernels per layer
        masked = (k_pos[None, :] > q_pos[:, None])  # [T, S]
        scores = scores.view(Hkv, g, T, -1).masked_fill(masked, float("-inf")).view(Hkv, g * T, -1)
    probs = torch.softmax(scores, dim=-1, dtype=torch.promote_types(q.dtype, torch.float32)).to(q.dtype)
    out = torch.bmm(probs, v)  # [Hkv, g*T, d]
    return out.view(Hkv, g, T, d).permute(2, 0, 1, 3).reshape(T, Hq * d)


@dataclass
class LayerWeights:
    input_norm: torch.Tensor  # [896]
    qkv_w: torch.Tensor  # [1152, 896]: q rows 0-895, k rows 896-1023, v rows 1024-1151
    qkv_b: torch.Tensor  # [1152]
    o_w: torch.Tensor  # [896, 896], no bias
    post_norm: torch.Tensor  # [896]
    gate_up_w: torch.Tensor  # [9728, 896]: gate rows 0-4863, up rows 4864-9727
    down_w: torch.Tensor  # [896, 4864]


class Qwen2Model:
    """Qwen2ForCausalLM inference: embeddings, decoder layers, final norm, (tied) LM head."""

    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor]):
        """Build from HF-named tensors (engine.loader.load_weights). Fused tensors are concatenated copies, and
        the separate originals are released from `weights` as they are fused so peak memory stays near one model."""
        self.cfg = cfg
        self.embed = weights["model.embed_tokens.weight"]  # [151936, 896]
        self.final_norm = weights["model.norm.weight"]
        self.lm_head = weights["lm_head.weight"]  # the same tensor as embed when tied
        self.device, self.dtype = self.embed.device, self.embed.dtype
        self.layers = []
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            take = lambda name: weights.pop(p + name)  # noqa: E731
            qkv_w = torch.cat([take(f"self_attn.{n}_proj.weight") for n in "qkv"])
            qkv_b = torch.cat([take(f"self_attn.{n}_proj.bias") for n in "qkv"])
            gate_up_w = torch.cat([take("mlp.gate_proj.weight"), take("mlp.up_proj.weight")])
            self.layers.append(LayerWeights(
                input_norm=take("input_layernorm.weight"), qkv_w=qkv_w, qkv_b=qkv_b,
                o_w=take("self_attn.o_proj.weight"), post_norm=take("post_attention_layernorm.weight"),
                gate_up_w=gate_up_w, down_w=take("mlp.down_proj.weight")))
        self.rope = Rope(cfg, self.device)
        self.scale = cfg.head_dim ** -0.5
        # False runs Q/K/V and gate/up as separate matmuls on row views of the fused weights (no extra memory). The
        # math is the same, but cuBLAS picks a different kernel and reduction order for N = 1152 than for N = 896,
        # so fused fp32 logits differ from HF by ~1 ulp per matmul (2.4e-4 at the logits), while unfused they are
        # bit-identical to HF. Tests use it to pin the forward pass exactly; the engine keeps fusion.
        self.fused_matmuls = True

    @classmethod
    def from_pretrained(cls, model_id: str | None = None, device: str = "cuda",
                        dtype: torch.dtype = torch.bfloat16) -> "Qwen2Model":
        """Load from the local HF cache."""
        from engine.loader import DEFAULT_MODEL, load_weights, snapshot_dir

        d = snapshot_dir(model_id or DEFAULT_MODEL)
        cfg = ModelConfig.from_json(d / "config.json")
        return cls(cfg, load_weights(d, cfg, device, dtype))

    def layer_forward(self, i: int, x: torch.Tensor, positions: torch.Tensor, cache: ContiguousKVCache | None,
                      rope_rows: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Decoder layer i on hidden states x [T, 896] at `positions` [T]. rope_rows = Rope.tables(positions),
        computed once per forward; computed here if not given."""
        cfg, w = self.cfg, self.layers[i]
        T, Hq, Hkv, d = x.shape[0], cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
        cos, sin = rope_rows if rope_rows is not None else self.rope.tables(positions, x.dtype)
        h = rms_norm(x, w.input_norm, cfg.rms_norm_eps)
        if self.fused_matmuls:
            qkv = F.linear(h, w.qkv_w, w.qkv_b)  # [T, 1152]
            qk = apply_rope(qkv[:, :cfg.q_size + cfg.kv_size].view(T, Hq + Hkv, d), cos, sin)  # [T, 16, 64]
            q, k = qk[:, :Hq], qk[:, Hq:]
            v = qkv[:, cfg.q_size + cfg.kv_size:].view(T, Hkv, d)
        else:
            sizes = [cfg.q_size, cfg.kv_size, cfg.kv_size]
            q, k, v = (F.linear(h, wt, b) for wt, b in zip(w.qkv_w.split(sizes), w.qkv_b.split(sizes)))
            q, k = apply_rope(q.view(T, Hq, d), cos, sin), apply_rope(k.view(T, Hkv, d), cos, sin)
            v = v.view(T, Hkv, d)
        if cache is None:
            k_all, v_all, k_pos = k.transpose(0, 1), v.transpose(0, 1), positions
        else:
            cache.write(i, k, v)
            k_all, v_all = cache.read(i)
            k_pos = cache.positions()
        x = x + F.linear(attention(q, k_all, v_all, positions, k_pos, self.scale), w.o_w)
        h = rms_norm(x, w.post_norm, cfg.rms_norm_eps)
        if self.fused_matmuls:
            gate, up = F.linear(h, w.gate_up_w).chunk(2, dim=-1)
        else:
            gate, up = (F.linear(h, wt) for wt in w.gate_up_w.chunk(2))
        return x + F.linear(F.silu(gate) * up, w.down_w)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, cache: ContiguousKVCache | None = None,
                last_only: bool = False) -> torch.Tensor:
        """Logits [T, vocab] (or [1, vocab] with last_only) in the model dtype for input_ids [T] at positions [T].

        Without a cache, the T tokens attend only to each other (causally by position). With a cache, positions
        must be cache.next_positions(T): their K/V are appended, and they attend to every cached position up to
        their own. last_only skips the LM head for all but the final token: [T, 896] x [896, 151936] reads 272 MB of
        weights either way, but the [T, 151936] output is 0.6 MB per token in fp32.
        """
        if cache is not None:
            cache.advance(input_ids.shape[0])
        x = self.embed[input_ids]
        rope_rows = self.rope.tables(positions, x.dtype)
        for i in range(self.cfg.num_layers):
            x = self.layer_forward(i, x, positions, cache, rope_rows)
        if last_only:
            x = x[-1:]
        return F.linear(rms_norm(x, self.final_norm, self.cfg.rms_norm_eps), self.lm_head)
