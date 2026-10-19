"""INT8 KV quantization: reference quantize/dequantize for the three Phase 5 cache formats.

Symmetric absmax INT8 with fp16 scales. The scale is rounded to fp16 first and the values are quantized with that
rounded scale, so dequantization (q * scale) uses exactly the stored scale. Values are in [-127, 127]; -128 is never
produced, so the grid is symmetric. A group that is all zeros gets scale 0 and quantizes to zeros.

  per token:    one scale per (token, head) over the 64 (or 128) channels. K and V in format A, V in format B.
  per channel:  one scale per (block, head, channel) over the block's 16 tokens. K in format B. Keys have a few
                outlier channels with large magnitude in every token (KIVI, KVQuant); per-token scales let those
                channels set the scale for all the others, per-channel scales isolate them.

Bytes per token for 0.5B (24 layers, 2 KV heads, d = 64): bf16 12,288; format A 2 x 24 x 2 x (64 + 2) = 6,336;
format B 24 x 2 x (64 + 2) for V + 24 x 2 x (64 + 64 x 2 / 16) for K = 3,168 + 3,456 = 6,624 for full blocks, plus a
bf16 partial block per sequence.
"""

import torch

QMAX = 127


def _scale(absmax: torch.Tensor) -> torch.Tensor:
    """fp16 scale for an absmax: absmax / 127, rounded to fp16."""
    return (absmax.float() / QMAX).to(torch.float16)


def _quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """round(x / scale) clamped to [-127, 127] as int8; scale broadcasts against x. Zero scales give zeros."""
    s = scale.float()
    q = torch.where(s > 0, torch.round(x.float() / torch.where(s > 0, s, 1.0)), 0.0)
    return q.clamp(-QMAX, QMAX).to(torch.int8)


def quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., D] -> (int8 [..., D], fp16 scale [...]): one scale per vector over its last dimension."""
    scale = _scale(x.abs().amax(dim=-1))
    return _quantize(x, scale[..., None]), scale


def quantize_per_channel(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """x quantized with one scale per position of every dimension except `dim` (the tokens of a block)."""
    scale = _scale(x.abs().amax(dim=dim, keepdim=True))
    return _quantize(x, scale), scale.squeeze(dim)


def dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """q * scale in `dtype`; scale must already broadcast against q."""
    return (q.float() * scale.float()).to(dtype)


def relative_error(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """||x - x_hat|| / ||x|| over the whole tensor."""
    return ((x.float() - x_hat.float()).norm() / x.float().norm()).item()
