"""Fused INT8 quantize-and-store kernels for the Phase 5 caches: one launch where the reference uses ~10 torch ops.

Batch-1 decode is launch-bound (~12-20 µs per launch here), and the reference write path (engine/quant.py: abs,
amax, divide, fp16 cast, divide, round, clamp, int8 cast, then two scattered stores) ran once for K and once for V
in every layer: several hundred extra launches per step, which doubled INT8 TPOT at low load on Qwen2.5-1.5B.

Both kernels reproduce engine/quant.py bit for bit, so switching changes speed only:
  scale = fp16(absmax / 127)            (fp32 divide, then round to fp16)
  q = clamp(rint(x / fp32(scale)), -127, 127)   with correctly rounded division (div_rn) and round-half-even (rint),
                                                 as torch.round does; a zero scale gives zeros.

quantize_store_tokens: grid (T, H). Program (t, h) quantizes the d-vector x[t, h] and writes it and its scale to the
pool at the token's (block, offset). Per-token K and V (format A), and V in format B.
quantize_store_blocks: grid (n, H). Program (i, h) quantizes a [B, d] tile (one block's 16 tokens of K, from
staging) with one scale per channel and writes the int8 block and its d channel scales. K in format B.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _tokens_kernel(x_ptr, blk_ptr, off_ptr, q_ptr, s_ptr, stride_xt, stride_xh,
                   stride_qh, stride_qn, stride_qb, stride_sh, stride_sn, D: tl.constexpr):
    """Quantize x[t, h, :] per token and store it at its slot."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    x = tl.load(x_ptr + t * stride_xt + h * stride_xh + d).to(tl.float32)
    scale = (tl.max(tl.abs(x), 0) / 127.0).to(tl.float16)
    s32 = scale.to(tl.float32)
    q = tl.where(s32 > 0, libdevice.rint(libdevice.div_rn(x, tl.where(s32 > 0, s32, 1.0))), 0.0)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    blk = tl.load(blk_ptr + t).to(tl.int64)
    off = tl.load(off_ptr + t)
    tl.store(q_ptr + h * stride_qh + blk * stride_qn + off * stride_qb + d, q)
    tl.store(s_ptr + h * stride_sh + blk * stride_sn + off, scale)


@triton.jit
def _blocks_kernel(x_ptr, src_ptr, dst_ptr, q_ptr, s_ptr, stride_xh, stride_xn, stride_xb,
                   stride_qh, stride_qn, stride_qb, stride_sh, stride_sn, B: tl.constexpr, D: tl.constexpr):
    """Quantize the [B, D] tile x[h, src[i]] with one scale per channel and store it as block dst[i]."""
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.arange(0, B)
    d = tl.arange(0, D)
    src = tl.load(src_ptr + i).to(tl.int64)
    dst = tl.load(dst_ptr + i).to(tl.int64)
    x = tl.load(x_ptr + h * stride_xh + src * stride_xn + b[:, None] * stride_xb + d[None, :]).to(tl.float32)
    scale = (tl.max(tl.abs(x), 0) / 127.0).to(tl.float16)  # [D]
    s32 = scale.to(tl.float32)[None, :]
    q = tl.where(s32 > 0, libdevice.rint(libdevice.div_rn(x, tl.where(s32 > 0, s32, 1.0))), 0.0)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(q_ptr + h * stride_qh + dst * stride_qn + b[:, None] * stride_qb + d[None, :], q)
    tl.store(s_ptr + h * stride_sh + dst * stride_sn + d, scale)


def quantize_store_tokens(x: torch.Tensor, blocks: torch.Tensor, offsets: torch.Tensor, q_pool: torch.Tensor,
                          s_pool: torch.Tensor) -> None:
    """x [T, H, D] (rows may be strided; the last dim contiguous) -> q_pool [H, N, B, D] int8 and s_pool [H, N, B]
    fp16 at (blocks[t], offsets[t]). One launch."""
    T, H, D = x.shape
    if T == 0:
        return
    assert x.stride(-1) == 1 and q_pool.is_contiguous() and s_pool.is_contiguous()
    _tokens_kernel[(T, H)](x, blocks, offsets, q_pool, s_pool, x.stride(0), x.stride(1),
                           q_pool.stride(0), q_pool.stride(1), q_pool.stride(2), s_pool.stride(0), s_pool.stride(1),
                           D=D)


def quantize_store_blocks(src_pool: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, q_pool: torch.Tensor,
                          s_pool: torch.Tensor) -> None:
    """Blocks src_pool[:, src[i]] ([H, n_src, B, D], e.g. bf16 staging) -> q_pool[:, dst[i]] int8 with per-channel
    scales s_pool[:, dst[i]] ([H, N, D] fp16). One launch."""
    if src.numel() == 0:
        return
    H, _, B, D = src_pool.shape
    assert src_pool.is_contiguous() and q_pool.is_contiguous() and s_pool.is_contiguous()
    _blocks_kernel[(src.numel(), H)](src_pool, src, dst, q_pool, s_pool, src_pool.stride(0), src_pool.stride(1),
                                     src_pool.stride(2), q_pool.stride(0), q_pool.stride(1), q_pool.stride(2),
                                     s_pool.stride(0), s_pool.stride(1), B=B, D=D)
