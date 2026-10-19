"""Triton paged-attention decode: one new query token per sequence, K/V read in place from the paged pool.

Shapes for Qwen2.5-0.5B: q [S, 14, 64]; per layer the pool is K, V [2, N, B, 64]; g = 7 query heads share a KV head.

Single pass (grid S x H_kv): a program owns one (sequence, KV head). It loads that group's 7 query rows as a
[16, 64] tile (zero rows pad g to 16, the minimum tl.dot size), then walks the context in TILE-token tiles:
  block ids for the tile's tokens <- block table (one lookup per token, so TILE need not be a multiple of B)
  K [TILE, 64] <- pool, in place                        scores [16, TILE] = q K^T * scale   (fp32 accumulate)
  online softmax: m_new = max(m, rowmax(scores)); alpha = exp(m - m_new); p = exp(scores - m_new)
  l = l * alpha + rowsum(p);  acc = acc * alpha + p V   with V [TILE, 64] <- pool, in place
and writes acc / l. Each KV head's cache is read once for all g query heads, which is why decode attention has
arithmetic intensity about g: 7 query heads x 2 FLOPs x 2 (QK and PV) per K/V element of 2 B.

Precision: q K^T takes bf16 x bf16 products, which are exact in fp32, accumulated in fp32. P stays fp32 for P V (V is
upcast and the dot runs as true fp32, input_precision="ieee"): rounding P to bf16 is what put HF eager and the Phase 1
reference 2 points of argmax agreement below SDPA. With fp32 caches both dots are IEEE fp32.

Split-K (grid S x H_kv x splits, the Flash-Decoding idea): at batch 1 the single pass launches only 2 programs on a
20-SM GPU, so a long context is read by 2 SMs. Splitting the context into chunks gives each split's program a
partial (m, l, unnormalized acc); a combine kernel merges them per query head:
  M = max_s m_s,  out = sum_s exp(m_s - M) acc_s / sum_s exp(m_s - M) l_s.
A split past a sequence's end (ragged batches) stores m = -inf, l = 0, acc = 0 and drops out of the combine.

INT8 caches (Phase 5, engine/quant_cache.py) reuse the same loop with dequantization fused in. Per-token K: the int8
tile converts exactly to the compute dtype for the tensor-core dot and the per-token scale multiplies the [16, TILE]
scores, so score_t = (q . k_int_t) * s_t. Per-token V: the scale folds into P, acc += (p * s_v) V_int. Per-channel K:
the [TILE, 64] tile is dequantized elementwise with its block's channel scales, and the sequence's partial last block
comes from a bf16 staging slot; the dot then runs in fp32.
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

TILE = 64  # tokens per inner-loop iteration
TARGET_PROGRAMS = 40  # about 2 per SM on the 20-SM RTX 3050 Laptop
MIN_SPLIT_TOKENS = 256  # smallest split
# Measured (results/decode_sweep/20261019-1853*, 0.5B, block 16, 3 runs): a split call costs ~80 µs against ~43 µs for
# a single pass (one more launch, the combine, three allocations), and single-pass time grows ~33 µs per 1k tokens at
# <= 8 programs. Split-K lost at <= 2,048 tokens (1.11-1.87x slower at batch 1-4), won at 4,096 (0.58-0.65x) and
# 8,192 (0.32-0.46x), and never helped at batch 8 (16 programs, already ~127 GB/s at 8k).
SPLIT_MAX_PROGRAMS = 8  # batch <= 4 with 2 KV heads
SPLIT_MIN_CONTEXT = 3072  # interpolated crossover between the measured 2,048 (loss) and 4,096 (win)


@triton.jit
def _decode_kernel(q_ptr, k_ptr, v_ptr, bt_ptr, ctx_ptr, o_ptr, m_ptr, l_ptr, scale,
                   stride_qs, stride_qh, stride_kh, stride_kn, stride_kb, stride_bts,
                   stride_os, stride_oh, stride_osp, stride_ms, stride_mh, chunk,
                   ks_ptr, vs_ptr, stage_ptr, tail_ptr, stride_ksh, stride_ksn, stride_vsh, stride_vsn,
                   stride_sth, stride_stn, stride_stb,
                   G: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
                   TILE: tl.constexpr, SPLIT: tl.constexpr, FP32: tl.constexpr, KV_MODE: tl.constexpr):
    """One (sequence, KV head[, split]) program; see the module docstring. KV_MODE 0: K/V in the compute dtype;
    1: INT8 with per-token fp16 scales for K and V; 2: INT8 K with per-(block, channel) scales, the partial last
    block's K in a bf16 staging slot, and per-token INT8 V."""
    s = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    ctx = tl.load(ctx_ptr + s)
    start = sp * chunk
    end = tl.minimum(start + chunk, ctx)

    g = tl.arange(0, G_PAD)
    d = tl.arange(0, D)
    rows = h * G + g
    gmask = g < G
    q = tl.load(q_ptr + s * stride_qs + rows[:, None] * stride_qh + d[None, :], mask=gmask[:, None], other=0.0)

    m_i = tl.full([G_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([G_PAD], dtype=tl.float32)
    acc = tl.zeros([G_PAD, D], dtype=tl.float32)
    kv_head = h * stride_kh
    n_quant = (ctx // BLOCK) * BLOCK  # KV_MODE 2: tokens in full (quantized) blocks; the rest are staged in bf16
    tail = tl.load(tail_ptr + s) if KV_MODE == 2 else 0
    for t0 in range(start, end, TILE):
        t = t0 + tl.arange(0, TILE)
        tmask = t < end
        blk = tl.load(bt_ptr + s * stride_bts + t // BLOCK, mask=tmask, other=0).to(tl.int64)
        off = kv_head + blk * stride_kn + (t % BLOCK) * stride_kb
        if KV_MODE == 0:
            k = tl.load(k_ptr + off[:, None] + d[None, :], mask=tmask[:, None], other=0.0)
            if FP32:
                scores = tl.dot(q, tl.trans(k), input_precision="ieee")
            else:
                scores = tl.dot(q, tl.trans(k))
        elif KV_MODE == 1:
            # int8 -> compute dtype is exact (|x| <= 127); the per-token scale multiplies the score column
            k = tl.load(k_ptr + off[:, None] + d[None, :], mask=tmask[:, None], other=0).to(q.dtype)
            if FP32:
                scores = tl.dot(q, tl.trans(k), input_precision="ieee")
            else:
                scores = tl.dot(q, tl.trans(k))
            ks = tl.load(ks_ptr + h * stride_ksh + blk * stride_ksn + t % BLOCK, mask=tmask, other=0.0)
            scores = scores * ks.to(tl.float32)[None, :]
        else:
            is_q = t < n_quant
            kq = tl.load(k_ptr + off[:, None] + d[None, :], mask=(tmask & is_q)[:, None], other=0).to(tl.float32)
            ksc = tl.load(ks_ptr + h * stride_ksh + blk[:, None] * stride_ksn + d[None, :],
                          mask=(tmask & is_q)[:, None], other=0.0).to(tl.float32)
            kt = tl.load(stage_ptr + h * stride_sth + tail * stride_stn + (t % BLOCK)[:, None] * stride_stb
                         + d[None, :], mask=(tmask & (t >= n_quant))[:, None], other=0.0).to(tl.float32)
            k = tl.where(is_q[:, None], kq * ksc, kt)
            scores = tl.dot(q.to(tl.float32), tl.trans(k), input_precision="ieee")
        scores = tl.where(tmask[None, :], scores * scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if KV_MODE == 0:
            v = tl.load(v_ptr + off[:, None] + d[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        else:
            # per-token V scale folded into P: sum_t p_t s_t v_int_t; l keeps the unscaled p
            v = tl.load(v_ptr + off[:, None] + d[None, :], mask=tmask[:, None], other=0).to(tl.float32)
            vs = tl.load(vs_ptr + h * stride_vsh + blk * stride_vsn + t % BLOCK, mask=tmask, other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p * vs.to(tl.float32)[None, :], v, input_precision="ieee")
        m_i = m_new

    if SPLIT:
        o = o_ptr + s * stride_os + rows[:, None] * stride_oh + sp * stride_osp + d[None, :]
        tl.store(o, acc, mask=gmask[:, None])
        tl.store(m_ptr + s * stride_ms + rows * stride_mh + sp, m_i, mask=gmask)
        tl.store(l_ptr + s * stride_ms + rows * stride_mh + sp, l_i, mask=gmask)
    else:
        out = acc / l_i[:, None]
        o = o_ptr + s * stride_os + rows[:, None] * stride_oh + d[None, :]
        tl.store(o, out.to(o_ptr.dtype.element_ty), mask=gmask[:, None])


@triton.jit
def _combine_kernel(part_ptr, m_ptr, l_ptr, o_ptr, n_splits,
                    stride_ps, stride_ph, stride_psp, stride_ms, stride_mh, stride_os, stride_oh,
                    D: tl.constexpr, SPLITS_PAD: tl.constexpr):
    """Merge the splits' partial softmax states for one (sequence, query head)."""
    s = tl.program_id(0)
    hq = tl.program_id(1)
    sp = tl.arange(0, SPLITS_PAD)
    d = tl.arange(0, D)
    smask = sp < n_splits
    m = tl.load(m_ptr + s * stride_ms + hq * stride_mh + sp, mask=smask, other=float("-inf"))
    l_s = tl.load(l_ptr + s * stride_ms + hq * stride_mh + sp, mask=smask, other=0.0)
    big_m = tl.max(m, 0)  # finite: split 0 always holds at least one token
    w = tl.where(smask, tl.exp(m - big_m), 0.0)
    part = tl.load(part_ptr + s * stride_ps + hq * stride_ph + sp[:, None] * stride_psp + d[None, :],
                   mask=smask[:, None], other=0.0)
    out = tl.sum(w[:, None] * part, 0) / tl.sum(w * l_s, 0)
    tl.store(o_ptr + s * stride_os + hq * stride_oh + d, out.to(o_ptr.dtype.element_ty))


def choose_splits(num_seqs: int, num_kv_heads: int, max_context: int) -> int:
    """Splits per sequence. Split only when the grid is too small to saturate bandwidth (<= 8 programs) and the
    context is long enough that a single pass costs more than a split call's ~80 µs floor (>= 3,072 tokens); then aim
    for about 40 programs with splits of at least 256 tokens. Batch 1 at 4,096 tokens: 16 splits; batch 1 at 2,048,
    or batch 8 at any length: no split."""
    programs = num_seqs * num_kv_heads
    if programs > SPLIT_MAX_PROGRAMS or max_context < SPLIT_MIN_CONTEXT:
        return 1
    return max(1, min(math.ceil(TARGET_PROGRAMS / programs), math.ceil(max_context / MIN_SPLIT_TOKENS)))


@dataclass
class QuantKV:
    """INT8 cache side inputs for one layer. mode 1: k_scale, v_scale [Hkv, N, B] fp16 (per token). mode 2: k_scale
    [Hkv, N, D] fp16 (per block and channel), v_scale [Hkv, N, B], stage [Hkv, n_stage, B, D] bf16 holding each
    sequence's partial last block of K, tail_slots [S] int32 (the sequence's staging slot, -1 if its last block is
    full)."""
    mode: int
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    stage: torch.Tensor | None = None
    tail_slots: torch.Tensor | None = None


def paged_decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_tables: torch.Tensor,
                           context_lens: torch.Tensor, scale: float, max_context: int,
                           num_splits: int | None = None, quant: QuantKV | None = None) -> torch.Tensor:
    """Attention of one query token per sequence over its paged context.

    q [S, Hq, D]; k_cache, v_cache [Hkv, N, B, D] (one layer of the pool, contiguous; int8 when `quant` is given);
    block_tables [S, max_blocks] int32; context_lens [S] int32 (each >= 1, the new token included); max_context =
    max(context_lens), known on the CPU so choosing splits needs no sync. num_splits None picks with choose_splits().
    Returns [S, Hq, D] in q's dtype.
    """
    S, Hq, D = q.shape
    Hkv, _, B, _ = k_cache.shape
    G = Hq // Hkv
    assert q.stride(-1) == 1 and k_cache.is_contiguous() and v_cache.is_contiguous() and block_tables.stride(-1) == 1
    splits = num_splits or choose_splits(S, Hkv, max_context)
    chunk = triton.cdiv(triton.cdiv(max_context, splits), TILE) * TILE
    splits = min(splits, triton.cdiv(max_context, chunk))  # no split that is empty for every sequence
    out = torch.empty(S, Hq, D, device=q.device, dtype=q.dtype)
    common = dict(G=G, G_PAD=max(16, triton.next_power_of_2(G)), D=D, BLOCK=B, TILE=TILE,
                  FP32=q.dtype == torch.float32, KV_MODE=quant.mode if quant else 0)
    if quant is None:
        qargs = (q, q, q, context_lens, 0, 0, 0, 0, 0, 0, 0)  # unused pointers and strides
    else:
        st = quant.stage if quant.stage is not None else q
        tails = quant.tail_slots if quant.tail_slots is not None else context_lens
        st_strides = (quant.stage.stride(0), quant.stage.stride(1), quant.stage.stride(2)) if quant.stage is not None \
            else (0, 0, 0)
        qargs = (quant.k_scale, quant.v_scale, st, tails, quant.k_scale.stride(0), quant.k_scale.stride(1),
                 quant.v_scale.stride(0), quant.v_scale.stride(1), *st_strides)
    if splits == 1:
        _decode_kernel[(S, Hkv, 1)](
            q, k_cache, v_cache, block_tables, context_lens, out, out, out, scale,
            q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), block_tables.stride(0),
            out.stride(0), out.stride(1), 0, 0, 0, max_context, *qargs, SPLIT=False, **common)
        return out
    part = torch.empty(S, Hq, splits, D, device=q.device, dtype=torch.float32)
    m = torch.empty(S, Hq, splits, device=q.device, dtype=torch.float32)
    l_s = torch.empty_like(m)
    _decode_kernel[(S, Hkv, splits)](
        q, k_cache, v_cache, block_tables, context_lens, part, m, l_s, scale,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), block_tables.stride(0),
        part.stride(0), part.stride(1), part.stride(2), m.stride(0), m.stride(1), chunk, *qargs, SPLIT=True,
        **common)
    _combine_kernel[(S, Hq)](
        part, m, l_s, out, splits, part.stride(0), part.stride(1), part.stride(2), m.stride(0), m.stride(1),
        out.stride(0), out.stride(1), D=D, SPLITS_PAD=triton.next_power_of_2(splits))
    return out
