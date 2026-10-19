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
"""

import math

import torch
import triton
import triton.language as tl

TILE = 64  # tokens per inner-loop iteration
TARGET_PROGRAMS = 40  # about 2 per SM on the 20-SM RTX 3050 Laptop
MIN_SPLIT_TOKENS = 256  # below this a split's fixed cost outweighs the parallelism


@triton.jit
def _decode_kernel(q_ptr, k_ptr, v_ptr, bt_ptr, ctx_ptr, o_ptr, m_ptr, l_ptr, scale,
                   stride_qs, stride_qh, stride_kh, stride_kn, stride_kb, stride_bts,
                   stride_os, stride_oh, stride_osp, stride_ms, stride_mh, chunk,
                   G: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
                   TILE: tl.constexpr, SPLIT: tl.constexpr, FP32: tl.constexpr):
    """One (sequence, KV head[, split]) program; see the module docstring."""
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
    for t0 in range(start, end, TILE):
        t = t0 + tl.arange(0, TILE)
        tmask = t < end
        blk = tl.load(bt_ptr + s * stride_bts + t // BLOCK, mask=tmask, other=0).to(tl.int64)
        off = kv_head + blk * stride_kn + (t % BLOCK) * stride_kb
        k = tl.load(k_ptr + off[:, None] + d[None, :], mask=tmask[:, None], other=0.0)
        if FP32:
            scores = tl.dot(q, tl.trans(k), input_precision="ieee")
        else:
            scores = tl.dot(q, tl.trans(k))
        scores = tl.where(tmask[None, :], scores * scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + off[:, None] + d[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
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
    """Splits per sequence: enough programs to cover the SMs about twice, but no split under MIN_SPLIT_TOKENS.
    Batch 1 at 4,096 tokens: 2 programs -> 16 splits of 256 tokens = 32 programs. Batch 32: 1 split."""
    programs = num_seqs * num_kv_heads
    if programs >= TARGET_PROGRAMS:
        return 1
    return max(1, min(math.ceil(TARGET_PROGRAMS / programs), math.ceil(max_context / MIN_SPLIT_TOKENS)))


def paged_decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, block_tables: torch.Tensor,
                           context_lens: torch.Tensor, scale: float, max_context: int,
                           num_splits: int | None = None) -> torch.Tensor:
    """Attention of one query token per sequence over its paged context.

    q [S, Hq, D]; k_cache, v_cache [Hkv, N, B, D] (one layer of the pool, contiguous); block_tables [S, max_blocks]
    int32; context_lens [S] int32 (each >= 1, the new token included); max_context = max(context_lens), known on the
    CPU so choosing splits needs no sync. num_splits None picks with choose_splits(). Returns [S, Hq, D] in q's dtype.
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
                  FP32=q.dtype == torch.float32)
    if splits == 1:
        _decode_kernel[(S, Hkv, 1)](
            q, k_cache, v_cache, block_tables, context_lens, out, out, out, scale,
            q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), block_tables.stride(0),
            out.stride(0), out.stride(1), 0, 0, 0, max_context, SPLIT=False, **common)
        return out
    part = torch.empty(S, Hq, splits, D, device=q.device, dtype=torch.float32)
    m = torch.empty(S, Hq, splits, device=q.device, dtype=torch.float32)
    l_s = torch.empty_like(m)
    _decode_kernel[(S, Hkv, splits)](
        q, k_cache, v_cache, block_tables, context_lens, part, m, l_s, scale,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), block_tables.stride(0),
        part.stride(0), part.stride(1), part.stride(2), m.stride(0), m.stride(1), chunk, SPLIT=True, **common)
    _combine_kernel[(S, Hq)](
        part, m, l_s, out, splits, part.stride(0), part.stride(1), part.stride(2), m.stride(0), m.stride(1),
        out.stride(0), out.stride(1), D=D, SPLITS_PAD=triton.next_power_of_2(splits))
    return out
