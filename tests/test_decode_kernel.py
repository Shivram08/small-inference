"""Phase 3 invariant 5: the Triton paged-decode kernel equals a reference on every edge case.

Reference: gather each sequence's blocks and compute softmax(q K^T * scale) V in fp64. Cases: shuffled block tables,
ragged context lengths within a batch (1, 15, 16, 17, 33, 100, 1000), block sizes 8 / 16 / 32, both model head
layouts (0.5B: 14 q / 2 kv / d 64; 1.5B: 12 / 2 / 128), bf16 and fp32, single pass and split-K (including more
splits than a short sequence has tokens). Each kernel variant compiles once (cached in ~/.triton); the first run
of this file takes about a minute, later runs seconds. GPU memory: tens of MB.
"""

import pytest
import torch

from engine.block_manager import BlockManager
from engine.triton_decode import choose_splits, paged_decode_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
RAGGED = [1, 15, 16, 17, 33, 100, 1000]
HEADS = {"0.5B": (14, 2, 64), "1.5B": (12, 2, 128)}


def make_case(lengths, Hq, Hkv, D, B, dtype, seed=0):
    """Random q and pool, with each sequence's blocks scattered through a shuffled pool."""
    g = torch.Generator().manual_seed(seed)
    need = sum(-(-n // B) for n in lengths)
    m = BlockManager(need * 2, B)
    for s in range(m.num_blocks):
        m.allocate(10_000 + s, B)
    for s in torch.randperm(m.num_blocks, generator=g).tolist():
        m.free(10_000 + s)
    for i, n in enumerate(lengths):
        m.allocate(i, n)
    tables = [m.block_table(i) for i in range(len(lengths))]
    width = max(len(t) for t in tables)
    bt = torch.tensor([t + [-1] * (width - len(t)) for t in tables], dtype=torch.int32, device="cuda")
    k = torch.randn(Hkv, m.num_blocks, B, D, generator=g).to("cuda", dtype)
    v = torch.randn(Hkv, m.num_blocks, B, D, generator=g).to("cuda", dtype)
    q = torch.randn(len(lengths), Hq, D, generator=g).to("cuda", dtype)
    ctx = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return q, k, v, bt, ctx, tables


def reference(q, k, v, tables, lengths, scale):
    """fp64 attention per sequence over its gathered blocks; query head h uses KV head h // g."""
    S, Hq, D = q.shape
    Hkv = k.shape[0]
    out = torch.empty(S, Hq, D, dtype=torch.float64, device=q.device)
    for i, n in enumerate(lengths):
        blocks = torch.tensor(tables[i], device=q.device)
        ki = k[:, blocks].reshape(Hkv, -1, D)[:, :n].double().repeat_interleave(Hq // Hkv, 0)  # [Hq, n, D]
        vi = v[:, blocks].reshape(Hkv, -1, D)[:, :n].double().repeat_interleave(Hq // Hkv, 0)
        p = torch.softmax((q[i].double()[:, None, :] @ ki.transpose(1, 2))[:, 0] * scale, dim=-1)  # [Hq, n]
        out[i] = (p[:, None, :] @ vi)[:, 0]
    return out


def check(lengths, heads, B, dtype, splits):
    """Run the kernel and compare against the fp64 reference."""
    Hq, Hkv, D = HEADS[heads]
    q, k, v, bt, ctx, tables = make_case(lengths, Hq, Hkv, D, B, dtype)
    scale = D ** -0.5
    out = paged_decode_attention(q, k, v, bt, ctx, scale, max(lengths), num_splits=splits)
    ref = reference(q, k, v, tables, lengths, scale)
    tol = 1e-5 if dtype == torch.float32 else 1e-2  # bf16 output rounding is 2^-9 relative
    torch.testing.assert_close(out.double(), ref, rtol=tol, atol=tol)
    return out, ref


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("heads", list(HEADS))
@pytest.mark.parametrize("B", [8, 16, 32])
@pytest.mark.parametrize("splits", [1, 3, None], ids=["single", "split3", "auto"])
def test_ragged_batch_matches_reference(dtype, heads, B, splits):
    """Invariant 5: a ragged batch with lengths 1 through 1000 over shuffled tables."""
    check(RAGGED, heads, B, dtype, splits)


@pytest.mark.parametrize("n", [1, 2, 63, 64, 65, 4097])
def test_single_sequence_lengths(n):
    """Batch 1 at tile and split boundaries (TILE = 64), single pass and auto split."""
    for splits in (1, None):
        check([n], "0.5B", 16, torch.bfloat16, splits)


def test_more_splits_than_tokens():
    """Splits that start past a short sequence's end contribute nothing (m = -inf, l = 0)."""
    check([1, 3, 2000], "0.5B", 16, torch.float32, 16)


def test_bf16_error_is_output_rounding():
    """In bf16 the kernel's error against fp64 is about output rounding (P stays fp32), not P rounding."""
    out, ref = check(RAGGED, "0.5B", 16, torch.bfloat16, 1)
    rel = ((out.double() - ref).abs() / ref.abs().clamp_min(1e-3)).median().item()
    print(f"\nbf16 kernel median relative error vs fp64: {rel:.1e} (bf16 half-ulp: {2 ** -9:.1e})")
    assert rel < 2 ** -8


def test_choose_splits():
    """The measured policy: split only for small grids (<= 8 programs) at long contexts (>= 3,072 tokens)."""
    assert choose_splits(1, 2, 4096) == 16
    assert choose_splits(1, 2, 8192) == 20
    assert choose_splits(4, 2, 4096) == 5
    assert choose_splits(1, 2, 2048) == 1  # split-K measured 1.26x slower here
    assert choose_splits(1, 2, 300) == 1
    assert choose_splits(8, 2, 8192) == 1  # 16 programs: never helped
    assert choose_splits(32, 2, 4096) == 1
