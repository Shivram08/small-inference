"""Reference causal GQA attention for one sequence, shared by the uncached path and every KV cache backend."""

import torch


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
