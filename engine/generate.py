"""Greedy generation for one sequence, with and without the KV cache. EOS is ignored: exactly n tokens come out."""

import torch

from engine.kv_cache import ContiguousKVCache
from engine.model import Qwen2Model


@torch.inference_mode()
def greedy_generate(model: Qwen2Model, prompt_ids: list[int], n: int, use_cache: bool = True) -> list[int]:
    """The n greedy tokens after prompt_ids.

    Cached: one prefill over the prompt, then n - 1 single-token decode steps reading the cache.
    Uncached: n full forwards over the growing sequence, recomputing every K/V each time (the reference).
    """
    if not prompt_ids or n < 1:
        raise ValueError("need a non-empty prompt and n >= 1")
    dev = model.device
    if not use_cache:
        seq = torch.tensor(prompt_ids, device=dev)
        out = []
        for _ in range(n):
            logits = model.forward(seq, torch.arange(seq.numel(), device=dev), last_only=True)
            tok = logits[0].argmax()
            out.append(int(tok))
            seq = torch.cat([seq, tok[None]])
        return out

    cache = ContiguousKVCache(model.cfg, len(prompt_ids) + n, dev, model.dtype)
    ids = torch.tensor(prompt_ids, device=dev)
    out = []
    for _ in range(n):
        logits = model.forward(ids, cache.next_positions(ids.numel()), cache, last_only=True)
        ids = logits[0].argmax()[None]
        out.append(int(ids))
    return out
