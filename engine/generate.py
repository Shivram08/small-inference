"""Greedy generation for one sequence, with and without the KV cache. EOS is ignored: exactly n tokens come out."""

import torch

from engine.block_manager import BlockManager
from engine.kv_cache import ContiguousKVCache
from engine.model import Qwen2Model
from engine.paged_cache import PagedKVCache, build_metadata


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


@torch.inference_mode()
def greedy_generate_paged(model: Qwen2Model, prompt_ids: list[int], n: int, manager: BlockManager | None = None,
                          cache: PagedKVCache | None = None, seq_id: int = 0, block_size: int = 16,
                          prefill_chunk: int | None = None) -> list[int]:
    """The n greedy tokens after prompt_ids, through the paged cache.

    Pass a manager and cache to run inside an existing (e.g. fragmented) pool; otherwise a pool just big enough is
    made. The prompt is prefilled in chunks of prefill_chunk tokens (default: all at once). The sequence's blocks
    are freed at the end.
    """
    if not prompt_ids or n < 1:
        raise ValueError("need a non-empty prompt and n >= 1")
    if manager is None:
        manager = BlockManager(-(-(len(prompt_ids) + n) // block_size), block_size)
        cache = PagedKVCache(model.cfg, manager.num_blocks, block_size, model.device, model.dtype)
    dev = model.device
    manager.allocate(seq_id, 0)
    try:
        step = prefill_chunk or len(prompt_ids)
        for start in range(0, len(prompt_ids), step):
            chunk = prompt_ids[start:start + step]
            manager.append_slots(seq_id, len(chunk))
            meta = build_metadata(manager, [(seq_id, start, len(chunk))], dev)
            logits = model.forward(torch.tensor(chunk, device=dev), meta.positions, cache, last_only=True, meta=meta)
        out = [int(logits[0].argmax())]
        for _ in range(n - 1):
            pos = manager.seq_len(seq_id)
            manager.append_slot(seq_id)
            meta = build_metadata(manager, [(seq_id, pos, 1)], dev)
            logits = model.forward(torch.tensor(out[-1:], device=dev), meta.positions, cache, last_only=True,
                                   meta=meta)
            out.append(int(logits[0].argmax()))
        return out
    finally:
        manager.free(seq_id)
