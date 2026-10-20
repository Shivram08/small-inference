"""Phase 4 serving engine: continuous batching over the paged KV cache with the Triton decode kernel.

Each step() is one forward pass over a packed batch planned by engine/scheduler.py: decode tokens of running
requests plus prefill chunks, up to max_num_batched_tokens. Greedy sampling, EOS ignored, exactly output_len tokens
per request. The step returns after the sampled tokens reach the CPU (one .tolist() for the whole batch).

Memory for 0.5B in bf16: weights 0.99 GB; the default 2 GiB KV pool is 10,922 blocks of 16 tokens (about 175k
tokens, e.g. 85 requests at the trace's 2,048-token maximum).
"""

import torch

from engine.block_manager import BlockManager
from engine.model import Qwen2Model
from engine.paged_cache import PagedKVCache, build_metadata
from engine.scheduler import Scheduler, SchedulerConfig

DEFAULT_KV_BYTES = 2 * 2**30


class ContinuousEngine:
    def __init__(self, model: Qwen2Model | None = None, kv_bytes: int = DEFAULT_KV_BYTES, block_size: int = 16,
                 num_blocks: int | None = None, config: SchedulerConfig | None = None, attn_backend: str = "triton",
                 kv_format: str = "bf16"):
        """Use `model` (default: 0.5B, bf16, GPU) with a pool of num_blocks blocks, or as many as fit in kv_bytes.
        kv_format: "bf16" (the model dtype), "int8" (per-token K/V), or "int8-kchannel" (per-channel K, per-token V);
        at the same kv_bytes the INT8 formats hold about twice the tokens."""
        from engine.quant_cache import QuantPagedKVCache, blocks_for_budget

        self.model = model or Qwen2Model.from_pretrained()
        cfg, dev, dt = self.model.cfg, self.model.device, self.model.dtype
        config = config or SchedulerConfig()
        self.kv_format = kv_format
        if kv_format == "bf16":
            elem = torch.tensor([], dtype=dt).element_size()
            n = num_blocks or PagedKVCache.blocks_for_budget(cfg, kv_bytes, block_size, elem)
            self.manager = BlockManager(n, block_size)
            self.cache = PagedKVCache(cfg, n, block_size, dev, dt, attn_backend)
        else:
            n_stage = 2 * config.max_num_seqs  # at most 2 partial blocks per sequence (engine/quant_cache.py)
            n = num_blocks or blocks_for_budget(cfg, kv_format, kv_bytes, block_size, n_stage)
            self.manager = BlockManager(n, block_size)
            self.cache = QuantPagedKVCache(cfg, n, block_size, dev, dt, attn_backend, kv_format, self.manager,
                                           n_stage)
        self.scheduler = Scheduler(self.manager, config)

    def add_request(self, req_id: int, prompt_ids: list[int], output_len: int) -> None:
        """Queue a request (FIFO)."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if output_len < 2:
            raise ValueError(f"output_len must be >= 2, got {output_len}")
        self.scheduler.add(req_id, prompt_ids, output_len)

    @torch.inference_mode()
    def step(self) -> list[tuple[int, int, bool]]:
        """Plan, run one forward over the packed batch, sample greedily, and commit. Returns the new tokens."""
        plan = self.scheduler.schedule()
        if not plan.seqs:
            return []
        dev = self.model.device
        ids = [t for s in plan.seqs for t in s.req.known[s.start:s.start + s.n]]
        meta = build_metadata(self.manager, [(s.req.req_id, s.start, s.n) for s in plan.seqs], dev)
        logits = self.model.forward(torch.tensor(ids, device=dev), meta.positions, self.cache, last_only=True,
                                    meta=meta)
        tokens = logits.argmax(-1).tolist()  # one GPU -> CPU sync per step, before the runner timestamps
        sampled = {s.req.req_id: tok for s, tok in zip(plan.seqs, tokens) if s.samples}
        return self.scheduler.commit(plan, sampled)

    def has_unfinished(self) -> bool:
        """True while any request is waiting or running."""
        return self.scheduler.has_unfinished()

    def reset_stats(self) -> None:
        """Zero the scheduler's memory-pressure counters."""
        self.scheduler.reset_stats()

    def stats(self) -> dict:
        """Preemptions, peak running sequences, and peak pool use since the last reset (recorded in run.json)."""
        return self.scheduler.stats() | {"kv_format": self.kv_format}
