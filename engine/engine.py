"""Phase 1 serving engine: batch-1 FIFO over the hand-written Qwen2 with a contiguous KV cache.

It mirrors bench/engines/hf.py (one request at a time, prefill then one token per step(), greedy, EOS ignored) so the
harness comparison isolates the forward pass. One cache of max_len tokens is allocated up front and reused by every
request; nothing is allocated per request. Each step() returns after the token reaches the CPU (int()).
"""

from collections import deque
from dataclasses import dataclass

import torch

from engine.kv_cache import ContiguousKVCache
from engine.model import Qwen2Model

MAX_LEN = 2048  # trace filter: prompt + output <= 2048


@dataclass
class _Request:
    req_id: int
    prompt_ids: list[int]
    output_len: int
    generated: int = 0
    last_token: torch.Tensor | None = None  # [1] on device


class ContiguousEngine:
    def __init__(self, model: Qwen2Model | None = None, max_len: int = MAX_LEN):
        """Use `model`, or load the default 0.5B model in bf16 on the GPU."""
        self.model = model or Qwen2Model.from_pretrained()
        self.cache = ContiguousKVCache(self.model.cfg, max_len, self.model.device, self.model.dtype)
        self.waiting: deque[_Request] = deque()
        self.running: _Request | None = None

    def add_request(self, req_id: int, prompt_ids: list[int], output_len: int) -> None:
        """Append a request to the FIFO queue."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if output_len < 2:
            raise ValueError(f"output_len must be >= 2, got {output_len}")
        if len(prompt_ids) + output_len > self.cache.max_len:
            raise ValueError(f"prompt {len(prompt_ids)} + output {output_len} exceeds max_len {self.cache.max_len}")
        self.waiting.append(_Request(req_id, list(prompt_ids), output_len))

    @torch.inference_mode()
    def step(self) -> list[tuple[int, int, bool]]:
        """Prefill the next waiting request or decode one token of the running one; returns that token."""
        if self.running is None:
            if not self.waiting:
                return []
            req = self.running = self.waiting.popleft()
            self.cache.reset()
            ids = torch.tensor(req.prompt_ids, device=self.model.device)
        else:
            req = self.running
            ids = req.last_token
        logits = self.model.forward(ids, self.cache.next_positions(ids.numel()), self.cache, last_only=True)
        req.last_token = logits[0].argmax()[None]
        token_id = int(req.last_token)  # GPU -> CPU sync before the runner timestamps

        req.generated += 1
        finished = req.generated == req.output_len
        if finished:
            self.running = None
        return [(req.req_id, token_id, finished)]

    def has_unfinished(self) -> bool:
        """True while a request is running or waiting."""
        return self.running is not None or bool(self.waiting)
