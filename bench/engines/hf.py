"""HF transformers baseline: batch-1 FIFO, manual prefill/decode loop with past_key_values, greedy, EOS ignored.

Loads bf16 with attn_implementation="sdpa". Sampling is a raw argmax over the last position's logits, so the
model's generation_config.json defaults (do_sample, temperature, top_p/top_k, repetition_penalty 1.1) do not apply.
Each step() returns only after the sampled token is on the CPU (.item()), as the Engine contract requires.
"""

from collections import deque
from dataclasses import dataclass

import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class _Request:
    req_id: int
    prompt_ids: list[int]
    output_len: int
    generated: int = 0
    past: object = None  # HF cache object after prefill
    last_token: torch.Tensor | None = None  # [1, 1] on device


class HFEngine:
    def __init__(self, model: str = MODEL, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        """Load the model once; requests then run one at a time in arrival order."""
        from transformers import AutoModelForCausalLM

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, attn_implementation="sdpa",
                                                          local_files_only=True).to(device).eval()
        self.waiting: deque[_Request] = deque()
        self.running: _Request | None = None

    def add_request(self, req_id: int, prompt_ids: list[int], output_len: int) -> None:
        """Append a request to the FIFO queue."""
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if output_len < 2:
            raise ValueError(f"output_len must be >= 2, got {output_len}")
        self.waiting.append(_Request(req_id, list(prompt_ids), output_len))

    @torch.inference_mode()
    def step(self) -> list[tuple[int, int, bool]]:
        """Prefill the next waiting request or decode one token of the running one; returns that token."""
        if self.running is None:
            if not self.waiting:
                return []
            req = self.running = self.waiting.popleft()
            ids = torch.tensor([req.prompt_ids], device=self.device)
            out = self.model(ids, use_cache=True)
        else:
            req = self.running
            out = self.model(req.last_token, past_key_values=req.past, use_cache=True)
        req.past = out.past_key_values
        req.last_token = out.logits[:, -1].argmax(-1, keepdim=True)
        token_id = req.last_token.item()  # GPU -> CPU sync before the runner timestamps

        req.generated += 1
        finished = req.generated == req.output_len
        if finished:
            req.past = req.last_token = None  # free the KV cache now, not when the next request starts
            self.running = None
        return [(req.req_id, token_id, finished)]

    def has_unfinished(self) -> bool:
        """True while a request is running or waiting."""
        return self.running is not None or bool(self.waiting)
