"""Synthetic-timing engine for testing the harness: batch-1 FIFO, prefill costs a*P, each decode step costs b."""

import time
from collections import deque
from dataclasses import dataclass

SPIN_S = 1e-3  # sleep until this far before the deadline, then spin (time.sleep can overshoot)


def wait_until(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > SPIN_S:
        time.sleep(remaining - SPIN_S)
    while time.perf_counter() < deadline:
        pass


@dataclass
class _Request:
    req_id: int
    prompt_len: int
    output_len: int
    generated: int = 0


class FakeEngine:
    def __init__(self, a: float, b: float):
        """a: prefill seconds per prompt token. b: seconds per decode step."""
        self.a = a
        self.b = b
        self.waiting: deque[_Request] = deque()
        self.running: _Request | None = None

    def add_request(self, req_id: int, prompt_ids: list[int], output_len: int) -> None:
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if output_len < 2:
            raise ValueError(f"output_len must be >= 2, got {output_len}")
        self.waiting.append(_Request(req_id, len(prompt_ids), output_len))

    def step(self) -> list[tuple[int, int, bool]]:
        start = time.perf_counter()
        if self.running is None:
            if not self.waiting:
                return []
            self.running = self.waiting.popleft()
            cost = self.a * self.running.prompt_len  # prefill, emits the first token
        else:
            cost = self.b
        wait_until(start + cost)

        req = self.running
        token_id = req.generated  # deterministic placeholder token
        req.generated += 1
        finished = req.generated == req.output_len
        if finished:
            self.running = None
        return [(req.req_id, token_id, finished)]

    def has_unfinished(self) -> bool:
        return self.running is not None or bool(self.waiting)
