"""Iteration-level scheduler (Orca) with chunked prefill under a token budget (Sarathi-Serve) and recompute
preemption. Pure CPU logic over the BlockManager; engine/continuous.py runs the forward passes.

One rule covers prefill, decode, chunked prefill, and recompute. A request holds `known` tokens (prompt, then each
generated token) and `num_computed`, how many of them have K/V in the cache. A step computes some of
known[num_computed:]; when num_computed reaches len(known), the step's logits for that request are sampled and the
new token is appended to known. Decode is the n = 1 case. Preemption frees the blocks and sets num_computed = 0, so
the request later recomputes prompt + generated tokens in one (chunked) prefill and then continues decoding.

Per step, in order:
  1. Decodes. Every running request whose prefill is done gets 1 token. If its next token needs a new block and none
     is free, the most recently admitted running request is preempted (possibly itself) until it fits.
  2. Prefill chunks under the token budget: first running requests with an unfinished (chunked) prefill, then
     waiting requests in FIFO order, preempted ones first. Admission reserves blocks for the request's whole
     remaining prefill, and needs that plus a watermark of free blocks (headroom so admitted requests can decode
     without thrashing) and a free sequence slot. Reserving the whole prefill up front means a half-prefilled request
     never waits for blocks: allocating chunk by chunk let two admitted prefills deadlock on the same free blocks.
     Only decodes grow a sequence, and that is where preemption happens. With chunking off, a prefill is scheduled
     only if it fits the remaining budget in full (or runs alone if it is longer than the whole budget).

Shapes: with a 512-token budget and 0.5B, a step is at most a [512, 896] activation; a pure-decode step at batch 32
is [32, 896], whose linear layers sit in the cuBLAS dip measured in Phase 0 for M = 4-16 (and recover by M = 32).
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from engine.block_manager import BlockManager


class State(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


@dataclass
class Request:
    req_id: int
    prompt_len: int
    output_len: int
    known: list[int]  # prompt ids, then generated ids
    arrival_index: int
    state: State = State.WAITING
    num_computed: int = 0  # leading tokens of `known` whose K/V are in the cache
    num_preemptions: int = 0

    @property
    def num_generated(self) -> int:
        """Output tokens produced so far."""
        return len(self.known) - self.prompt_len

    @property
    def in_prefill(self) -> bool:
        """More than one token to compute (a prompt, or a recompute after preemption)."""
        return len(self.known) - self.num_computed > 1


@dataclass
class ScheduledSeq:
    req: Request
    start: int  # position of the first token computed this step
    n: int  # tokens computed this step

    @property
    def samples(self) -> bool:
        """Whether this step finishes the request's known tokens, so its logits are sampled."""
        return self.start + self.n == len(self.req.known)


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 512
    chunked_prefill: bool = True
    watermark_blocks: int = 0


@dataclass
class StepPlan:
    seqs: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[int] = field(default_factory=list)  # req_ids preempted while planning this step

    @property
    def num_tokens(self) -> int:
        """Tokens in the forward pass."""
        return sum(s.n for s in self.seqs)


class Scheduler:
    def __init__(self, manager: BlockManager, config: SchedulerConfig):
        """Schedule requests over the blocks of `manager`."""
        self.manager, self.cfg = manager, config
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # in admission order: the last one is preempted first
        self._arrivals = 0
        self.num_preemptions = 0
        self.reset_stats()

    def reset_stats(self) -> None:
        """Zero the counters reported by stats() (the runner calls this after warm-up)."""
        self.stat_preemptions, self.peak_running, self.peak_used_blocks, self.steps = 0, 0, 0, 0

    def stats(self) -> dict:
        """Memory-pressure counters since the last reset: preemptions, peak running sequences, peak blocks in use."""
        m = self.manager
        return {"preemptions": self.stat_preemptions, "peak_running_seqs": self.peak_running,
                "peak_used_blocks": self.peak_used_blocks, "num_blocks": m.num_blocks,
                "peak_pool_fraction": self.peak_used_blocks / m.num_blocks, "steps": self.steps}

    def add(self, req_id: int, prompt_ids: list[int], output_len: int) -> Request:
        """Queue a request. It must be able to fit in the whole pool on its own, or it could never run."""
        total = len(prompt_ids) + output_len
        if self.manager.blocks_for(total) > self.manager.num_blocks:
            raise ValueError(f"request {req_id} needs {self.manager.blocks_for(total)} blocks; the pool has "
                             f"{self.manager.num_blocks}")
        req = Request(req_id, len(prompt_ids), output_len, list(prompt_ids), self._arrivals)
        self._arrivals += 1
        self.waiting.append(req)
        return req

    def has_unfinished(self) -> bool:
        """True while any request is waiting or running."""
        return bool(self.waiting) or bool(self.running)

    def _preempt(self, req: Request) -> None:
        """Recompute preemption: drop the request's K/V and put it at the front of the queue."""
        self.manager.free(req.req_id)
        self.running.remove(req)
        req.state, req.num_computed = State.PREEMPTED, 0
        req.num_preemptions += 1
        self.num_preemptions += 1
        self.stat_preemptions += 1
        self.waiting.appendleft(req)

    def schedule(self) -> StepPlan:
        """Plan one forward pass and reserve its slots in the block manager."""
        plan, budget = StepPlan(), self.cfg.max_num_batched_tokens
        m = self.manager

        # 1. Decodes, oldest first; preempt from the newest end when a block is missing.
        for req in list(self.running):
            if req.state is not State.RUNNING or req.in_prefill:
                continue
            while not m.can_append(req.req_id, 1):
                victim = self.running[-1]
                self._preempt(victim)
                plan.preempted.append(victim.req_id)
                if victim is req:
                    break
            if req.state is State.RUNNING and budget > 0:
                m.append_slots(req.req_id, 1)
                plan.seqs.append(ScheduledSeq(req, req.num_computed, 1))
                budget -= 1

        # 2a. Continue unfinished (chunked) prefills of running requests. Their blocks were reserved at admission.
        for req in self.running:
            if budget == 0:
                break
            if req.in_prefill:
                n = self._prefill_chunk(req, budget)
                if n:
                    plan.seqs.append(ScheduledSeq(req, req.num_computed, n))
                    budget -= n

        # 2b. Admit waiting requests in order (preempted ones were put at the front).
        while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            req = self.waiting[0]
            remaining = len(req.known) - req.num_computed
            watermark = self.cfg.watermark_blocks if self.running else 0  # an idle engine must admit something
            if not m.can_allocate(remaining, watermark):
                break  # FIFO: a request that doesn't fit blocks the ones behind it
            n = self._prefill_chunk(req, budget)
            if n == 0:
                break
            self.waiting.popleft()
            m.allocate(req.req_id, remaining)  # the whole prefill: later chunks never wait for blocks
            req.state = State.RUNNING
            self.running.append(req)
            plan.seqs.append(ScheduledSeq(req, 0, n))
            budget -= n
        self.steps += 1
        self.peak_running = max(self.peak_running, len(self.running))
        self.peak_used_blocks = max(self.peak_used_blocks, m.num_blocks - m.num_free_blocks())
        return plan

    def _prefill_chunk(self, req: Request, budget: int) -> int:
        """Tokens of the request's remaining prefill to compute now (0: wait for a later step)."""
        remaining = len(req.known) - req.num_computed
        if self.cfg.chunked_prefill:
            return min(remaining, budget)
        if remaining <= budget or (budget == self.cfg.max_num_batched_tokens and remaining > budget):
            return remaining  # whole prefill; one longer than the budget runs alone in an otherwise empty step
        return 0

    def commit(self, plan: StepPlan, sampled: dict[int, int]) -> list[tuple[int, int, bool]]:
        """Apply a finished forward: advance num_computed, append sampled tokens, finish and free requests.
        `sampled` maps req_id -> token for the seqs whose `samples` is true. Returns (req_id, token, finished)."""
        out = []
        for s in plan.seqs:
            req = s.req
            req.num_computed = s.start + s.n
            if not s.samples:
                continue
            token = sampled[req.req_id]
            req.known.append(token)
            finished = req.num_generated == req.output_len
            out.append((req.req_id, token, finished))
            if finished:
                req.state = State.FINISHED
                self.running.remove(req)
                self.manager.free(req.req_id)
        return out
