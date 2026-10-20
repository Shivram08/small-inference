"""Phase 4 scheduler and continuous-batching engine on the miniature fp64 model (CPU, gather attention).

Invariant 6 (batch invariance): every request's greedy output equals its solo output (Phase 1 greedy_generate)
under any mix: all at once, staggered arrivals, tiny token budgets that chunk every prefill, chunking off.
Invariant 7 (forced preemption): with a pool barely larger than one request, requests are preempted and recomputed
repeatedly, yet all finish with unchanged outputs and every block is free at the end.
fp64 makes the different matmul shapes of batching and chunking agree to ~1e-15, so outputs are compared exactly.
"""

import random

import pytest
import torch

from bench.engines.base import Engine
from engine.continuous import ContinuousEngine
from engine.generate import greedy_generate
from engine.model import Qwen2Model
from engine.scheduler import SchedulerConfig
from tests.tiny_model import TINY, random_weights


@pytest.fixture(scope="module")
def tiny():
    """The miniature model in fp64 on the CPU."""
    return Qwen2Model(TINY, random_weights(TINY))


def make_requests(seed: int, n: int = 8, max_prompt: int = 40, max_out: int = 25):
    """(prompt_ids, output_len) pairs with varied lengths, including a 1-token prompt."""
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        p = 1 if i == 0 else rng.randint(2, max_prompt)
        reqs.append(([rng.randrange(TINY.vocab_size) for _ in range(p)], rng.randint(2, max_out)))
    return reqs


@pytest.fixture(scope="module")
def solo(tiny):
    """Reference outputs for each seed's requests, each generated alone with the contiguous cache."""
    cache = {}

    def get(seed):
        """Solo outputs for make_requests(seed)."""
        if seed not in cache:
            cache[seed] = [greedy_generate(tiny, p, n) for p, n in make_requests(seed)]
        return cache[seed]
    return get


def engine(tiny, num_blocks=64, block_size=4, **cfg):
    """A continuous engine on the miniature model."""
    return ContinuousEngine(tiny, num_blocks=num_blocks, block_size=block_size, config=SchedulerConfig(**cfg),
                            attn_backend="gather")


def run(eng, requests, arrivals=None, check=False):
    """Feed requests (arrivals[i] = step at which request i is added; default all at step 0), step until done.
    Returns per-request token lists and the (req_id, token, finished) stream."""
    arrivals = arrivals or [0] * len(requests)
    outs, stream, step = {i: [] for i in range(len(requests))}, [], 0
    pending = sorted(range(len(requests)), key=lambda i: arrivals[i])
    while pending or eng.has_unfinished():
        while pending and arrivals[pending[0]] <= step:
            i = pending.pop(0)
            eng.add_request(i, *requests[i])
        for rid, tok, fin in eng.step():
            outs[rid].append(tok)
            stream.append((rid, tok, fin))
        if check:
            eng.manager.check_invariants()
        step += 1
        assert step < 10_000, "engine stopped making progress"
    return [outs[i] for i in range(len(requests))], stream


def test_conforms_to_protocol(tiny):
    """ContinuousEngine satisfies the Engine protocol."""
    assert isinstance(engine(tiny), Engine)


CONFIGS = {
    "solo-seqs": dict(max_num_seqs=1),
    "batch8": dict(max_num_seqs=8),
    "budget7-chunked": dict(max_num_seqs=8, max_num_batched_tokens=7),
    "budget64-no-chunk": dict(max_num_seqs=3, max_num_batched_tokens=64, chunked_prefill=False),
    "budget16-no-chunk-oversize": dict(max_num_seqs=8, max_num_batched_tokens=16, chunked_prefill=False),
}


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("seed", [0, 1])
def test_batch_invariance(tiny, solo, config, seed):
    """Invariant 6: outputs equal solo outputs for each config, with all-at-once and staggered arrivals."""
    reqs = make_requests(seed)
    for arrivals in (None, [0, 0, 1, 3, 3, 4, 10, 11]):
        outs, stream = run(engine(tiny, **CONFIGS[config]), reqs, arrivals, check=True)
        assert outs == solo(seed), f"{config}, arrivals {arrivals}"
        for i, (_, n) in enumerate(reqs):  # exactly one finished flag, on the last token
            assert [fin for rid, _, fin in stream if rid == i] == [False] * (n - 1) + [True]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forced_preemption(tiny, solo, seed):
    """Invariant 7: a pool exactly as large as the largest request forces recompute preemption, including of
    requests mid-decode (their recompute covers prompt + generated tokens); outputs are unchanged and no block
    leaks."""
    reqs = make_requests(seed)
    need = max(-(-(len(p) + n) // 4) for p, n in reqs)
    eng = engine(tiny, num_blocks=need, max_num_seqs=8)
    generated_at_preemption = []
    original = eng.scheduler._preempt

    def recording(req):
        """Note how many tokens the victim had generated."""
        generated_at_preemption.append(req.num_generated)
        original(req)
    eng.scheduler._preempt = recording
    outs, _ = run(eng, reqs, check=True)
    assert outs == solo(seed)
    assert eng.scheduler.num_preemptions > 0, "the pool was not small enough to force preemption"
    assert any(g > 0 for g in generated_at_preemption), "no request was preempted mid-decode"
    assert eng.manager.num_free_blocks() == eng.manager.num_blocks
    print(f"\nseed {seed}: {eng.scheduler.num_preemptions} preemptions in a {need}-block pool, victims had "
          f"generated {generated_at_preemption} tokens")


def test_step_plans_respect_limits(tiny):
    """Every plan: tokens within the budget, sequences within max_num_seqs, decodes before prefill chunks, and a
    prefill chunk only for requests whose earlier tokens are all computed."""
    eng = engine(tiny, max_num_seqs=4, max_num_batched_tokens=12)
    plans = []
    original = eng.scheduler.schedule

    def recording():
        """Record each plan as (n, start, len(known) at planning time)."""
        plan = original()
        plans.append([(s.n, s.start, len(s.req.known)) for s in plan.seqs])
        return plan
    eng.scheduler.schedule = recording
    run(eng, make_requests(3))
    for plan in plans:
        assert sum(n for n, _, _ in plan) <= 12 and len(plan) <= 4
        kinds = ["decode" if n == 1 and start == known - 1 else "prefill" for n, start, known in plan]
        assert kinds == sorted(kinds)  # "decode" < "prefill": all decodes first
    assert any(sum(n for n, _, _ in plan) == 12 for plan in plans)  # the budget was actually filled


def test_chunked_prefill_emits_first_token_at_last_chunk(tiny):
    """A 30-token prompt with an 8-token budget takes 4 steps (8 + 8 + 8 + 6) before its first token."""
    eng = engine(tiny, max_num_batched_tokens=8)
    eng.add_request(0, list(range(30)), 3)
    emitted = [eng.step() for _ in range(6)]
    assert [len(e) for e in emitted] == [0, 0, 0, 1, 1, 1]
    assert not eng.has_unfinished() and eng.step() == []


@pytest.mark.parametrize("seed", range(12))
def test_randomized_stress(tiny, seed):
    """Random requests, arrivals, budgets, sequence limits, chunking, and pool sizes (down to one request's worth):
    bookkeeping stays consistent every step, everything finishes, outputs equal solo, nothing leaks."""
    rng = random.Random(100 + seed)
    reqs = make_requests(100 + seed, n=rng.randint(3, 10), max_prompt=30, max_out=12)
    B = rng.choice([1, 4, 8])
    need = max(-(-(len(p) + n) // B) for p, n in reqs)
    eng = engine(tiny, num_blocks=need + rng.randint(0, 3 * need), block_size=B,
                 max_num_seqs=rng.randint(1, 8), max_num_batched_tokens=rng.choice([5, 16, 64]),
                 chunked_prefill=rng.random() < 0.7, watermark_blocks=rng.randint(0, 2))
    arrivals = sorted(rng.randint(0, 20) for _ in reqs)
    outs, _ = run(eng, reqs, arrivals, check=True)
    assert outs == [greedy_generate(tiny, p, n) for p, n in reqs]
    assert eng.manager.num_free_blocks() == eng.manager.num_blocks


def test_rejects_impossible_requests(tiny):
    """A request that could never fit the pool, an empty prompt, or output_len < 2 is refused up front."""
    eng = engine(tiny, num_blocks=4, block_size=4)
    with pytest.raises(ValueError, match="blocks"):
        eng.add_request(0, list(range(10)), 10)  # 20 tokens > 16 slots
    with pytest.raises(ValueError):
        eng.add_request(1, [], 5)
    with pytest.raises(ValueError):
        eng.add_request(2, [1], 1)


def test_stats_report_memory_pressure(tiny):
    """stats() counts preemptions and peak pool use since reset; a pool the size of one request is fully used."""
    reqs = make_requests(0)
    need = max(-(-(len(p) + n) // 4) for p, n in reqs)
    eng = engine(tiny, num_blocks=need, max_num_seqs=8)
    eng.reset_stats()
    run(eng, reqs)
    st = eng.stats()
    assert st["preemptions"] == eng.scheduler.num_preemptions > 0
    assert st["peak_used_blocks"] == need and st["peak_pool_fraction"] == 1.0 and 1 < st["peak_running_seqs"] <= 8
    eng.reset_stats()
    assert eng.stats()["preemptions"] == 0
