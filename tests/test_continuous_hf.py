"""Phase 4 on Qwen2.5-0.5B: invariants 6 and 7 in fp32 with the Triton decode backend, and a bf16 run through the
open-loop harness.

Batching and chunking change the M of every matmul, so fp32 logits differ in the last bits between mixes (cuBLAS
picks kernels by shape, and the decode kernel's split count depends on the batch); the invariant is on greedy tokens,
which must not change. About a minute: one fp32 model (2 GB), then one bf16 model (1 GB).
"""

import json

import pytest
import torch

from bench.runner import TraceRequest, replay
from engine.continuous import ContinuousEngine
from engine.generate import greedy_generate
from engine.model import Qwen2Model
from engine.scheduler import SchedulerConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
LENGTHS = [1, 17, 40, 100, 160, 250, 33, 64]
OUT = 24


@pytest.fixture(scope="module")
def requests():
    """Eight ShareGPT trace prompts truncated to varied lengths, 24 output tokens each."""
    with open("traces/sharegpt_n100_seed0_burst.jsonl") as f:
        prompts = [r["prompt_ids"] for r in map(json.loads, f) if len(r["prompt_ids"]) >= 250]
    return [(prompts[i][:n], OUT) for i, n in enumerate(LENGTHS)]


@pytest.fixture(scope="module")
def model32():
    """Qwen2.5-0.5B in fp32."""
    try:
        m = Qwen2Model.from_pretrained(dtype=torch.float32)
    except OSError:
        pytest.skip("Qwen2.5-0.5B-Instruct not in the local HF cache")
    yield m
    del m
    torch.cuda.empty_cache()


def run_all(eng, requests):
    """Add every request at once and step to completion; per-request token lists."""
    for i, (p, n) in enumerate(requests):
        eng.add_request(i, p, n)
    outs = {i: [] for i in range(len(requests))}
    while eng.has_unfinished():
        for rid, tok, _ in eng.step():
            outs[rid].append(tok)
    eng.manager.check_invariants()
    assert eng.manager.num_free_blocks() == eng.manager.num_blocks
    return [outs[i] for i in range(len(requests))]


@pytest.fixture(scope="module")
def solo(model32, requests):
    """Each request alone through the same engine (max_num_seqs = 1)."""
    eng = ContinuousEngine(model32, num_blocks=256, config=SchedulerConfig(max_num_seqs=1))
    return run_all(eng, requests)


def test_solo_matches_phase1(model32, requests, solo):
    """Alone in the continuous engine equals Phase 1's contiguous greedy decode."""
    assert solo == [greedy_generate(model32, p, n) for p, n in requests]


@pytest.mark.parametrize("config", [dict(max_num_seqs=8), dict(max_num_seqs=8, max_num_batched_tokens=64),
                                    dict(max_num_seqs=3, max_num_batched_tokens=300, chunked_prefill=False)],
                         ids=["batch8", "chunk64", "no-chunk-3seqs"])
def test_batch_invariance_fp32(model32, requests, solo, config):
    """Invariant 6: each request's 24 greedy tokens are identical alone and in the mix."""
    assert run_all(ContinuousEngine(model32, num_blocks=256, config=SchedulerConfig(**config)), requests) == solo


def test_forced_preemption_fp32(model32, requests):
    """Invariant 7: five short prompts with 60-token outputs in a pool the size of the largest request (8 blocks of
    16). Decode growth outruns the pool, so requests are preempted mid-decode and recomputed; every output equals
    its Phase 1 greedy decode, and no block leaks."""
    reqs = [(requests[i][0], 60) for i in (0, 1, 2, 6, 7)]  # prompt lengths 1, 17, 40, 33, 64
    need = max(-(-(len(p) + n) // 16) for p, n in reqs)
    eng = ContinuousEngine(model32, num_blocks=need, config=SchedulerConfig(max_num_seqs=8))
    victims = []
    original = eng.scheduler._preempt

    def recording(req):
        """Note how many tokens the victim had generated."""
        victims.append(req.num_generated)
        original(req)
    eng.scheduler._preempt = recording
    assert run_all(eng, reqs) == [greedy_generate(model32, p, n) for p, n in reqs]
    print(f"\n{len(victims)} preemptions in a {need}-block pool; victims had generated {victims} tokens")
    assert any(g > 0 for g in victims), "no request was preempted mid-decode"


def test_bf16_through_the_harness(requests):
    """The bf16 engine under the open-loop runner: arrivals 20 ms apart, every token delivered and in order, and
    more than one request in flight at once (it batches)."""
    eng = ContinuousEngine(kv_bytes=256 * 2**20)  # bf16 0.5B, 1,365 blocks
    reqs = [TraceRequest(i, 0.02 * i, p, n) for i, (p, n) in enumerate(requests)]
    records, steps = replay(eng, reqs)
    assert all(len(r.token_s) == n for r, (_, n) in zip(records, requests))
    assert max(s.n_tokens for s in steps) > 1  # some step produced tokens for several requests
    assert eng.manager.num_free_blocks() == eng.manager.num_blocks
