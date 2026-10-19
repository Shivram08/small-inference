"""Phase 6: the vLLM adapter follows the Engine contract (skipped unless run in the vLLM venv with vllm installed).

Run in the vLLM venv: .venv-vllm/bin/python -m pytest tests/test_vllm_engine.py -p no:cacheprovider -o addopts=""
Loads Qwen2.5-0.5B in vLLM (eager, so no CUDA-graph capture): about a minute.
"""

import json

import pytest

vllm = pytest.importorskip("vllm")


@pytest.fixture(scope="module")
def engine():
    """One eager vLLM engine for the module."""
    from bench.engines.vllm_engine import VLLMEngine
    return VLLMEngine(enforce_eager=True, gpu_memory_utilization=0.5)


def test_contract_and_exact_lengths(engine):
    """Every request gets exactly output_len tokens with one finished flag at the end, EOS ignored."""
    from bench.engines.base import Engine
    assert isinstance(engine, Engine)
    with open("traces/sharegpt_n100_seed0_burst.jsonl") as f:
        rows = [json.loads(line) for line, _ in zip(f, range(4))]
    lens = [3, 40, 120, 7]
    for r, n in zip(rows, lens):
        engine.add_request(r["id"], r["prompt_ids"], n)
    out = []
    while engine.has_unfinished():
        out += engine.step()
    for r, n in zip(rows, lens):
        mine = [(tok, fin) for rid, tok, fin in out if rid == r["id"]]
        assert len(mine) == n and [fin for _, fin in mine] == [False] * (n - 1) + [True]


def test_short_greedy_request(engine):
    """A single 8-token greedy request completes with exactly 8 tokens (printed, to compare by eye with this
    engine's bf16 greedy output for the same prompt; bf16 kernels differ, so no equality is asserted)."""
    with open("traces/sharegpt_n100_seed0_burst.jsonl") as f:
        prompt = json.loads(next(f))["prompt_ids"]
    engine.add_request(999, prompt, 8)
    toks = []
    while engine.has_unfinished():
        toks += [t for rid, t, _ in engine.step() if rid == 999]
    assert len(toks) == 8
    print(f"\nvLLM first 8 greedy tokens: {toks}")
