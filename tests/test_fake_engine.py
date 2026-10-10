"""M0.1 gate: the fake engine follows the Engine protocol, batch-1 FIFO semantics, and its timing model."""

import statistics
import time

import pytest

from bench.engines.base import Engine
from bench.engines.fake import FakeEngine


def drain(engine):
    """Step until done. Returns (outputs, per-step durations)."""
    outputs, durations = [], []
    while engine.has_unfinished():
        t0 = time.perf_counter()
        out = engine.step()
        durations.append(time.perf_counter() - t0)
        outputs.extend(out)
    return outputs, durations


def test_conforms_to_protocol():
    assert isinstance(FakeEngine(a=0.0, b=0.0), Engine)


def test_exact_output_lengths_and_fifo_order():
    eng = FakeEngine(a=0.0, b=0.0)
    lens = {7: 2, 3: 5, 11: 3}
    for rid, n in lens.items():
        eng.add_request(rid, [1] * 10, n)
    outputs, _ = drain(eng)

    # FIFO at batch 1: each request runs to completion before the next starts.
    expected_ids = [rid for rid, n in lens.items() for _ in range(n)]
    assert [rid for rid, _, _ in outputs] == expected_ids
    # Exactly one finished flag per request, on its last token.
    for rid, n in lens.items():
        flags = [fin for r, _, fin in outputs if r == rid]
        assert flags == [False] * (n - 1) + [True]
    assert not eng.has_unfinished()


def test_idle_step_returns_empty_without_waiting():
    eng = FakeEngine(a=1.0, b=1.0)
    t0 = time.perf_counter()
    assert eng.step() == []
    assert time.perf_counter() - t0 < 1e-3
    assert not eng.has_unfinished()


def test_requests_added_mid_run_are_served():
    eng = FakeEngine(a=0.0, b=0.0)
    eng.add_request(0, [1], 3)
    eng.step()
    eng.add_request(1, [1], 2)
    outputs, _ = drain(eng)
    assert [rid for rid, _, _ in outputs] == [0, 0, 1, 1]


@pytest.mark.parametrize("bad_len", [0, 1])
def test_rejects_output_len_below_two(bad_len):
    with pytest.raises(ValueError):
        FakeEngine(a=0.0, b=0.0).add_request(0, [1, 2], bad_len)


def test_rejects_empty_prompt():
    with pytest.raises(ValueError):
        FakeEngine(a=0.0, b=0.0).add_request(0, [], 4)


def test_step_timing_matches_model():
    a, b, P, n = 1e-4, 5e-3, 300, 41  # prefill 30 ms, then 40 decode steps of 5 ms
    eng = FakeEngine(a=a, b=b)
    eng.add_request(0, [1] * P, n)
    _, durations = drain(eng)

    prefill, decode = durations[0], durations[1:]
    assert len(decode) == n - 1
    print(f"\nprefill {prefill * 1e3:.3f} ms (target {a * P * 1e3:.1f}), "
          f"decode median {statistics.median(decode) * 1e3:.3f} ms, max {max(decode) * 1e3:.3f} ms (target {b * 1e3:.1f})")
    # The spin-wait never returns early, and lateness should be well under 0.2 ms.
    assert a * P <= prefill < a * P + 2e-4
    assert all(b <= d for d in decode)
    assert statistics.median(decode) < b + 2e-4
