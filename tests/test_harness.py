"""M0.4 gate: the whole harness (trace -> runner -> metrics) recovers the fake engine's known timing in real time.

Fake engine: a = 10 µs/token, b = 1 ms, P = 300, output_len = 21, so S = 3 ms + 20 * 1 ms = 23 ms and
capacity 1/S = 43.5 req/s. Slow tests (seconds of wall clock each): run with `pytest -m slow`.
"""

import numpy as np
import pytest

from bench.engines.fake import FakeEngine
from bench.metrics import load_jsonl, summarize_run
from bench.runner import run
from bench.trace import poisson_arrivals, write_trace

A, B, P, N = 10e-6, 1e-3, 300, 21
S = A * P + (N - 1) * B
CAPACITY = 1 / S


def replay_fake(tmp_path, arrivals, name):
    """Run the fake engine over the given arrivals; return (summary, steps)."""
    trace = write_trace(tmp_path / f"{name}.jsonl", arrivals, [[1] * P] * len(arrivals), [N] * len(arrivals))
    run_dir = run(FakeEngine(a=A, b=B), trace, "fake", {"a": A, "b": B}, results_root=tmp_path / "results")
    return summarize_run(run_dir), load_jsonl(run_dir / "steps.jsonl")


def runner_overhead(steps) -> float:
    """Median gap between one step's end and the next step's start (the median ignores idle waits between requests)."""
    gaps = np.array([b["start_s"] - a["end_s"] for a, b in zip(steps, steps[1:])])
    return float(np.median(gaps)) if gaps.size else 0.0


def test_mean_interarrival_matches_rate():
    """Mean Poisson gap is 1/λ within 4 standard errors (sd of an exponential gap = mean, so SE = 1/(λ√n))."""
    rate, n = 5.0, 20_000
    gaps = np.diff(poisson_arrivals(rate, n, seed=0), prepend=0.0)
    assert abs(gaps.mean() - 1 / rate) < 4 / (rate * np.sqrt(n))
    assert np.array_equal(poisson_arrivals(rate, 100, seed=1), poisson_arrivals(rate, 100, seed=1))


@pytest.mark.slow
def test_unloaded_ttft_and_tpot(tmp_path):
    """Arrivals 50 ms apart (> S = 23 ms), so no queueing: TTFT ≈ a·P and TPOT ≈ b plus runner overhead."""
    summary, steps = replay_fake(tmp_path, [0.05 * k for k in range(60)], "unloaded")
    overhead = runner_overhead(steps)
    ttft, tpot = summary["ttft_s"]["mean"], summary["tpot_s"]["mean"]
    print(f"\nTTFT {ttft * 1e3:.4f} ms (a·P = {A * P * 1e3:.1f}), TPOT {tpot * 1e3:.4f} ms (b = {B * 1e3:.1f}), "
          f"runner overhead {overhead * 1e6:.1f} µs/step, TPOT p99 {summary['tpot_s']['p99'] * 1e3:.4f} ms")
    # TTFT = admission lag (~16 µs idle) + prefill. The fake never finishes early.
    assert A * P <= ttft < A * P + 0.1e-3
    # TPOT = b + overhead between steps (the fake times itself from its own start) + spin overshoot (~3 µs) +
    # occasional WSL preemption (TPOT p99 ~1.1 ms). Measured excess over 3 runs: 12-17 µs. Allow 50 µs (5% of b).
    assert B <= tpot < B + overhead + 50e-6
    assert overhead < 100e-6
    assert summary["overload"]["flag"] is False


@pytest.mark.slow
def test_burst_capacity_is_one_over_s(tmp_path):
    """A burst trace measures capacity 1/S, lowered only by per-step runner overhead (21 steps per request)."""
    summary, steps = replay_fake(tmp_path, [0.0] * 150, "burst")
    expected = 1 / (S + N * runner_overhead(steps))
    print(f"\nburst capacity {summary['achieved_rate_req_s']:.2f} req/s, 1/S = {CAPACITY:.2f}, "
          f"overhead-corrected {expected:.2f}")
    assert summary["burst"] and summary["overload"]["flag"] is None
    assert summary["achieved_rate_req_s"] == pytest.approx(expected, rel=0.02)


@pytest.mark.slow
@pytest.mark.parametrize("rho, overloaded, n", [(0.4, False, 250), (1.2, True, 150)])
def test_overload_flag(tmp_path, rho, overloaded, n):
    """The overload flag is off at ρ = 0.4 and on at ρ = 1.2 (Poisson arrivals, fixed seed)."""
    rate = rho * CAPACITY
    summary, _ = replay_fake(tmp_path, poisson_arrivals(rate, n, seed=42), f"rho{rho}")
    o = summary["overload"]
    print(f"\nρ={rho}: offered {summary['offered_rate_req_s']:.2f}, achieved {summary['achieved_rate_req_s']:.2f} "
          f"req/s, queue slope {summary['queue_slope_req_s']:.3f} req/s, flags {o}")
    assert summary["offered_rate_req_s"] == pytest.approx(rate, rel=0.2)  # realized Poisson rate, n = 150-250
    assert o["flag"] is overloaded
    if overloaded:
        assert o["rate_short"] and o["queue_growing"]
