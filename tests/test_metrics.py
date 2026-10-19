"""M0.3: metric formulas on hand-built and simulated records with exact known answers (no wall clock)."""

import json

import pytest

from bench.metrics import describe, summarize, summarize_run


def simulate_fifo(arrivals, prefill_s, decode_s, n_tokens):
    """Exact batch-1 FIFO timeline in runner record format: one step per token, prefill then decode steps."""
    requests, steps = [], []
    t, started, finished = 0.0, 0, 0
    for i, a in enumerate(arrivals):
        t = max(t, a)
        token_s = []
        for k in range(n_tokens):
            start = t
            t += prefill_s if k == 0 else decode_s
            token_s.append(t)
            started += k == 0
            finished += k == n_tokens - 1
            arrived = sum(x <= t for x in arrivals)
            steps.append({"start_s": start, "end_s": t, "n_tokens": 1,
                          "outstanding": arrived - finished, "waiting": arrived - started})
        requests.append({"id": i, "arrival_s": a, "prompt_len": 10, "output_len": n_tokens,
                         "admit_s": a, "token_s": token_s, "output_ids": list(range(n_tokens))})
    return requests, steps


def test_per_request_formulas():
    """TTFT, TPOT, E2E and pooled ITL follow the spec definitions exactly."""
    reqs = [{"id": 0, "arrival_s": 0.0, "token_s": [0.5, 0.6, 0.8]},
            {"id": 1, "arrival_s": 1.0, "token_s": [1.2, 1.5]}]
    s = summarize(reqs, [], warmup_frac=0.0)
    assert s["ttft_s"]["mean"] == pytest.approx((0.5 + 0.2) / 2)
    assert s["tpot_s"]["mean"] == pytest.approx((0.15 + 0.3) / 2)
    assert s["e2e_s"]["mean"] == pytest.approx((0.8 + 0.5) / 2)
    assert s["itl_s"]["n"] == 3 and s["itl_s"]["mean"] == pytest.approx((0.1 + 0.2 + 0.3) / 3)


def test_describe_percentiles():
    """Percentiles use linear interpolation; an empty sample gives None."""
    d = describe(range(101))  # 0..100
    assert (d["p50"], d["p90"], d["p99"], d["mean"]) == (50, 90, 99, 50)
    assert describe([])["mean"] is None


def test_warmup_dropped_by_arrival_not_completion():
    """The earliest arrivals are dropped even if they finish last."""
    reqs = [{"id": i, "arrival_s": float(i), "token_s": [i + 0.1, i + 0.2]} for i in range(10)]
    reqs[0]["token_s"] = [50.0, 51.0]  # arrives first, finishes last
    s = summarize(reqs, [], warmup_frac=0.1)
    assert s["n_warmup_dropped"] == 1 and s["n_kept"] == 9
    assert s["ttft_s"]["mean"] == pytest.approx(0.1)  # the slow request is gone


def test_underloaded_rates_match_and_no_flag():
    """Arrivals every 1 s, service 0.2 s: achieved equals offered, queue flat, no overload."""
    reqs, steps = simulate_fifo([float(i) for i in range(50)], prefill_s=0.05, decode_s=0.05, n_tokens=4)
    s = summarize(reqs, steps)
    assert s["offered_rate_req_s"] == pytest.approx(1.0)
    assert s["achieved_rate_req_s"] == pytest.approx(1.0)
    assert s["ttft_s"]["mean"] == pytest.approx(0.05) and s["tpot_s"]["mean"] == pytest.approx(0.05)
    assert s["queue_slope_req_s"] == pytest.approx(0.0, abs=1e-9)
    assert s["overload"]["flag"] is False


def test_overloaded_rate_short_and_queue_growing():
    """Arrivals every 0.5 s, service 1 s: completions at 1/s, waiting queue grows at ~1 request/s."""
    reqs, steps = simulate_fifo([0.5 * i for i in range(60)], prefill_s=0.25, decode_s=0.25, n_tokens=4)
    s = summarize(reqs, steps)
    assert s["offered_rate_req_s"] == pytest.approx(2.0)
    assert s["achieved_rate_req_s"] == pytest.approx(1.0)
    assert s["queue_slope_req_s"] == pytest.approx(1.0, rel=0.1)
    assert s["overload"] == {"flag": True, "rate_short": True, "queue_growing": True,
                             "rate_tol": 0.05, "slope_tol": 0.05, "gen_spread_frac": 0.0}  # equal generation times


def test_burst_gives_capacity_and_no_flag():
    """All arrivals at t = 0: offered is null, achieved is 1/S, the overload flag is not applicable."""
    reqs, steps = simulate_fifo([0.0] * 20, prefill_s=0.1, decode_s=0.1, n_tokens=5)  # S = 0.5 s
    s = summarize(reqs, steps)
    assert s["burst"] and s["offered_rate_req_s"] is None and s["overload"]["flag"] is None
    assert s["achieved_rate_req_s"] == pytest.approx(2.0)


def test_throughput_and_goodput():
    """Throughput counts tokens in the kept finish window; goodput counts kept requests meeting both SLOs."""
    # Prefill 0.1 s, then 4 decode steps of 0.05 s: tokens at i+0.1, +0.15, +0.2, +0.25, +0.3. TPOT 0.05 s.
    reqs, steps = simulate_fifo([float(i) for i in range(20)], prefill_s=0.1, decode_s=0.05, n_tokens=5)
    s = summarize(reqs, steps, warmup_frac=0.0)
    # Finish window [0.3, 19.3]: all 100 tokens except the 4 of request 0 before 0.3.
    assert s["output_throughput_tok_s"] == pytest.approx((20 * 5 - 4) / 19.0)
    assert s["goodput_frac"] == 1.0
    tight = summarize(reqs, steps, warmup_frac=0.0, slo_tpot_s=0.04)  # TPOT 0.05 s now misses the SLO
    assert tight["goodput_frac"] == 0.0 and tight["goodput_req_s"] == 0.0


def test_summarize_run_writes_valid_json(tmp_path):
    """summary.json is strict JSON (no Infinity), even for a burst run."""
    reqs, steps = simulate_fifo([0.0] * 10, prefill_s=0.1, decode_s=0.1, n_tokens=3)
    for name, rows in [("requests.jsonl", reqs), ("steps.jsonl", steps)]:
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    summarize_run(tmp_path)
    json.loads((tmp_path / "summary.json").read_text(), parse_constant=lambda c: pytest.fail(f"non-JSON {c}"))


def test_rate_check_allows_generation_spread_but_not_queueing():
    """A short trace whose requests generate for very different times reads 'short' on the raw rate even though the
    server keeps up; the generation-time allowance clears it. Queueing delay does not count toward the allowance."""
    # 20 arrivals 1 s apart; generation alternates 0.1 s and 6 s (output length), no queueing (infinite servers).
    reqs = []
    for i in range(20):
        a, g = float(i), (0.1 if i % 2 else 6.0)
        reqs.append({"id": i, "arrival_s": a, "token_s": [a + 0.05, a + 0.05 + g], "output_len": 2})
    steps = [{"start_s": t * 0.5, "end_s": t * 0.5 + 0.5, "n_tokens": 1, "outstanding": 1, "waiting": 0}
             for t in range(60)]
    s = summarize(reqs, steps, warmup_frac=0.0)
    raw_short = s["achieved_rate_req_s"] < 0.95 * s["offered_rate_req_s"]
    assert raw_short and not s["overload"]["rate_short"] and s["overload"]["gen_spread_frac"] > 0.2
