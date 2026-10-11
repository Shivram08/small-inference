"""Raw run records -> summary metrics. Reads a run folder written by bench/runner.py and adds summary.json;
the raw records are never modified, so a summary can always be regenerated (e.g. with a different warmup_frac).

Definitions (n = output tokens of a request, t_1..t_n its token timestamps):
  TTFT = t_1 - arrival   TPOT = (t_n - t_1)/(n - 1)   E2E = t_n - arrival   ITL = pooled t_{k+1} - t_k

Warmup: the first `warmup_frac` of requests by scheduled arrival are dropped from every statistic.
Rates use the kept requests only:
  offered  = (k - 1) / (last arrival - first arrival)
  achieved = (k - 1) / (last finish  - first finish)
When the engine keeps up, every finish is about arrival + service time, so the two spans match. Under overload
finishes spread out at 1/S and achieved < offered. For a burst trace (one arrival time) offered is infinite
(written as null) and achieved is the engine's capacity.
Overload flag: achieved < (1 - rate_tol) * offered, or the waiting-queue slope over the second half of the kept
arrival window exceeds slope_tol * offered (requests/s). It is null for a burst trace, which is overloaded by design.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

SLO_TTFT_S = 2.0
SLO_TPOT_S = 0.100
PERCENTILES = (50, 90, 99)


def load_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per line."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def describe(values) -> dict:
    """Mean, p50, p90, p99 and count of a 1-D sample (None for an empty sample)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"n": 0, "mean": None, **{f"p{p}": None for p in PERCENTILES}}
    return {"n": int(x.size), "mean": float(x.mean()), **{f"p{p}": float(np.percentile(x, p)) for p in PERCENTILES}}


def queue_slope(steps: list[dict], t_lo: float, t_hi: float) -> float | None:
    """Least-squares slope (requests/s) of the waiting queue over steps ending in [t_lo, t_hi]."""
    pts = [(s["end_s"], s["waiting"]) for s in steps if t_lo <= s["end_s"] <= t_hi]
    if len(pts) < 2 or pts[-1][0] == pts[0][0]:
        return None
    t, w = np.array(pts).T
    return float(np.polyfit(t, w, 1)[0])


def summarize(requests: list[dict], steps: list[dict], warmup_frac: float = 0.1, slo_ttft_s: float = SLO_TTFT_S,
              slo_tpot_s: float = SLO_TPOT_S, rate_tol: float = 0.05, slope_tol: float = 0.05) -> dict:
    """Compute every summary metric from raw request and step records."""
    ordered = sorted(requests, key=lambda r: (r["arrival_s"], r["id"]))
    n_drop = int(len(ordered) * warmup_frac)
    kept = ordered[n_drop:]
    if len(kept) < 2:
        raise ValueError(f"need at least 2 requests after warmup, have {len(kept)}")

    ttft = [r["token_s"][0] - r["arrival_s"] for r in kept]
    tpot = [(r["token_s"][-1] - r["token_s"][0]) / (len(r["token_s"]) - 1) for r in kept]
    e2e = [r["token_s"][-1] - r["arrival_s"] for r in kept]
    itl = np.concatenate([np.diff(r["token_s"]) for r in kept])

    arrivals = sorted(r["arrival_s"] for r in kept)
    finishes = sorted(r["token_s"][-1] for r in kept)
    arrival_span, finish_span = arrivals[-1] - arrivals[0], finishes[-1] - finishes[0]
    burst = arrival_span == 0
    offered = math.inf if burst else (len(kept) - 1) / arrival_span
    achieved = (len(kept) - 1) / finish_span

    # Output throughput: every token (any request) emitted inside the kept finish window.
    all_tokens = np.concatenate([r["token_s"] for r in requests])
    in_window = np.count_nonzero((all_tokens >= finishes[0]) & (all_tokens <= finishes[-1]))

    meets_slo = [a < slo_ttft_s and b < slo_tpot_s for a, b in zip(ttft, tpot)]
    goodput_frac = sum(meets_slo) / len(kept)

    if burst:
        slope = rate_short = queue_growing = flag = None
    else:
        slope = queue_slope(steps, arrivals[0] + arrival_span / 2, arrivals[-1])
        rate_short = achieved < (1 - rate_tol) * offered
        queue_growing = slope is not None and slope > slope_tol * offered
        flag = rate_short or queue_growing
    return {
        "n_requests": len(requests),
        "n_warmup_dropped": n_drop,
        "n_kept": len(kept),
        "burst": burst,
        "ttft_s": describe(ttft),
        "tpot_s": describe(tpot),
        "itl_s": describe(itl),
        "e2e_s": describe(e2e),
        "offered_rate_req_s": None if burst else offered,
        "achieved_rate_req_s": achieved,
        "output_throughput_tok_s": in_window / finish_span,
        "slo": {"ttft_s": slo_ttft_s, "tpot_s": slo_tpot_s},
        "goodput_frac": goodput_frac,
        "goodput_req_s": goodput_frac * achieved,
        "queue_slope_req_s": slope,
        "overload": {"flag": flag, "rate_short": rate_short, "queue_growing": queue_growing,
                     "rate_tol": rate_tol, "slope_tol": slope_tol},
    }


def summarize_run(run_dir: Path, **kwargs) -> dict:
    """Summarize a run folder and write summary.json next to the raw records."""
    run_dir = Path(run_dir)
    summary = summarize(load_jsonl(run_dir / "requests.jsonl"), load_jsonl(run_dir / "steps.jsonl"), **kwargs)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """CLI: summarize one run folder and print the summary."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    args = p.parse_args()
    print(json.dumps(summarize_run(args.run_dir, warmup_frac=args.warmup_frac), indent=2))


if __name__ == "__main__":
    main()
