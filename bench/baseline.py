"""HF baseline campaign: burst run -> capacity -> rate traces at 0.3x/0.7x/1.2x -> 3 shuffled repetitions.

The first burst run measures capacity (achieved rate on the burst trace, about 1/S) and counts as repetition 1's
burst. Every repetition runs each trace once, in an order shuffled per repetition with a recorded seed, so slow
drift (temperature, background load, the Phase 0 evening slowdown) is spread across rates instead of always hitting
the same position. Between runs the GPU cools to its idle temperature at campaign start (plus a tolerance) rather
than for a fixed time. Every run samples GPU and host-CPU telemetry. A manifest listing every run folder is written
to results/<stamp>_hf_baseline/manifest.json and updated after each run.
"""

import argparse
import json
import random
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from bench.env import gpu_temperature
from bench.metrics import summarize_run
from bench.runner import run
from bench.trace import build_traces

FRACTIONS = (0.3, 0.7, 1.2)
N, SEED = 100, 0
COOL_TOL_C, COOL_MIN_S, COOL_MAX_S, COOL_POLL_S = 3.0, 30.0, 600.0, 5.0


def rep_orders(burst: Path, rate_traces: list[Path], reps: int, seed: int) -> list[list[Path]]:
    """Run order per repetition. Repetition 1 already ran its burst first (to measure capacity), so it shuffles only
    the rate traces; later repetitions shuffle all four."""
    rng = random.Random(seed)
    orders = []
    for rep in range(1, reps + 1):
        order = list(rate_traces) if rep == 1 else [burst, *rate_traces]
        rng.shuffle(order)
        orders.append(order)
    return orders


def cool_down(target_c: float | None, read_temp: Callable[[], float | None] = gpu_temperature,
              sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
              min_s: float = COOL_MIN_S, max_s: float = COOL_MAX_S, poll_s: float = COOL_POLL_S) -> dict:
    """Wait at least min_s, then until the GPU is at or below target_c, giving up after max_s.

    With no target or no readable temperature this is a plain min_s wait. Returns what happened, for the manifest.
    """
    start, start_c = clock(), read_temp()
    sleep(min_s)
    temp = read_temp()
    while target_c is not None and temp is not None and temp > target_c and clock() - start < max_s:
        sleep(poll_s)
        temp = read_temp()
    timed_out = target_c is not None and temp is not None and temp > target_c
    return {"waited_s": clock() - start, "start_c": start_c, "end_c": temp, "target_c": target_c,
            "timed_out": timed_out}


def round_sig(x: float, digits: int = 3) -> float:
    """Round to `digits` significant figures, so trace names stay readable (e.g. 0.0512)."""
    return float(f"{x:.{digits - 1}e}")


def main() -> None:
    """Run the whole campaign with one loaded HF engine."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--traces", type=Path, default=Path("traces"))
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--order-seed", type=int, default=0, help="seed for the per-repetition run order")
    p.add_argument("--cool-to", type=float, default=None,
                   help="cooldown target in °C (default: idle GPU temperature at campaign start + tolerance)")
    p.add_argument("--cool-tol", type=float, default=COOL_TOL_C)
    args = p.parse_args()

    from bench.engines.hf import MODEL, HFEngine
    idle_c = gpu_temperature()  # before loading the model, so it is the idle baseline
    engine = HFEngine()
    config = {"model": MODEL, "dtype": "bfloat16", "attn": "sdpa", "batch": 1}
    campaign = args.results_root / f"{datetime.now():%Y%m%d-%H%M%S}_hf_baseline"
    campaign.mkdir(parents=True, exist_ok=False)
    target_c = args.cool_to if args.cool_to is not None else (idle_c + args.cool_tol if idle_c is not None else None)
    manifest = {"n": N, "seed": SEED, "fractions": FRACTIONS, "order_seed": args.order_seed,
                "idle_gpu_c": idle_c, "cool_target_c": target_c, "runs": []}

    def replay(trace: Path, rep: int) -> dict:
        """One run + summary, recorded in the manifest."""
        t0 = time.time()
        run_dir = run(engine, trace, "hf", config, results_root=args.results_root, telemetry=True)
        s = summarize_run(run_dir)
        manifest["runs"].append({"rep": rep, "trace": trace.name, "run_dir": str(run_dir),
                                 "wall_s": time.time() - t0, "achieved_rate_req_s": s["achieved_rate_req_s"],
                                 "overload": s["overload"]["flag"]})
        (campaign / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"rep {rep} {trace.name}: {time.time() - t0:.0f} s, achieved {s['achieved_rate_req_s']:.4f} req/s, "
              f"TTFT p50 {s['ttft_s']['p50']:.2f} s, TPOT p50 {s['tpot_s']['p50'] * 1e3:.1f} ms, "
              f"overload {s['overload']['flag']} -> {run_dir}", flush=True)
        manifest["runs"][-1]["cooldown_after"] = cool_down(target_c)
        (campaign / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return s

    burst = args.traces / f"sharegpt_n{N}_seed{SEED}_burst.jsonl"
    capacity = replay(burst, rep=1)["achieved_rate_req_s"]
    rates = [round_sig(f * capacity) for f in FRACTIONS]
    manifest.update({"capacity_req_s": capacity, "rates_req_s": rates})
    rate_traces = [args.traces / f"sharegpt_n{N}_seed{SEED}_rate{r:g}.jsonl" for r in rates]
    missing = [r for r, t in zip(rates, rate_traces) if not t.exists()]
    if missing:
        build_traces(Path("data/raw/ShareGPT_V3_unfiltered_cleaned_split.json"), args.traces, N, SEED, missing, False)
    print(f"capacity {capacity:.4f} req/s -> rates {rates}", flush=True)

    orders = rep_orders(burst, rate_traces, args.reps, args.order_seed)
    manifest["orders"] = [[t.name for t in order] for order in orders]
    for rep, order in enumerate(orders, start=1):
        for trace in order:
            replay(trace, rep)
    print(f"done: {campaign / 'manifest.json'}")


if __name__ == "__main__":
    main()
