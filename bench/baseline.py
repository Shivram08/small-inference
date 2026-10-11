"""HF baseline campaign: burst run -> capacity -> rate traces at 0.3x/0.7x/1.2x -> 3 interleaved repetitions.

Repetition r runs every trace once, in the order burst, 0.3x, 0.7x, 1.2x. The first burst run measures capacity
(achieved rate on the burst trace, about 1/S), and it counts as repetition 1's burst. Interleaving spreads slow
drift (temperature, background load) across all rates instead of concentrating it in one. A manifest listing
every run folder is written to results/<stamp>_hf_baseline/manifest.json and updated after each run.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from bench.metrics import summarize_run
from bench.runner import run
from bench.trace import build_traces

FRACTIONS = (0.3, 0.7, 1.2)
N, SEED = 100, 0
COOLDOWN_S = 30


def round_sig(x: float, digits: int = 3) -> float:
    """Round to `digits` significant figures, so trace names stay readable (e.g. 0.0512)."""
    return float(f"{x:.{digits - 1}e}")


def main() -> None:
    """Run the whole campaign with one loaded HF engine."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--traces", type=Path, default=Path("traces"))
    p.add_argument("--results-root", type=Path, default=Path("results"))
    args = p.parse_args()

    from bench.engines.hf import MODEL, HFEngine
    engine = HFEngine()
    config = {"model": MODEL, "dtype": "bfloat16", "attn": "sdpa", "batch": 1}
    campaign = args.results_root / f"{datetime.now():%Y%m%d-%H%M%S}_hf_baseline"
    campaign.mkdir(parents=True, exist_ok=False)
    manifest = {"n": N, "seed": SEED, "fractions": FRACTIONS, "runs": []}

    def replay(trace: Path, rep: int) -> dict:
        """One run + summary, recorded in the manifest."""
        t0 = time.time()
        run_dir = run(engine, trace, "hf", config, results_root=args.results_root)
        s = summarize_run(run_dir)
        manifest["runs"].append({"rep": rep, "trace": trace.name, "run_dir": str(run_dir),
                                 "wall_s": time.time() - t0, "achieved_rate_req_s": s["achieved_rate_req_s"],
                                 "overload": s["overload"]["flag"]})
        (campaign / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"rep {rep} {trace.name}: {time.time() - t0:.0f} s, achieved {s['achieved_rate_req_s']:.4f} req/s, "
              f"TTFT p50 {s['ttft_s']['p50']:.2f} s, TPOT p50 {s['tpot_s']['p50'] * 1e3:.1f} ms, "
              f"overload {s['overload']['flag']} -> {run_dir}", flush=True)
        time.sleep(COOLDOWN_S)
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

    for rep in range(1, args.reps + 1):
        for trace in ([] if rep == 1 else [burst]) + rate_traces:
            replay(trace, rep)
    print(f"done: {campaign / 'manifest.json'}")


if __name__ == "__main__":
    main()
