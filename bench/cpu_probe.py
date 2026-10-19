"""CPU-only probe for the bimodal-TPOT hypothesis: is single-thread Python speed itself bimodal on this host?

Phase 0 found HF decode TPOT switching in blocks between about 24 and 33 ms (ratio about 1.37), and hypothesized that
Windows moves the WSL vCPU running the decode thread between P-cores and E-cores. This probe takes the GPU out: it
times a fixed pure-Python workload (about 1 ms per chunk, interpreter-bound like HF's module overhead) in a loop
while bench/telemetry.py records which host core class is busiest. If the hypothesis holds, chunk time is bimodal
with a similar ratio, and its slow windows line up with samples whose busiest core is an E-core.

Writes results/cpu_probe/<stamp>/{windows.csv, summary.json, env.json, telemetry_*.csv, telemetry.json}. Runs one
thread for --duration seconds (default 120): one core busy, nothing else.
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from bench.env import mem_snapshot, static_env
from bench.telemetry import Telemetry, cpu_snapshot, parse_cpu_csv, summarize_cpu_sample

CHUNK_ITERS = 20_000
WINDOW_S = 0.25


def chunk() -> int:
    """Fixed interpreter-bound work: attribute lookups, calls, and integer arithmetic."""
    acc = 0
    for i in range(CHUNK_ITERS):
        acc = (acc + i * 7) % 1_000_003
    return acc


def windows(chunks: list[tuple[float, float]], window_s: float) -> list[dict]:
    """Group (start_unix_s, duration_s) chunks into fixed windows; median chunk µs per window."""
    if not chunks:
        return []
    t0, rows, cur, cur_start = chunks[0][0], [], [], chunks[0][0]
    for start, dur in chunks:
        if start - cur_start >= window_s and cur:
            rows.append({"unix_s": cur_start, "median_chunk_us": statistics.median(cur) * 1e6, "n": len(cur)})
            cur, cur_start = [], t0 + window_s * int((start - t0) / window_s)
        cur.append(dur)
    rows.append({"unix_s": cur_start, "median_chunk_us": statistics.median(cur) * 1e6, "n": len(cur)})
    return rows


def busiest_class_at(cpu_samples: dict[int, dict], unix_s: float) -> str | None:
    """Busiest host core class in the CPU telemetry sample closest to unix_s (within 2 s)."""
    if not cpu_samples:
        return None
    t = min(cpu_samples, key=lambda ms: abs(ms / 1e3 - unix_s))
    return summarize_cpu_sample(cpu_samples[t])["busiest"]["class"] if abs(t / 1e3 - unix_s) <= 2 else None


def main() -> None:
    """Run the probe and summarize chunk-time spread, overall and by busiest core class."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--results-root", type=Path, default=Path("results/cpu_probe"))
    args = p.parse_args()

    out_dir = args.results_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    env = {"static": static_env(), "cpu_before": cpu_snapshot(), "mem_before": mem_snapshot()}
    for _ in range(50):  # warm up the interpreter's specializations
        chunk()

    chunks = []
    with Telemetry(out_dir):
        end = time.time() + args.duration
        while (start := time.time()) < end:
            t = time.perf_counter()
            chunk()
            chunks.append((start, time.perf_counter() - t))
    env["cpu_after"] = cpu_snapshot()

    rows = windows(chunks, WINDOW_S)
    cpu_samples = parse_cpu_csv((out_dir / "telemetry_cpu.csv").read_text())
    for r in rows:
        r["busiest_class"] = busiest_class_at(cpu_samples, r["unix_s"])
    with open(out_dir / "windows.csv", "w") as f:
        f.write("unix_s,median_chunk_us,n,busiest_class\n")
        for r in rows:
            f.write(f"{r['unix_s']:.3f},{r['median_chunk_us']:.1f},{r['n']},{r['busiest_class']}\n")

    us = sorted(d * 1e6 for _, d in chunks)
    q = lambda f: us[min(len(us) - 1, int(f * len(us)))]  # noqa: E731
    by_class = {}
    for r in rows:
        by_class.setdefault(r["busiest_class"], []).append(r["median_chunk_us"])
    summary = {
        "duration_s": args.duration, "chunk_iters": CHUNK_ITERS, "n_chunks": len(us), "window_s": WINDOW_S,
        "chunk_us": {"p01": q(0.01), "p10": q(0.10), "p50": q(0.50), "p90": q(0.90), "p99": q(0.99)},
        "p90_over_p10": q(0.90) / q(0.10),
        "window_median_us_by_busiest_class": {
            str(c): {"n_windows": len(v), "p50": statistics.median(v)} for c, v in sorted(by_class.items(), key=str)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(summary, indent=2))
    print(out_dir)


if __name__ == "__main__":
    main()
