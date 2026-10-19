"""Profile batch-1 decode steps of an engine (torch.profiler, CPU + CUDA) and split each step's wall time into parts.

Per profiled step (medians over steps): wall time, GPU busy time (union of kernel intervals), kernel count, kernel
launch API calls and their CPU time, time blocked in sync calls (the .item() copy waiting for the GPU), and the
remainder (Python, nn.Module, and aten dispatch). Unprofiled steps are timed too, to show the profiler's overhead.

Works with any engine whose step() decodes one token for a single running request (--engine hf, contiguous,
or continuous: batch 1 through the continuous engine, i.e. the Triton decode kernel at one sequence). Writes
results/profile/<stamp>_<engine>/{summary.json, key_averages.txt, env.json}, plus trace.json.gz for Perfetto
(gitignored). One model load and a few hundred decode steps: well under a minute.
"""

import argparse
import gzip
import json
import shutil
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function

from bench.env import gpu_snapshot, static_env
from bench.runner import load_trace

STEP_LABEL = "decode_step"
LAUNCH_APIS = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx"}
SYNC_APIS = {"cudaMemcpyAsync", "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"}


def interval_union_us(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by a set of (start, end) intervals; overlaps are counted once."""
    total, cur_start, cur_end = 0.0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def breakdown(events) -> list[dict]:
    """Per-step accounting from profiler FunctionEvents, assigning events to steps by time.

    Every kernel a step launches finishes inside that step, because the step ends with .item(), which waits for the
    GPU. So a kernel belongs to the step whose CPU range contains its start.
    """
    steps = sorted((e.time_range.start, e.time_range.end) for e in events
                   if e.name == STEP_LABEL and e.device_type == DeviceType.CPU)
    rows = [{"wall_us": end - start, "kernel_intervals": [], "launch_calls": 0, "launch_api_us": 0.0,
             "sync_us": 0.0} for start, end in steps]

    def step_of(t: float) -> int | None:
        """Index of the step whose range contains t."""
        for i, (start, end) in enumerate(steps):
            if start <= t <= end:
                return i
        return None

    for e in events:
        i = step_of(e.time_range.start)
        if i is None:
            continue
        if e.device_type == DeviceType.CUDA:
            # record_function also emits a GPU-side annotation spanning the whole step; it is not a kernel.
            if e.name != STEP_LABEL:
                rows[i]["kernel_intervals"].append((e.time_range.start, e.time_range.end))
        elif e.name in LAUNCH_APIS:
            rows[i]["launch_calls"] += 1
            rows[i]["launch_api_us"] += e.time_range.elapsed_us()
        elif e.name in SYNC_APIS:
            rows[i]["sync_us"] += e.time_range.elapsed_us()
    for r in rows:
        intervals = r.pop("kernel_intervals")
        r["n_kernels"] = len(intervals)
        r["gpu_busy_us"] = interval_union_us(intervals)
        r["other_cpu_us"] = r["wall_us"] - r["launch_api_us"] - r["sync_us"]
    return rows


def median_row(rows: list[dict]) -> dict:
    """Field-wise median over steps."""
    return {k: statistics.median(r[k] for r in rows) for k in rows[0]}


def main() -> None:
    """Load the engine, prefill one trace prompt, time unprofiled decode steps, then profile a few more."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engine", choices=["hf", "contiguous", "continuous"], default="hf")
    p.add_argument("--trace", type=Path, default=Path("traces/sharegpt_n100_seed0_burst.jsonl"))
    p.add_argument("--request", type=int, default=0, help="index of the trace request whose prompt is used")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--unprofiled", type=int, default=100)
    p.add_argument("--profiled", type=int, default=20)
    p.add_argument("--results-root", type=Path, default=Path("results/profile"))
    args = p.parse_args()

    out_dir = args.results_root / f"{datetime.now():%Y%m%d-%H%M%S}_{args.engine}"
    out_dir.mkdir(parents=True, exist_ok=False)
    env = {"static": static_env(args.trace), "gpu_before": gpu_snapshot()}

    from bench.engines import load_engine
    engine, config = load_engine(args.engine)
    MODEL = config["model"]
    prompt = load_trace(args.trace)[args.request].prompt_ids
    total = 1 + args.warmup + args.unprofiled + args.profiled
    engine.add_request(0, prompt, total)
    for _ in range(1 + args.warmup):  # prefill + warmup decodes
        engine.step()

    unprofiled_us = []
    for _ in range(args.unprofiled):
        t0 = time.perf_counter()
        engine.step()
        unprofiled_us.append((time.perf_counter() - t0) * 1e6)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.profiled):
            with record_function(STEP_LABEL):
                engine.step()
    env["gpu_after"] = gpu_snapshot()

    rows = breakdown(prof.events())
    med = median_row(rows)
    summary = {
        "engine": args.engine, "model": MODEL, "prompt_len": len(prompt), "warmup": args.warmup,
        "context_len_at_profile": len(prompt) + 1 + args.warmup + args.unprofiled,
        "unprofiled_step_us": {"p50": statistics.median(unprofiled_us), "min": min(unprofiled_us),
                               "max": max(unprofiled_us), "n": len(unprofiled_us)},
        "profiled_step_median": med,
        "profiled_steps": rows,
        "fractions_of_profiled_wall": {k: med[k] / med["wall_us"]
                                       for k in ["gpu_busy_us", "launch_api_us", "sync_us", "other_cpu_us"]},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    (out_dir / "key_averages.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
    raw = out_dir / "trace.json"
    prof.export_chrome_trace(str(raw))
    with open(raw, "rb") as src, gzip.open(out_dir / "trace.json.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    raw.unlink()

    print(f"unprofiled step p50 {summary['unprofiled_step_us']['p50'] / 1e3:.2f} ms; profiled medians: "
          f"wall {med['wall_us'] / 1e3:.2f} ms, GPU busy {med['gpu_busy_us'] / 1e3:.2f} ms, "
          f"{med['n_kernels']:.0f} kernels, {med['launch_calls']:.0f} launches = {med['launch_api_us'] / 1e3:.2f} ms, "
          f"sync {med['sync_us'] / 1e3:.2f} ms, other CPU {med['other_cpu_us'] / 1e3:.2f} ms -> {out_dir}")


if __name__ == "__main__":
    main()
