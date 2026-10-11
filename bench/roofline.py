"""Measured roofline: memory bandwidth, bf16 matmul peak, a decode-shaped matmul sweep, and kernel launch overhead.

GPU timings use torch.cuda.Event after warmup and report the median of REPS repetitions (plus min and max).
Launch overhead is CPU wall time for LAUNCHES tiny kernels with one sync at the end, since that is what an eager
decode loop pays. The kernel count per decode step comes from profiling one HF Qwen2 decode step.
Each run writes a new results/roofline/<stamp>/ folder: roofline.json, env.json, roofline.png.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bench.env import gpu_snapshot, static_env

REPS, WARMUP = 20, 5
DTYPE = torch.bfloat16
COPY_BYTES = 512 * 2**20
MATMUL_SIZES = [1024, 2048, 3072, 4096, 6144, 8192]
SWEEP_K, SWEEP_N = 896, 4864  # Qwen2.5-0.5B hidden -> MLP intermediate
SWEEP_M = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
LAUNCHES = 1000
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def time_cuda(fn, reps: int = REPS, warmup: int = WARMUP) -> np.ndarray:
    """Seconds per call of fn, one CUDA event pair per repetition, after warmup calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for start, end in pairs:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return np.array([s.elapsed_time(e) for s, e in pairs]) / 1e3


def spread(x: np.ndarray) -> dict:
    """Median, min and max of a sample."""
    return {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


def measure_bandwidth(nbytes: int = COPY_BYTES) -> dict:
    """dst.copy_(src) on a bf16 tensor; each copy reads and writes nbytes, so BW = 2 * nbytes / time."""
    src = torch.randn(nbytes // 2, device="cuda", dtype=DTYPE)
    dst = torch.empty_like(src)
    t = time_cuda(lambda: dst.copy_(src))
    del src, dst
    return {"bytes": nbytes, "time_s": spread(t), "gb_s": spread(2 * nbytes / t / 1e9),
            "gb_s_at_median": 2 * nbytes / float(np.median(t)) / 1e9}


def measure_matmul_peak(sizes=MATMUL_SIZES) -> dict:
    """Square bf16 matmuls; FLOP/s = 2 n^3 / time. Peak is the best median over sizes."""
    rows = []
    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=DTYPE)
        b = torch.randn(n, n, device="cuda", dtype=DTYPE)
        t = time_cuda(lambda: a @ b)
        rows.append({"n": n, "time_s": spread(t), "tflop_s_at_median": 2 * n**3 / float(np.median(t)) / 1e12})
        del a, b
    return {"sizes": rows, "peak_tflop_s": max(r["tflop_s_at_median"] for r in rows)}


def intensity(m: int, k: int, n: int, elem_bytes: int = 2) -> float:
    """FLOP per byte of an [m,k] x [k,n] matmul that reads both inputs and writes the output once."""
    return 2 * m * k * n / (elem_bytes * (m * k + k * n + m * n))


def decode_sweep(k: int = SWEEP_K, n: int = SWEEP_N, ms=SWEEP_M) -> list[dict]:
    """F.linear(x [M,K], W [N,K]) as in a decode-step MLP up-projection, for growing batch M."""
    w = torch.randn(n, k, device="cuda", dtype=DTYPE)
    rows = []
    for m in ms:
        x = torch.randn(m, k, device="cuda", dtype=DTYPE)
        t = float(np.median(time_cuda(lambda: F.linear(x, w))))
        nbytes = 2 * (m * k + k * n + m * n)
        rows.append({"M": m, "intensity_flop_b": intensity(m, k, n), "time_s": t,
                     "tflop_s": 2 * m * k * n / t / 1e12, "gb_s": nbytes / t / 1e9})
    return rows


def launch_overhead(n: int = LAUNCHES, reps: int = REPS) -> dict:
    """CPU wall µs per launch: n in-place adds on a 1-element tensor, one sync at the end."""
    x = torch.zeros(1, device="cuda")
    for _ in range(n):
        x.add_(1.0)
    torch.cuda.synchronize()
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(n):
            x.add_(1.0)
        torch.cuda.synchronize()
        per.append((time.perf_counter() - t0) / n * 1e6)
    return {"launches": n, "us_per_launch": spread(np.array(per))}


def hf_decode_profile(prompt_len: int = 128, steps: int = 20) -> dict:
    """Kernel count of one HF Qwen2 bf16 decode step (batch 1), and its median wall time over `steps` steps."""
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, attn_implementation="sdpa",
                                                 local_files_only=True).cuda().eval()
    ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device="cuda")

    @torch.inference_mode()
    def decode(n):
        """Prefill, then n greedy decode steps; returns per-step wall seconds (token .item() before timestamp)."""
        out = model(ids, use_cache=True)
        past, tok = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            out = model(tok, past_key_values=past, use_cache=True)
            past, tok = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
            tok.item()
            times.append(time.perf_counter() - t0)
        return times

    decode(5)  # warmup
    step_s = np.array(decode(steps))
    with torch.inference_mode():
        out = model(ids, use_cache=True)
        past, tok = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            out = model(tok, past_key_values=past, use_cache=True)
            out.logits[:, -1].argmax(-1).item()
    gpu_events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    kernels = [e for e in gpu_events if not e.name.lower().startswith(("memcpy", "memset"))]
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    return {"prompt_len": prompt_len, "kernels_per_step": len(kernels), "gpu_ops_per_step": len(gpu_events),
            "step_wall_s": spread(step_s), "weight_bytes": weight_bytes}


def plot(sweep: list[dict], bw_gb_s: float, peak_tflop_s: float, path: Path) -> None:
    """Log-log achieved TFLOP/s vs intensity for the decode sweep, under the ceiling min(peak, I * BW)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    I = np.array([r["intensity_flop_b"] for r in sweep])
    y = np.array([r["tflop_s"] for r in sweep])
    ridge = peak_tflop_s * 1e3 / bw_gb_s  # FLOP/byte where I * BW = peak
    xs = np.logspace(np.log10(I.min() / 1.5), np.log10(max(I.max(), ridge) * 3), 200)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot(xs, np.minimum(peak_tflop_s, xs * bw_gb_s / 1e3), color=ink2, lw=1.5, ls="--", zorder=1)
    ax.plot(I, y, color=series, lw=2, marker="o", ms=5, mec=surface, mew=1.5, zorder=3)
    for r in sweep:
        if r["M"] in (1, 8, 64, 512):
            ax.annotate(f"M={r['M']}", (r["intensity_flop_b"], r["tflop_s"]), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color=ink2)
    ax.annotate(f"memory bound: {bw_gb_s:.0f} GB/s", (xs[0] * 1.3, xs[0] * 1.3 * bw_gb_s / 1e3),
                textcoords="offset points", xytext=(0, 10), fontsize=8, color=ink2, rotation=0)
    ax.annotate(f"bf16 peak: {peak_tflop_s:.1f} TFLOP/s  (ridge {ridge:.0f} FLOP/B)", (ridge, peak_tflop_s),
                textcoords="offset points", xytext=(4, 6), fontsize=8, color=ink2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)", color=ink2)
    ax.set_ylabel("achieved TFLOP/s", color=ink2)
    ax.set_title(f"Decode-shaped matmul [M, {SWEEP_K}] x [{SWEEP_K}, {SWEEP_N}], bf16", color=ink, loc="left",
                 fontsize=11)
    ax.grid(True, which="major", color=grid, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    """Run every measurement and write a new results/roofline/<stamp>/ folder."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-root", type=Path, default=Path("results/roofline"))
    p.add_argument("--skip-hf", action="store_true", help="skip the HF decode-step kernel count")
    args = p.parse_args()
    run_dir = args.results_root / f"{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)

    env = {"static": static_env(), "gpu_before": gpu_snapshot()}
    result = {"reps": REPS, "warmup": WARMUP, "bandwidth": measure_bandwidth(), "matmul": measure_matmul_peak(),
              "decode_sweep": decode_sweep(), "launch": launch_overhead()}
    if not args.skip_hf:
        result["hf_decode"] = hf_decode_profile()
    env["gpu_after"] = gpu_snapshot()

    bw = result["bandwidth"]["gb_s_at_median"]
    peak = result["matmul"]["peak_tflop_s"]
    derived = {"ridge_flop_b": peak * 1e3 / bw}
    if "hf_decode" in result:
        hf = result["hf_decode"]
        derived["batch1_bw_bound_tok_s"] = bw * 1e9 / hf["weight_bytes"]
        derived["launch_overhead_per_step_s"] = hf["kernels_per_step"] * result["launch"]["us_per_launch"]["median"] / 1e6
    result["derived"] = derived

    (run_dir / "roofline.json").write_text(json.dumps(result, indent=2))
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    plot(result["decode_sweep"], bw, peak, run_dir / "roofline.png")
    print(run_dir)


if __name__ == "__main__":
    main()
