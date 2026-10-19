"""Cost of the Phase 2 gather-based paged attention versus contiguous attention, for one decode token per sequence.

Shapes are one layer of Qwen2.5-0.5B in bf16: q [1, 14, 64] per sequence; K and V per sequence are 2 heads x S x 64,
so 512 * S bytes per layer (256 KiB at S = 512; across 24 layers, 6 MiB). Per context length S and batch size:
  contiguous: attention() over [2, S, 64] views of a contiguous cache (reads K/V once)
  paged:      gather the sequence's 16-token blocks into contiguous [2, S, 64] copies, then the same attention()
  gather:     the gather alone (reads and writes K/V: 2 x 512 * S bytes per layer per sequence)
Two views per point: GPU kernel time (summed kernel durations from torch.profiler, median per call, immune to launch
gaps) with the kernel count per call, and wall time (CUDA events around one call, launches included). The kernel
view is the gather's GPU cost; the count times the per-launch cost (~12-20 µs here, varying with a WSL host state, see
the design log) is its launch cost. Wall times are CUDA-event medians over --reps after warmup, per layer. Block tables are
shuffled. Events time the GPU stream, which includes the gaps where the GPU waits for the CPU to launch the next
kernel; at small S those gaps dominate (a few kernels at ~11 µs each), so small-S numbers measure launch cost, and only
large S approaches a bandwidth measurement. Both matter: batch-1 decode is CPU-bound. The gap between paged and contiguous is what the Phase 3 kernel, which reads blocks in place, removes.

Writes results/paged_gather/<stamp>/{results.json, env.json}. A few seconds of GPU time at the default sizes.
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch

from bench.env import gpu_snapshot, static_env
from engine.attention import attention
from engine.block_manager import BlockManager
from engine.config import ModelConfig
from engine.kv_cache import ContiguousKVCache
from engine.paged_cache import PagedKVCache, build_metadata

QWEN_05B = ModelConfig(vocab_size=151936, hidden_size=896, intermediate_size=4864, num_layers=1, num_q_heads=14,
                       num_kv_heads=2, head_dim=64, rms_norm_eps=1e-6, rope_theta=1e6, max_position_embeddings=32768,
                       tie_word_embeddings=True)  # one layer: the benchmark times a single layer's attention


def kernel_profile(fn, reps: int) -> tuple[float, float]:
    """(median summed GPU kernel µs per call, kernels per call) from torch.profiler: immune to launch gaps."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, record_function
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            with record_function("call"):
                fn()
            torch.cuda.synchronize()
    events = prof.events()
    calls = sorted((e.time_range.start, e.time_range.end) for e in events
                   if e.name == "call" and e.device_type == DeviceType.CPU)
    per_call = [[0.0, 0] for _ in calls]
    for e in events:
        if e.device_type != DeviceType.CUDA or e.name == "call":
            continue
        for i, (a, b) in enumerate(calls):
            if a <= e.time_range.start <= b:
                per_call[i][0] += e.time_range.elapsed_us()
                per_call[i][1] += 1
                break
    return statistics.median(x[0] for x in per_call), statistics.median(x[1] for x in per_call)


def time_us(fn, reps: int, warmup: int = 5) -> float:
    """Median CUDA-event time of fn() in µs."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return statistics.median(times)


def measure(S: int, batch: int, block_size: int, reps: int, seed: int = 0) -> dict:
    """One (context length, batch) point: µs per layer for contiguous, paged, and gather-only."""
    cfg, dev, dt = QWEN_05B, "cuda", torch.bfloat16
    g = torch.Generator().manual_seed(seed)
    nb = -(-S // block_size)
    m = BlockManager(nb * batch * 2, block_size)
    for s in range(m.num_blocks):  # shuffle the free list so tables are scattered
        m.allocate(10_000 + s, block_size)
    for s in torch.randperm(m.num_blocks, generator=g).tolist():
        m.free(10_000 + s)
    paged = PagedKVCache(cfg, m.num_blocks, block_size, dev, dt)
    paged.k.normal_()
    paged.v.normal_()
    for s in range(batch):
        m.allocate(s, S)
    meta = build_metadata(m, [(s, S - 1, 1) for s in range(batch)], dev)
    paged.begin(batch, meta)
    contig = [ContiguousKVCache(cfg, S, dev, dt) for _ in range(batch)]
    for c in contig:
        c.k.normal_()
        c.v.normal_()
        c.advance(S)
    q = torch.randn(batch, 14, 64, device=dev, dtype=dt)
    pos = torch.full((1,), S - 1, device=dev)
    scale = 64 ** -0.5

    def run_contiguous():
        """Per sequence: attention over its contiguous cache views."""
        for i, c in enumerate(contig):
            k, v = c.read(0)
            attention(q[i:i + 1], k, v, pos, c.positions(), scale)

    def run_paged():
        """The paged cache's attend(): gather + attention per sequence."""
        paged.attend(0, q, meta.positions, scale)

    def run_gather():
        """Gather only."""
        for i in range(batch):
            paged.gather(0, i)

    kv_bytes = 2 * cfg.num_kv_heads * S * cfg.head_dim * 2 * batch  # K and V, one layer, all sequences
    r = {"context_len": S, "batch": batch, "block_size": block_size, "kv_bytes_per_layer": kv_bytes,
         "contiguous_us": time_us(run_contiguous, reps), "paged_us": time_us(run_paged, reps),
         "gather_us": time_us(run_gather, reps)}
    for name, fn in (("contiguous", run_contiguous), ("paged", run_paged), ("gather", run_gather)):
        r[f"{name}_kernel_us"], r[f"{name}_kernels"] = kernel_profile(fn, min(reps, 20))
    r["paged_over_contiguous_kernel"] = r["paged_kernel_us"] / r["contiguous_kernel_us"]
    r["paged_over_contiguous"] = r["paged_us"] / r["contiguous_us"]
    r["gather_gb_s"] = 2 * kv_bytes / (r["gather_us"] * 1e3)  # read + write
    r["paged_step_overhead_ms_24_layers"] = 24 * (r["paged_us"] - r["contiguous_us"]) / 1e3
    return r


def main() -> None:
    """Sweep context length x batch and write the results."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--contexts", default="128,512,1024,2048,4096")
    p.add_argument("--batches", default="1,8")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--results-root", type=Path, default=Path("results/paged_gather"))
    args = p.parse_args()

    out_dir = args.results_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    env = {"static": static_env(), "gpu_before": gpu_snapshot()}
    rows = []
    with torch.inference_mode():
        for batch in map(int, args.batches.split(",")):
            for S in map(int, args.contexts.split(",")):
                r = measure(S, batch, args.block_size, args.reps)
                rows.append(r)
                print(f"S={S:5d} batch={batch:2d}: GPU kernel time contiguous {r['contiguous_kernel_us']:6.1f} µs "
                      f"({r['contiguous_kernels']:.0f} kernels), paged {r['paged_kernel_us']:6.1f} µs "
                      f"({r['paged_kernels']:.0f}), {r['paged_over_contiguous_kernel']:.2f}x; wall incl. launches "
                      f"{r['contiguous_us']:7.1f} vs {r['paged_us']:7.1f} µs", flush=True)
    env["gpu_after"] = gpu_snapshot()
    (out_dir / "results.json").write_text(json.dumps({"args": vars(args) | {"results_root": str(args.results_root)},
                                                      "rows": rows}, indent=2))
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(out_dir)


if __name__ == "__main__":
    main()
