"""Phase 3 sweep: achieved bandwidth of the Triton paged-decode kernel vs context length, batch, and block size.

One layer of Qwen2.5-0.5B in bf16 (q [S, 14, 64], pool [2, N, B, 64]), one decode token per sequence, block tables
shuffled. Bytes per call are the K and V the kernel must read, sum(context) x 2 heads x 64 x 2 (K, V) x 2 B = 512 B
per token, against the measured 156 GB/s copy bandwidth as the ceiling. For each point, single pass (1 split) and
the auto split-K choice are both timed.

Timing: CUDA events around --calls back-to-back launches, divided by --calls, median of --reps. When a call is
shorter than the launch cost (~11 µs per kernel), the GPU idles between calls and this measures launch throughput,
not bandwidth; that is the honest number for the engine, which launches the same way.

Writes results/decode_sweep/<stamp>/{results.json, env.json, bandwidth.png, split_k.png}. --plot <results.json>
redraws the figures from raw results only. Default sweep: 126 points, about a minute of GPU time.
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch

MEASURED_BW_GB_S = 156.0  # results/roofline/20261011-*
HQ, HKV, D = 14, 2, 64


def time_call_us(fn, calls: int, reps: int) -> float:
    """Median over reps of (event time for `calls` back-to-back fn() calls) / calls, in µs."""
    for _ in range(3):
        fn()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / calls)
    return statistics.median(times)


def measure(S: int, batch: int, B: int, calls: int, reps: int, seed: int = 0) -> list[dict]:
    """Single-pass and auto-split timings for one (context, batch, block size) point."""
    from engine.triton_decode import choose_splits, paged_decode_attention

    g = torch.Generator().manual_seed(seed)
    blocks_per_seq = -(-S // B)
    n_blocks = 2 * batch * blocks_per_seq
    perm = torch.randperm(n_blocks, generator=g)[:batch * blocks_per_seq].view(batch, blocks_per_seq)
    bt = perm.to(torch.int32).cuda()
    k = torch.randn(HKV, n_blocks, B, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(batch, HQ, D, device="cuda", dtype=torch.bfloat16)
    ctx = torch.full((batch,), S, dtype=torch.int32, device="cuda")
    kv_bytes = batch * S * HKV * D * 2 * 2
    rows = []
    for mode, splits in (("single", 1), ("split", choose_splits(batch, HKV, S))):
        us = time_call_us(lambda: paged_decode_attention(q, k, v, bt, ctx, D ** -0.5, S, num_splits=splits),
                          calls, reps)
        rows.append({"context_len": S, "batch": batch, "block_size": B, "mode": mode, "splits": splits,
                     "us_per_call": us, "kv_bytes": kv_bytes, "gb_s": kv_bytes / (us * 1e3),
                     "fraction_of_measured_bw": kv_bytes / (us * 1e3) / MEASURED_BW_GB_S})
    return rows


def plot(results_path: Path, out_dir: Path | None = None) -> list[Path]:
    """Draw bandwidth.png (small multiples by block size and mode) and split_k.png from a results.json, into out_dir
    (default: next to the results)."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bench.plots import GRID, INK, INK2, SURFACE, style_axes

    rows = json.loads(results_path.read_text())["rows"]
    out_dir = Path(out_dir or results_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    series = {1: "#2a78d6", 8: "#eb6834", 32: "#1baf7a"}  # reference palette slots 1-3, by batch
    blocks = sorted({r["block_size"] for r in rows})
    fig, axes = plt.subplots(2, len(blocks), figsize=(4 * len(blocks), 6.4), dpi=150, facecolor=SURFACE,
                             sharex=True, sharey=True, squeeze=False)
    for col, B in enumerate(blocks):
        for row, mode in enumerate(("single", "split")):
            ax = axes[row][col]
            style_axes(ax, f"block {B}, {'single pass' if mode == 'single' else 'auto split-K'}")
            ax.axhline(MEASURED_BW_GB_S, color=INK2, lw=1, ls="--", zorder=1)
            if row == 0 and col == 0:
                ax.annotate("measured copy BW 156 GB/s", (0, MEASURED_BW_GB_S), xycoords=("axes fraction", "data"),
                            xytext=(4, 4), textcoords="offset points", color=INK2, fontsize=8)
            for batch, color in series.items():
                pts = sorted((r["context_len"], r["gb_s"]) for r in rows
                             if r["block_size"] == B and r["mode"] == mode and r["batch"] == batch)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, lw=2, zorder=2, label=f"batch {batch}")
                ax.scatter(xs, ys, s=22, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
                ax.annotate(f"batch {batch}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                            color=INK2, fontsize=8, va="center")
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, MEASURED_BW_GB_S * 1.15)
            if row == 1:
                ax.set_xlabel("context length (tokens)", color=INK2, fontsize=8)
            if col == 0:
                ax.set_ylabel("achieved K/V read bandwidth (GB/s)", color=INK2, fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Triton paged decode, one layer of Qwen2.5-0.5B (bf16): bandwidth vs context",
                 color=INK, x=0.01, y=0.995, ha="left", fontsize=11)
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", ncol=len(series),
               bbox_to_anchor=(0.005, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    bw_path = out_dir / "bandwidth.png"
    fig.savefig(bw_path, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150, facecolor=SURFACE)
    B = 16 if 16 in blocks else blocks[0]
    style_axes(ax, f"batch 1, block {B}: time per call, single pass vs split-K")
    for mode, color, label in (("single", "#2a78d6", "single pass"), ("split", "#eb6834", "auto split-K")):
        pts = sorted((r["context_len"], r["us_per_call"], r["splits"]) for r in rows
                     if r["block_size"] == B and r["mode"] == mode and r["batch"] == 1)
        xs, ys, sp = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2, label=label, zorder=2)
        ax.scatter(xs, ys, s=22, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
        ax.annotate(label + (f" ({sp[-1]} splits)" if mode == "split" else ""), (xs[-1], ys[-1]), xytext=(4, 0),
                    textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(plain)
    ax.tick_params(which="both", colors=INK2, labelsize=8)
    ax.set_xlabel("context length (tokens)", color=INK2, fontsize=8)
    ax.set_ylabel("µs per call (one layer)", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    ax.grid(True, which="minor", color=GRID, lw=0.4)
    fig.tight_layout()
    sk_path = out_dir / "split_k.png"
    fig.savefig(sk_path, facecolor=SURFACE)
    plt.close(fig)
    return [bw_path, sk_path]


def main() -> None:
    """Run the sweep (or only redraw with --plot) and write results and figures."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--contexts", default="128,256,512,1024,2048,4096,8192")
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--block-sizes", default="8,16,32")
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--results-root", type=Path, default=Path("results/decode_sweep"))
    p.add_argument("--plot", type=Path, default=None, help="only redraw figures from this results.json")
    args = p.parse_args()
    if args.plot:
        print(*plot(args.plot), sep="\n")
        return

    from bench.env import gpu_snapshot, static_env
    out_dir = args.results_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    env = {"static": static_env(), "gpu_before": gpu_snapshot()}
    rows = []
    with torch.inference_mode():
        for B in map(int, args.block_sizes.split(",")):
            for batch in map(int, args.batches.split(",")):
                for S in map(int, args.contexts.split(",")):
                    for r in measure(S, batch, B, args.calls, args.reps):
                        rows.append(r)
                        print(f"B={B:2d} batch={batch:2d} S={S:5d} {r['mode']:6s} (splits {r['splits']:2d}): "
                              f"{r['us_per_call']:8.1f} µs, {r['gb_s']:6.1f} GB/s "
                              f"({r['fraction_of_measured_bw']:.0%})", flush=True)
    env["gpu_after"] = gpu_snapshot()
    results = out_dir / "results.json"
    results.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "rows": rows}, indent=2))
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(*plot(results), sep="\n")


if __name__ == "__main__":
    main()
