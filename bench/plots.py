"""Rebuild analysis and plots from raw run records. Nothing here is edited by hand; rerun to regenerate.

Phase 0 (HF baseline campaign): decomposes each request's TTFT into queue wait and prefill, checks the measured
mean wait against M/G/1 (Pollaczek-Khinchine), fits prefill time vs prompt length, and draws four figures into
docs/figures/phase0/ plus analysis.json.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.metrics import load_jsonl

# Categorical slots 1-3 (validated all-pairs, light mode). Color follows the trace, never its rank.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRACE_STYLE = {  # trace key -> (label, color)
    "rate0.0441": ("0.3×", "#2a78d6"),
    "rate0.103": ("0.7×", "#eb6834"),
    "rate0.176": ("1.2×", "#1baf7a"),
    "burst": ("burst", "#8a8984"),
}
WARMUP_FRAC = 0.1


def trace_key(trace_name: str) -> str:
    """'sharegpt_n100_seed0_rate0.103.jsonl' -> 'rate0.103'."""
    return trace_name.removesuffix(".jsonl").split("_")[-1]


def load_campaign(manifest_path: Path) -> list[dict]:
    """Every run in a campaign manifest, with its raw records, summary and wall-clock start."""
    manifest = json.loads(Path(manifest_path).read_text())
    runs = []
    for entry in manifest["runs"]:
        d = Path(entry["run_dir"])
        runs.append({"rep": entry["rep"], "trace": trace_key(entry["trace"]), "dir": d,
                     "requests": load_jsonl(d / "requests.jsonl"), "steps": load_jsonl(d / "steps.jsonl"),
                     "summary": json.loads((d / "summary.json").read_text()),
                     "wall_start": json.loads((d / "run.json").read_text())["wall_start_unix"]})
    return runs


def decompose(requests: list[dict], steps: list[dict]) -> list[dict]:
    """Per request: queue wait, prefill time, and service time, using the step that produced the first token.

    The runner stamps a token with its step's end_s, so the prefill step is the one whose end_s equals token_s[0].
    wait = prefill start - scheduled arrival (includes waiting for the step in progress when it arrived).
    """
    by_end = {s["end_s"]: s for s in steps}
    out = []
    for r in requests:
        pre = by_end[r["token_s"][0]]
        out.append({"id": r["id"], "arrival_s": r["arrival_s"], "prompt_len": r["prompt_len"],
                    "output_len": r["output_len"], "wait_s": pre["start_s"] - r["arrival_s"],
                    "prefill_s": pre["end_s"] - pre["start_s"], "service_s": r["token_s"][-1] - pre["start_s"]})
    return out


def kept(rows: list[dict]) -> list[dict]:
    """Drop the first WARMUP_FRAC by arrival, matching bench/metrics.py."""
    rows = sorted(rows, key=lambda r: (r["arrival_s"], r["id"]))
    return rows[int(len(rows) * WARMUP_FRAC):]


def mg1_check(rows: list[dict], offered: float) -> dict:
    """Measured mean wait vs Pollaczek-Khinchine: Wq = λ E[S²] / (2 (1 - ρ)), with ρ = λ E[S]."""
    s = np.array([r["service_s"] for r in rows])
    rho = offered * s.mean()
    predicted = offered * (s**2).mean() / (2 * (1 - rho)) if rho < 1 else None
    return {"lambda": offered, "mean_S": float(s.mean()), "cv_S": float(s.std() / s.mean()), "rho": float(rho),
            "predicted_wait_s": predicted, "measured_wait_s": float(np.mean([r["wait_s"] for r in rows]))}


def style_axes(ax, title: str) -> None:
    """Recessive grid and axes, left-aligned title."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=10)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def figure(nrows=1, ncols=1, size=(7.5, 4.2)):
    """A figure on the chart surface."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=size, dpi=150, facecolor=SURFACE, squeeze=False)
    return plt, fig, axes


def plot_latency_vs_load(runs, capacity, path):
    """TTFT and TPOT percentiles vs offered load, one panel each (no dual axis); every rep is a dot."""
    plt, fig, axes = figure(1, 2, (9, 3.8))
    panels = [("TTFT (s, log)", "ttft_s", 1, True), ("TPOT (ms)", "tpot_s", 1e3, False)]
    pct_style = [("p50", "#2a78d6"), ("p99", "#eb6834")]
    for ax, (title, key, scale, log) in zip(axes[0], panels):
        style_axes(ax, title)
        for pct, color in pct_style:
            pts = defaultdict(list)
            for r in runs:
                if r["trace"] != "burst":
                    pts[r["summary"]["offered_rate_req_s"] / capacity].append(r["summary"][key][pct] * scale)
            xs = sorted(pts)
            ax.plot(xs, [np.mean(pts[x]) for x in xs], color=color, lw=2, zorder=2)
            for x in xs:
                ax.scatter([x] * len(pts[x]), pts[x], s=22, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
            ax.annotate(pct, (xs[-1], np.mean(pts[xs[-1]])), xytext=(6, 0), textcoords="offset points",
                        color=INK2, fontsize=8, va="center")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("offered load ρ (offered rate / burst capacity)", color=INK2, fontsize=8)
        ax.set_xlim(0.2, 1.35)
    fig.suptitle("HF baseline, batch 1: latency vs load (3 reps per rate)", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_ttft_cdf(runs, path):
    """Empirical CDF of TTFT per rate, all reps pooled (kept requests only), log x."""
    plt, fig, axes = figure(size=(7.5, 4.2))
    ax = axes[0, 0]
    style_axes(ax, "TTFT distribution per rate (3 reps pooled, warmup dropped)")
    for key in ["rate0.0441", "rate0.103", "rate0.176"]:
        label, color = TRACE_STYLE[key]
        ttft = np.sort([r["token_s"][0] - r["arrival_s"] for run in runs if run["trace"] == key
                        for r in kept(run["requests"])])
        y = np.arange(1, len(ttft) + 1) / len(ttft)
        ax.plot(ttft, y, color=color, lw=2)
        i = int(0.75 * len(ttft))
        ax.annotate(label, (ttft[i], y[i]), xytext=(-24, 6), textcoords="offset points", color=INK2, fontsize=9)
    ax.axvline(2.0, color=INK2, lw=1, ls="--")
    ax.annotate("SLO 2 s", (2.0, 0.04), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("TTFT (s)", color=INK2, fontsize=8)
    ax.set_ylabel("fraction of requests", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_queue_depth(runs, path):
    """Waiting-queue depth over time for rep 1 of each rate (small multiples, shared y)."""
    keys = ["rate0.0441", "rate0.103", "rate0.176"]
    plt, fig, axes = figure(3, 1, (7.5, 6))
    ymax = max(s["waiting"] for r in runs if r["trace"] in keys and r["rep"] == 1 for s in r["steps"])
    for ax, key in zip(axes[:, 0], keys):
        label, color = TRACE_STYLE[key]
        run = next(r for r in runs if r["trace"] == key and r["rep"] == 1)
        t = [s["end_s"] / 60 for s in run["steps"]]
        ax.step(t, [s["waiting"] for s in run["steps"]], where="post", color=color, lw=1.5)
        style_axes(ax, f"{label}  (offered {run['summary']['offered_rate_req_s']:.4f} req/s, "
                       f"overload {run['summary']['overload']['flag']})")
        ax.set_ylim(0, ymax + 1)
        ax.set_ylabel("waiting", color=INK2, fontsize=8)
    axes[-1, 0].set_xlabel("minutes since run start", color=INK2, fontsize=8)
    fig.suptitle("Requests waiting for prefill (rep 1)", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_decode_drift(runs, path):
    """Per-request TPOT against wall-clock time over the whole campaign, colored by trace."""
    plt, fig, axes = figure(size=(9, 4))
    ax = axes[0, 0]
    t0 = min(r["wall_start"] for r in runs)
    style_axes(ax, "Per-request TPOT over the campaign (CPU-bound: drifts with machine state)")
    for key in ["burst", "rate0.0441", "rate0.103", "rate0.176"]:
        label, color = TRACE_STYLE[key]
        xs, ys = [], []
        for run in (r for r in runs if r["trace"] == key):
            for q in run["requests"]:
                xs.append((run["wall_start"] + q["token_s"][0] - t0) / 3600)
                ys.append((q["token_s"][-1] - q["token_s"][0]) / (len(q["token_s"]) - 1) * 1e3)
        ax.scatter(xs, ys, s=8, color=color, alpha=0.7, lw=0, label=label)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=4, loc="upper left", markerscale=2)
    ax.set_xlabel("hours since campaign start", color=INK2, fontsize=8)
    ax.set_ylabel("TPOT (ms)", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def phase0(manifest_path: Path, out_dir: Path) -> dict:
    """Analysis numbers and figures for the HF baseline campaign."""
    runs = load_campaign(manifest_path)
    capacity = json.loads(Path(manifest_path).read_text())["capacity_req_s"]
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run, prefill_pts, prefill_by_id = [], [], defaultdict(dict)
    for run in runs:
        rows = decompose(run["requests"], run["steps"])
        for r in rows:
            prefill_by_id[(run["trace"], run["rep"])][r["id"]] = r["prefill_s"]
        if run["trace"] == "burst":  # back-to-back, so the GPU is never idle before a prefill
            prefill_pts += [(r["prompt_len"], r["prefill_s"]) for r in rows]
        k = kept(rows)
        entry = {"trace": run["trace"], "rep": run["rep"], "dir": str(run["dir"]),
                 "mean_wait_s": float(np.mean([r["wait_s"] for r in k])),
                 "mean_prefill_ms": float(np.mean([r["prefill_s"] for r in k]) * 1e3),
                 "wait_share_of_ttft": float(np.sum([r["wait_s"] for r in k]) /
                                             np.sum([r["wait_s"] + r["prefill_s"] for r in k]))}
        if run["trace"] != "burst":
            entry["mg1"] = mg1_check(k, run["summary"]["offered_rate_req_s"])
        per_run.append(entry)

    p, t = np.array(prefill_pts).T
    slope, intercept = np.polyfit(p, t * 1e3, 1)
    # Same request ids appear in every trace, so prefill can be compared request by request against burst.
    paired = {}
    for key in ["rate0.0441", "rate0.103", "rate0.176"]:
        diffs = [(prefill_by_id[(key, rep)][i] - prefill_by_id[("burst", rep)][i]) * 1e3
                 for rep in (1, 2, 3) for i in prefill_by_id[("burst", rep)]]
        paired[key] = {"median_extra_ms": float(np.median(diffs)), "mean_extra_ms": float(np.mean(diffs)),
                       "frac_over_50ms": float(np.mean(np.array(diffs) > 50))}
    analysis = {"manifest": str(manifest_path), "capacity_req_s": capacity, "runs": per_run,
                "prefill_fit_burst_ms": {"intercept": float(intercept), "per_prompt_token": float(slope),
                                         "prompt_len_range": [int(p.min()), int(p.max())]},
                "prefill_extra_vs_burst_same_request": paired}
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))

    plot_latency_vs_load(runs, capacity, out_dir / "latency_vs_load.png")
    plot_ttft_cdf(runs, out_dir / "ttft_cdf.png")
    plot_queue_depth(runs, out_dir / "queue_depth.png")
    plot_decode_drift(runs, out_dir / "decode_drift.png")
    return analysis


def main() -> None:
    """CLI: regenerate Phase 0 analysis and figures."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, default=Path("results/20261011-163739_hf_baseline/manifest.json"))
    p.add_argument("--out-dir", type=Path, default=Path("docs/figures/phase0"))
    args = p.parse_args()
    a = phase0(args.manifest, args.out_dir)
    fit = a["prefill_fit_burst_ms"]
    print(f"burst prefill ≈ {fit['intercept']:.1f} ms + {fit['per_prompt_token'] * 1e3:.1f} µs/prompt token "
          f"(prompt {fit['prompt_len_range']})")
    for key, d in a["prefill_extra_vs_burst_same_request"].items():
        print(f"prefill extra vs burst, same request, {key}: median {d['median_extra_ms']:.1f} ms, "
              f"mean {d['mean_extra_ms']:.1f} ms, {d['frac_over_50ms']:.0%} over 50 ms")
    print(f"{'trace':11s} rep  wait(s)  prefill(ms)  wait share | M/G/1: ρ    E[S]   CV(S)  predicted  measured")
    for r in a["runs"]:
        m = r.get("mg1")
        mg = (f"{m['rho']:.2f}  {m['mean_S']:5.2f}  {m['cv_S']:.2f}   "
              f"{m['predicted_wait_s'] if m['predicted_wait_s'] is None else round(m['predicted_wait_s'], 2)!s:>8}"
              f"  {m['measured_wait_s']:7.2f}") if m else ""
        print(f"{r['trace']:11s} {r['rep']}  {r['mean_wait_s']:8.2f}  {r['mean_prefill_ms']:8.1f}  "
              f"{r['wait_share_of_ttft']:9.3f}   | {mg}")


if __name__ == "__main__":
    main()
