"""Serving curves from campaign manifests: output throughput, p99 TPOT, and p99 ITL vs offered request rate.

Every point is recomputed from the run's raw logs (bench.metrics.summarize_run), never read from a cached summary.
One line per engine (mean over repetitions, with each repetition as a dot); burst runs are left out (they have no
offered rate). Takes one or more manifests, so engines measured in separate campaigns can share a figure as long as
they replayed the same traces.

Usage: python -m bench.rate_plots results/<campaign>/manifest.json [...] --out docs/figures/phase4
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.metrics import summarize_run

ENGINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # reference palette slots 1-4, in engine order
PANELS = [("output_throughput_tok_s", "output throughput (tok/s)", 1.0),
          ("tpot_s", "p99 TPOT (ms)", 1e3),
          ("itl_s", "p99 inter-token latency (ms)", 1e3)]


def collect(manifests: list[Path]) -> dict[str, dict[float, list[dict]]]:
    """engine -> offered trace rate -> list of run summaries."""
    data: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in manifests:
        for run in json.loads(Path(path).read_text())["runs"]:
            m = re.search(r"_rate([0-9.]+)\.jsonl$", run["trace"])
            if m:
                data[run.get("engine", "hf")][float(m.group(1))].append(summarize_run(Path(run["run_dir"])))
    return data


def value(summary: dict, key: str) -> float:
    """A panel's value: throughput as is, latency distributions at p99."""
    v = summary[key]
    return v["p99"] if isinstance(v, dict) else v


def plot(data: dict, out_dir: Path) -> Path:
    """Draw the three panels side by side and save rate_curves.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bench.plots import INK, INK2, SURFACE, style_axes

    out_dir.mkdir(parents=True, exist_ok=True)
    engines = list(data)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 3.8), dpi=150, facecolor=SURFACE)
    for ax, (key, label, scale) in zip(axes, PANELS):
        style_axes(ax, label)
        for engine, color in zip(engines, ENGINE_COLORS):
            rates = sorted(data[engine])
            reps = [[value(s, key) * scale for s in data[engine][r]] for r in rates]
            ax.plot(rates, [np.mean(r) for r in reps], color=color, lw=2, zorder=2, label=engine)
            for r, pts in zip(rates, reps):
                ax.scatter([r] * len(pts), pts, s=22, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
            ax.annotate(engine, (rates[-1], np.mean(reps[-1])), xytext=(4, 0), textcoords="offset points",
                        color=INK2, fontsize=8, va="center")
        ax.set_xlabel("offered rate (req/s)", color=INK2, fontsize=8)
        ax.set_ylim(bottom=0)
    if len(engines) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK2, loc="upper left",
                   ncol=len(engines), bbox_to_anchor=(0.005, 0.93))
    fig.suptitle("Serving vs offered load (ShareGPT n = 100, mean of repetitions; dots are repetitions)",
                 color=INK, x=0.01, y=0.99, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88 if len(engines) > 1 else 0.94))
    path = out_dir / "rate_curves.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    """CLI: manifests in, figure out."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("manifests", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(plot(collect(args.manifests), args.out))


if __name__ == "__main__":
    main()
