"""Rebuild every figure and table from raw logs, in one command (the Phase 6 reproducibility gate).

Reads docs/figures/sources.json, which maps each figure group to the result folders it is computed from, and runs
the matching generator: bench.plots (Phase 0 analysis), bench.decode_sweep.plot, bench.rate_plots, and
bench.kv_quant_eval.table_md. Nothing is read from a cached summary: the generators recompute from requests.jsonl /
steps.jsonl / results.json. Groups with no sources yet are listed as pending.

Usage: python -m bench.make_figures [--only phase0,phase5_table]
"""

import argparse
import json
from pathlib import Path

SOURCES = Path("docs/figures/sources.json")


def build(name: str, spec: dict) -> list[Path] | None:
    """Regenerate one group; None if it has no sources yet."""
    out = Path(spec["out"])
    kind = spec["kind"]
    if kind == "phase0":
        from bench.plots import phase0
        phase0(Path(spec["manifest"]), out)
        return sorted(out.glob("*.png")) + [out / "analysis.json"]
    if kind == "decode_sweep":
        if not spec.get("results"):
            return None
        from bench.decode_sweep import plot
        return plot(Path(spec["results"]), out)
    if kind == "rates":
        if not spec.get("manifests"):
            return None
        from bench.rate_plots import collect, plot
        return [plot(collect([Path(m) for m in spec["manifests"]]), out)]
    if kind == "serving_table":
        if not spec.get("manifests"):
            return None
        from bench.rate_plots import serving_table_md
        out.mkdir(parents=True, exist_ok=True)
        path = out / spec.get("filename", "serving.md")
        path.write_text(serving_table_md([Path(m) for m in spec["manifests"]]))
        return [path]
    if kind == "kv_quant_table":
        if not spec.get("results"):
            return None
        from bench.kv_quant_eval import table_md
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for r in spec["results"]:
            res = json.loads(Path(r).read_text())
            path = out / f"table_{res['model'].split('/')[-1]}.md"
            path.write_text(f"Source: `{r}`\n\n" + table_md(res))
            paths.append(path)
        return paths
    raise ValueError(f"{name}: unknown kind {kind}")


def main() -> None:
    """Build all (or --only) groups and report what was written and what is pending."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", default=None, help="comma-separated group names")
    args = p.parse_args()
    groups = {k: v for k, v in json.loads(SOURCES.read_text()).items() if not k.startswith("_")}
    if args.only:
        groups = {k: groups[k] for k in args.only.split(",")}
    for name, spec in groups.items():
        written = build(name, spec)
        if written is None:
            print(f"{name}: pending ({spec.get('pending', 'no sources')})")
        else:
            print(f"{name}: " + ", ".join(map(str, written)) + (f"  [still pending: {spec['pending']}]"
                                                                 if spec.get("pending") else ""))


if __name__ == "__main__":
    main()
