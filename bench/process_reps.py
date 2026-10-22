"""Campaign repetitions as separate processes: one engine per process, engines shuffled within each repetition.

For configurations whose engines can't share one GPU (Qwen2.5-1.5B: 3.1 GB of weights plus a 1.5 GiB pool per
format leaves room for one pool at a time on 6 GB). Each child is `bench.baseline --engines <one> --reps 1` with the
given extra arguments (which must include --rates, so every engine replays the same traces). The repetition order is
shuffled per repetition with a recorded seed, as in a single-process campaign, so slow drift spreads across engines.
Writes results/<stamp>_process_reps/manifest.json listing every child campaign manifest, for bench.rate_plots.

Usage: python -m bench.process_reps --engines continuous,continuous-int8,continuous-int8k --reps 3 -- \\
           --model Qwen/Qwen2.5-1.5B-Instruct --kv-gib 1.5 --max-seqs 256 --n 1000 --rates 1.45,2.89,4.34,5.79
"""

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    """Run every (repetition, engine) child in shuffled order and record their manifests."""
    if "--" not in sys.argv:
        sys.exit("pass bench.baseline arguments after --")
    split = sys.argv.index("--")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engines", required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--order-seed", type=int, default=0)
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--python", action="append", default=[], metavar="ENGINE=PATH",
                   help="interpreter for one engine (e.g. continuous=.venv/bin/python while vLLM runs from its venv)")
    p.add_argument("--child-timeout-s", type=float, default=None,
                   help="kill a child (and its whole process group, e.g. vLLM's engine core) after this long")
    p.add_argument("--resume", type=Path, default=None,
                   help="a previous process_reps folder: reuse its children that exited 0, rerun the rest in order")
    args = p.parse_args(sys.argv[1:split])
    pythons = dict(x.split("=", 1) for x in args.python)
    child_args = sys.argv[split + 1:]
    if "--rates" not in child_args:
        sys.exit("child arguments must include --rates (all engines must replay the same traces)")

    out = args.results_root / f"{datetime.now():%Y%m%d-%H%M%S}_process_reps"
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.order_seed)
    manifest = {"engines": args.engines.split(","), "reps": args.reps, "order_seed": args.order_seed,
                "child_args": child_args, "pythons": pythons, "child_timeout_s": args.child_timeout_s,
                "resumed_from": str(args.resume) if args.resume else None, "children": []}
    reuse = {}
    if args.resume:
        for c in json.loads((args.resume / "manifest.json").read_text())["children"]:
            if c["returncode"] == 0 and c["manifest"]:
                reuse[(c["rep"], c["engine"])] = c
    for rep in range(1, args.reps + 1):
        order = list(manifest["engines"])
        rng.shuffle(order)
        for engine in order:
            if (rep, engine) in reuse:
                manifest["children"].append(reuse[(rep, engine)] | {"reused": True})
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
                print(f"rep {rep} {engine}: reused -> {reuse[(rep, engine)]['manifest']}", flush=True)
                continue
            cmd = [pythons.get(engine, sys.executable), "-m", "bench.baseline", "--engines", engine, "--reps", "1",
                   "--order-seed", str(rep), "--results-root", str(args.results_root), *child_args]
            # Own process group, so a timeout also kills the child's own subprocesses (vLLM's engine core).
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    start_new_session=True)
            try:
                text, _ = proc.communicate(timeout=args.child_timeout_s)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                text, _ = proc.communicate()
                code = "timeout"
            done = re.search(r"^done: (.+manifest\.json)$", text, re.M)
            manifest["children"].append({"rep": rep, "engine": engine, "returncode": code,
                                         "manifest": done.group(1) if done else None})
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            (out / f"rep{rep}_{engine}.log").write_text(text)
            print(f"rep {rep} {engine}: exit {code} -> {done.group(1) if done else 'no manifest'}", flush=True)
    print(f"done: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
