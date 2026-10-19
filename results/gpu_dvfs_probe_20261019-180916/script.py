import json, sys, time, statistics
from pathlib import Path
import torch
from bench.engines import load_engine
from bench.runner import load_trace
from bench.telemetry import Telemetry, parse_cpu_csv, parse_gpu_csv, summarize_cpu_sample
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
eng, _ = load_engine("contiguous")
prompt = load_trace(Path("traces/sharegpt_n100_seed0_burst.jsonl"))[0].prompt_ids
steps = []
with Telemetry(out, gpu_interval_s=0.5, cpu_interval_s=1):
    end = time.time() + 90
    while time.time() < end:
        eng.add_request(0, prompt, 1500)
        while eng.has_unfinished() and time.time() < end:
            t = time.perf_counter(); eng.step(); steps.append((time.time(), (time.perf_counter() - t) * 1e3))
        while eng.has_unfinished(): eng.step()
cpu = parse_cpu_csv((out / "telemetry_cpu.csv").read_text())
gpu = parse_gpu_csv((out / "telemetry_gpu.csv").read_text())
rows = []
for ms in sorted(cpu):
    t = ms / 1e3
    win = [d for ts, d in steps if t - 1 <= ts < t]
    if len(win) < 5: continue
    s = summarize_cpu_sample(cpu[ms])
    g = min(gpu, key=lambda r: abs(r["unix_s"] - t))
    rows.append((statistics.median(win), s["busiest"]["effective_mhz"], s["effective_mhz_by_class"]["P"],
                 float(g["power.draw"]), float(g["clocks.sm"]), s["busiest"]["class"]))
json.dump(rows, open(out / "rows.json", "w"))
fast = [r for r in rows if r[0] < 22]; slow = [r for r in rows if r[0] >= 22]
for name, grp in (("fast (<22 ms)", fast), ("slow (>=22 ms)", slow)):
    if grp:
        print(f"{name:15s} n={len(grp):3d} step {statistics.median(r[0] for r in grp):5.1f} ms | busiest-core "
              f"{statistics.median(r[1] for r in grp):5.0f} MHz | mean P-class {statistics.median(r[2] for r in grp):5.0f} MHz"
              f" | GPU {statistics.median(r[3] for r in grp):4.1f} W, SM {statistics.median(r[4] for r in grp):5.0f} MHz"
              f" | busiest class {dict(__import__('collections').Counter(r[5] for r in grp))}")
print("".join("S" if r[0] >= 22 else "f" for r in rows))
