import sys, time, statistics
from pathlib import Path
from bench.engines import load_engine
from bench.runner import load_trace
eng, _ = load_engine("contiguous")
prompt = load_trace(Path("traces/sharegpt_n100_seed0_burst.jsonl"))[0].prompt_ids
def canary():
    acc = 0
    for i in range(4000):
        acc = (acc + i * 7) % 1_000_003
    return acc
rows = []  # (unix_s, step_ms, canary_us)
end = time.time() + 90
while time.time() < end:
    eng.add_request(0, prompt, 1500)
    while eng.has_unfinished() and time.time() < end:
        t0 = time.perf_counter(); eng.step(); t1 = time.perf_counter(); canary(); t2 = time.perf_counter()
        rows.append((time.time(), (t1 - t0) * 1e3, (t2 - t1) * 1e6))
    while eng.has_unfinished(): eng.step()
secs = {}
for ts, st, c in rows: secs.setdefault(int(ts), []).append((st, c))
per = [(statistics.median(x[0] for x in v), statistics.median(x[1] for x in v)) for k, v in sorted(secs.items()) if len(v) > 5]
fast = [p for p in per if p[0] < 22]; slow = [p for p in per if p[0] >= 22]
for name, g in (("fast", fast), ("slow", slow)):
    if g: print(f"{name}: {len(g):3d} s, step {statistics.median(p[0] for p in g):5.1f} ms, canary {statistics.median(p[1] for p in g):6.1f} us")
print("".join("S" if p[0] >= 22 else "f" for p in per))
