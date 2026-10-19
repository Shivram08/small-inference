import time, statistics, torch
from pathlib import Path
from bench.engines import load_engine
from bench.runner import load_trace
from engine.kv_cache import ContiguousKVCache
eng, _ = load_engine("contiguous")
m = eng.model
prompt = load_trace(Path("traces/sharegpt_n100_seed0_burst.jsonl"))[0].prompt_ids
rows = []
end = time.time() + 90
with torch.inference_mode():
    while time.time() < end:
        cache = ContiguousKVCache(m.cfg, 2048, "cuda", m.dtype)
        ids = torch.tensor(prompt, device="cuda")
        tok = m.forward(ids, cache.next_positions(len(prompt)), cache, last_only=True)[0].argmax()[None]
        for _ in range(1500):
            if time.time() >= end or cache.length + 1 > 2048: break
            t0 = time.perf_counter()
            logits = m.forward(tok, cache.next_positions(1), cache, last_only=True)
            t1 = time.perf_counter()
            tok = logits[0].argmax()[None]; int(tok)
            t2 = time.perf_counter()
            rows.append((time.time(), (t1 - t0) * 1e3, (t2 - t1) * 1e3))
secs = {}
for ts, a, b in rows: secs.setdefault(int(ts), []).append((a, b))
per = [(statistics.median(x[0] + x[1] for x in v), statistics.median(x[0] for x in v), statistics.median(x[1] for x in v)) for _, v in sorted(secs.items()) if len(v) > 5]
for name, g in (("fast", [p for p in per if p[0] < 22]), ("slow", [p for p in per if p[0] >= 22])):
    if g: print(f"{name}: {len(g):3d} s | step {statistics.median(p[0] for p in g):5.1f} ms = enqueue {statistics.median(p[1] for p in g):5.1f} + wait {statistics.median(p[2] for p in g):5.1f} ms")
print("".join("S" if p[0] >= 22 else "f" for p in per))
