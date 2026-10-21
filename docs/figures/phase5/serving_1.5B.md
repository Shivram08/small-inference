Medians over 3 repetitions, recomputed from raw logs.

| Engine | Burst tok/s | Burst preemptions | TPOT p50/p99 at 1.45 req/s (ms) | TPOT p50/p99 at 2.89 req/s (ms) | TPOT p50/p99 at 4.34 req/s (ms) | TPOT p50/p99 at 5.79 req/s (ms) |
|---|---|---|---|---|---|---|
| bf16 KV | 1,289 | 416 | 30.8 / 36.5 | 32.0 / 41.3 | 40.8 / 50.3 | 58.3 / 71.7 |
| INT8 per-token KV | 1,574 | 0 | 31.2 / 38.8 | 32.2 / 42.8 | 38.2 / 52.1 | 47.3 / 61.8 |
| INT8 per-channel K | 1,456 | 0 | 32.7 / 43.2 | 40.0 / 51.5 | 42.5 / 59.8 | 53.1 / 70.6 |
