Medians over 3 repetitions, recomputed from raw logs.

| Engine | Burst tok/s | Burst preemptions | TPOT p50/p99 at 1.73 req/s (ms) | TPOT p50/p99 at 3.46 req/s (ms) | TPOT p50/p99 at 5.19 req/s (ms) | TPOT p50/p99 at 6.92 req/s (ms) | TPOT p50/p99 at 8.65 req/s (ms) | TPOT p50/p99 at 10.4 req/s (ms) |
|---|---|---|---|---|---|---|---|---|
| bf16 KV | 1,969 | 0 | 27.8 / 36.9 | 29.2 / 39.5 | 31.2 / 43.8 | 34.0 / 48.6 | 30.7 / 39.0 | 39.7 / 44.7 |
| bf16 KV, unchunked prefill | 1,630 | 0 | 31.4 / 37.2 | 35.9 / 40.8 | 27.2 / 33.1 | 32.6 / 45.9 | 42.4 / 49.3 | 36.2 / 44.2 |
