| Format | Bytes/token | Tokens in budget | Perplexity | Mean KL vs bf16 (nats) | Top-1 agree | K rel. error | V rel. error |
|---|---|---|---|---|---|---|---|
| bf16 | 12,288 | 174,752 | 8.633 | 0.0000 | 100.0% | 0.0000 | 0.0000 |
| int8 | 6,336 | 338,928 | 8.984 | 0.0511 | 91.5% | 0.0090 | 0.0077 |
| int8-kchannel | 6,624 | 322,288 | 8.649 | 0.0092 | 96.5% | 0.0032 | 0.0077 |

K outlier ratio (max channel absmax / median, per layer): mean 5.4, max 23.7 (layer 8), min 2.1

Decode kernel, median of 3 repetition(s); batch x context points below ~4k at small batch are launch-bound.

| Batch | Context | bf16 µs/layer | int8 µs/layer | int8-kchannel µs/layer |
|---|---|---|---|---|
| 1 | 1024 | 58.2 | 67.6 | 73.5 |
| 1 | 4096 | 97.3 | 104.7 | 103.7 |
| 8 | 1024 | 56.5 | 66.1 | 62.0 |
| 8 | 4096 | 111.2 | 134.9 | 145.0 |
| 32 | 1024 | 111.2 | 70.5 | 97.5 |
| 32 | 4096 | 420.0 | 244.0 | 356.2 |
