| Format | Bytes/token | Tokens in budget | Perplexity | Mean KL vs bf16 (nats) | Top-1 agree | K rel. error | V rel. error |
|---|---|---|---|---|---|---|---|
| bf16 | 28,672 | 74,896 | 6.953 | 0.0000 | 100.0% | 0.0000 | 0.0000 |
| int8 | 14,560 | 147,488 | 7.565 | 0.1177 | 89.2% | 0.0119 | 0.0089 |
| int8-kchannel | 15,344 | 138,032 | 6.992 | 0.0389 | 94.1% | 0.0034 | 0.0089 |

K outlier ratio (max channel absmax / median, per layer): mean 7.2, max 68.1 (layer 0), min 2.7

Decode kernel, median of 3 repetition(s); batch x context points below ~4k at small batch are launch-bound.

| Batch | Context | bf16 µs/layer | int8 µs/layer | int8-kchannel µs/layer |
|---|---|---|---|---|
| 1 | 1024 | 54.0 | 55.0 | 65.5 |
| 1 | 4096 | 103.6 | 89.0 | 122.0 |
| 8 | 1024 | 63.9 | 64.1 | 70.6 |
| 8 | 4096 | 214.4 | 232.9 | 238.0 |
| 32 | 1024 | 218.5 | 156.3 | 241.9 |
| 32 | 4096 | 834.8 | 576.5 | 929.0 |
