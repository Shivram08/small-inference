Source: `results/kv_quant/20261019-142247_Qwen2.5-0.5B-Instruct/results.json`

| Format | Bytes/token | Tokens in budget | Perplexity | Mean KL vs bf16 (nats) | Top-1 agree | K rel. error | V rel. error |
|---|---|---|---|---|---|---|---|
| bf16 | 12,288 | 174,752 | 8.633 | 0.0000 | 100.0% | 0.0000 | 0.0000 |
| int8 | 6,336 | 338,928 | 8.984 | 0.0511 | 91.5% | 0.0090 | 0.0077 |
| int8-kchannel | 6,624 | 322,288 | 8.649 | 0.0092 | 96.5% | 0.0032 | 0.0077 |

K outlier ratio (max channel absmax / median, per layer): mean 5.4, max 23.7 (layer 8), min 2.1
