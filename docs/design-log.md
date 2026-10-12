# Design Log

Two or three lines per phase: what was chosen, what was tried and failed, and why. This becomes the README's design section.

## Phase 0: Measurement and roofline

- Roofline (3 runs): 156 GB/s copy bandwidth (93% of the 168 GB/s theoretical), 18.7 TFLOP/s bf16 peak, ridge about 120 FLOP/B. Both bandwidth and peak differ from the pre-measurement estimates (190–220 GB/s, 10–13 TFLOP/s), which assumed the wrong SKU.
- cuBLAS dip: for the decode MLP shape [M, 896] × [896, 4864], M = 4–16 runs at 73–89 GB/s versus 131–133 at M = 1–2 (a kernel switch: M = 4 takes 120 µs, M = 2 takes 66 µs). Batch-size throughput curves must be read with this in mind.
- Harness: open-loop replay with scheduled arrival times; runner-side queue depth by scheduled arrival, so the Engine protocol stays unchanged. Rates use kept-request spans (offered = arrival span, achieved = finish span), which is unbiased when the engine keeps up and gives 1/S on a burst trace. Validated against a fake engine with known timing: TPOT within 1.7% of b, overload off at ρ = 0.4 and on at ρ = 1.2.
- Traces: ShareGPT V3, human → gpt first turns only (the gpt-first conversations are split continuations), Qwen chat template (+29 tokens), n = 100, seed 0, no output cap. Mean output 230 tokens.
- HF baseline (bf16, SDPA, batch 1): S = 6.8 s, capacity about 0.15 req/s (0.139–0.169 over 3 burst reps). Rates 0.0441 / 0.103 / 0.176 req/s. TPOT 24–28 ms vs the 6.3 ms bandwidth bound: CPU-bound, with 1006 kernels × 11.3 µs of launch overhead alone. Even at 0.3× load, 18% of requests miss TTFT < 2 s from FIFO head-of-line blocking behind long requests, as M/G/1 predicts. Limitation: busy runs drift by up to 25% (CPU-side, not yet instrumented).

## Phase 1: Forward pass and contiguous KV cache

## Phase 2: Paged KV cache and block manager

## Phase 3: Triton paged-attention decode

## Phase 4: Continuous batching scheduler

## Phase 5: INT8 KV cache

## Phase 6: Benchmark vs. vLLM and write-up
