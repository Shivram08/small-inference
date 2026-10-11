# Design Log

Two or three lines per phase: what was chosen, what was tried and failed, and why. This becomes the README's design section.

## Phase 0: Measurement and roofline

- Roofline (3 runs): 156 GB/s copy bandwidth (93% of the 168 GB/s theoretical), 18.7 TFLOP/s bf16 peak, ridge about 120 FLOP/B. Both bandwidth and peak differ from the pre-measurement estimates (190–220 GB/s, 10–13 TFLOP/s), which assumed the wrong SKU.
- cuBLAS dip: for the decode MLP shape [M, 896] × [896, 4864], M = 4–16 runs at 73–89 GB/s versus 131–133 at M = 1–2 (a kernel switch: M = 4 takes 120 µs, M = 2 takes 66 µs). Batch-size throughput curves must be read with this in mind.

## Phase 1: Forward pass and contiguous KV cache

## Phase 2: Paged KV cache and block manager

## Phase 3: Triton paged-attention decode

## Phase 4: Continuous batching scheduler

## Phase 5: INT8 KV cache

## Phase 6: Benchmark vs. vLLM and write-up
