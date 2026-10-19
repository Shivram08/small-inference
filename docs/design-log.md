# Design Log

Two or three lines per phase: what was chosen, what was tried and failed, and why. This becomes the README's design section.

## Phase 0: Measurement and roofline

- Roofline (3 runs): 156 GB/s copy bandwidth (93% of the 168 GB/s theoretical), 18.7 TFLOP/s bf16 peak, ridge about 120 FLOP/B. Both bandwidth and peak differ from the pre-measurement estimates (190–220 GB/s, 10–13 TFLOP/s), which assumed the wrong SKU.
- cuBLAS dip: for the decode MLP shape [M, 896] × [896, 4864], M = 4–16 runs at 73–89 GB/s versus 131–133 at M = 1–2 (a kernel switch: M = 4 takes 120 µs, M = 2 takes 66 µs). Batch-size throughput curves must be read with this in mind.
- Harness: open-loop replay with scheduled arrival times; runner-side queue depth by scheduled arrival, so the Engine protocol stays unchanged. Rates use kept-request spans (offered = arrival span, achieved = finish span), which is unbiased when the engine keeps up and gives 1/S on a burst trace. Validated against a fake engine with known timing: TPOT within 1.7% of b, overload off at ρ = 0.4 and on at ρ = 1.2.
- Traces: ShareGPT V3, human → gpt first turns only (the gpt-first conversations are split continuations), Qwen chat template (+29 tokens), n = 100, seed 0, no output cap. Mean output 230 tokens.
- HF baseline (bf16, SDPA, batch 1): S = 6.8 s, capacity about 0.15 req/s (0.139–0.169 over 3 burst reps). Rates 0.0441 / 0.103 / 0.176 req/s. TPOT 24–28 ms vs the 6.3 ms bandwidth bound: CPU-bound, with 1006 kernels × 11.3 µs of launch overhead alone. Even at 0.3× load, 18% of requests miss TTFT < 2 s from FIFO head-of-line blocking behind long requests, as M/G/1 predicts. Limitation: busy runs drift by up to 25% (CPU-side, not yet instrumented).
- Phase 0 analysis (`bench/plots.py` → `docs/figures/phase0/`):
  - **TPOT is bimodal**, about 23–25 ms or about 32–34 ms, switching in blocks (sometimes mid-run). That rules out gradual thermal drift. Hypothesis, untested: the decode thread is being moved between performance and efficiency cores on the hybrid laptop CPU (ratio ≈ 1.37).
  - **Cold-start prefill:** the same request's prefill is a median 153 ms slower at 0.3× than in burst, because the dGPU idles down between requests. The penalty falls with load (62% / 27% / 4% of requests over 50 ms at 0.3× / 0.7× / 1.2×), tracking the fraction of arrivals that find the server idle. It shows as a step at about 0.2 s in the 0.3× TTFT distribution.
  - **Warm prefill ≈ 25 ms + 63.5 µs/prompt token.** The intercept is about one CPU-bound decode step. The slope is close to the compute bound of 2 × 494M params / 18.7 TFLOP/s = 53 µs/token, so long prefills run at about 83% of matmul peak.
  - **M/G/1 (Pollaczek–Khinchine)** predicts the mean queue wait to the right order (ρ 0.25: 1.3–1.6 s predicted vs 0.85–1.14 measured; ρ 0.6: 4.9–6.2 vs 3.3–5.1). Measured is 65–82% of predicted, plausibly from finite runs that start with an empty queue, and the high variance of a 90-request mean. Head-of-line blocking explains the TTFT tail at low load.
- Measurement follow-ups (2026-10-19, a measurement change): per-run GPU and host-CPU telemetry, shuffled run order, cooldown to a temperature target. The idle GPU accrues SW power-cap and SW thermal "throttle" time at 1 µs/µs, so Phase 0's before/after counter checks could not have shown throttling either way; the busy-sample reason mask replaces them. A first CPU-only probe (Balanced mode, laptop in use, not yet repeated) is block-bimodal without the GPU, with slow blocks on a P-core at 2.2 GHz vs 3.4 GHz, which points at P-core clock drops rather than E-core migration.

## Phase 1: Forward pass and contiguous KV cache

- Plain tensors and functions, flat [T, ...] tokens with explicit positions, fused QKV ([896, 1152]) and gate/up ([896, 9728]) matmuls. Unfused, the fp32 forward pass is bit-identical to HF eager, which pins every op exactly; fused, logits differ by 2.4e-4 (8e-6 relative), all from cuBLAS reduction order, so the "~1e-4" gate is met in relative terms and checked bit-exactly via the unfused switch. bf16 argmax agreement with fp32 matches HF bf16 eager (95.5% vs 95.6%); SDPA's 97.4% comes from keeping attention probabilities in fp32, a target for the Phase 3 kernel.
- Tried and fixed: GQA via a broadcast `torch.matmul` against K as [2, 1, 64, S] silently expanded and copied K and V 7× per layer; stacking the group into M (q as [2, 7T, 64], `bmm`) reads each KV head once. With RoPE tables gathered once per forward and a 4-kernel rotation, a decode step went from 1381 to 905 launches (HF: 1008). Softmax upcasts to at least fp32 (a hard fp32 cast broke fp64 unit tests).

## Phase 2: Paged KV cache and block manager

- 16-token blocks (192 KiB across 24 layers for 0.5B bf16), pool `[layers, H_kv, N, 16, 64]` so one head-block is 2 KiB contiguous and a sequence's gather reshapes to `[H_kv, S, 64]` for free. Pure-Python block manager with a LIFO free list and reference counts (always 0/1 until prefix caching). Both caches expose the same `begin / write / attend` interface, so the model has no paged-specific code; metadata already packs several sequences into one flat forward, for Phase 4.
- The reference paged attention gathers into contiguous K/V and reuses the Phase 1 `attention()` instead of SDPA, which makes paged == contiguous checkable bitwise (it is, in bf16 and fp32, through shuffled tables and block-boundary lengths). The cost is a full copy of each sequence's K/V per layer per step, which the Phase 3 kernel removes.

## Phase 3: Triton paged-attention decode

- One program per (sequence, KV head) handles the whole GQA group as a [16, 64] query tile, so each KV head's blocks are read once (intensity about g = 7). Tiles of 64 tokens look up their block per token, which decouples the tile from the block size (8, 16, and 32 all work under `tl.dot`'s 16 minimum). P stays fp32 through P·V, the fix for the 2-point bf16 argmax gap Phase 1 traced to rounding P. Split-K (Flash-Decoding) partials plus a combine kernel fill the 20 SMs at batch 1, where the single pass has only 2 programs.
- The decode path is one launch per layer (two with split-K) instead of the gather path's per-sequence gather + matmul + softmax chain, and the prefill rows of a mixed batch keep the gather reference. Tests were mutation-checked: three deliberate kernel bugs each fail 26–44 of the 45 invariant-5 cases.

## Phase 4: Continuous batching scheduler

- One state per request (`known` tokens, `num_computed`) makes prefill, decode, chunked prefill, and recompute the same operation, so preemption is just "free the blocks, set num_computed to 0, requeue at the front"; the recompute covers prompt + generated tokens and then sampling continues. Decodes are scheduled first, then partial prefills, then FIFO admissions, under a token budget (Sarathi-Serve).
- Admission reserves the whole remaining prefill's blocks. The first version reserved chunk by chunk, which can deadlock two half-prefilled requests on the same free blocks; reserving up front means only decodes grow sequences, which is where preemption lives. Batch invariance holds on greedy tokens in fp32 (not on logit bits: batching changes every matmul's M, and the decode kernel's split count depends on the batch).

## Phase 5: INT8 KV cache

## Phase 6: Benchmark vs. vLLM and write-up
