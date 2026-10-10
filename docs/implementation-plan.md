# Implementation Plan

A living document that we update as each milestone lands. Status: `[ ]` todo, `[~]` in progress, `[x]` done (its gating test passes).

Every milestone runs through the same loop: concept discussion with real numbers → interface design → implementation → gating test passes → measure with the harness → design-log entry → only then compare against nano-vllm or vLLM.

---

## Step 0: Environment

- [x] `git init -b main`, `.gitignore` (weights, venvs, profiler output; `traces/` and `results/` tracked), `.gitattributes` (LF everywhere)
- [ ] git identity + first commit
- [x] Repo moved to WSL ext4 at `~/Small_Inference` (checksums verified, CRLF converted to LF). `E:\Small_Inference` is the stale original, to delete once we've confirmed the move
- [x] Ubuntu 24.04 installed in WSL2 and set as the default distro
- [x] GPU access inside WSL: `nvidia-smi` and `nvidia-smi -q -d PERFORMANCE` work. Vivobook Pro 15 N6506MJ (Optimus: display on the Intel Arc iGPU, dGPU at 0 MiB used, so all 6 GB is ours). The NVML failure earlier was transient (dGPU idle-powered-off), not a setting. Power cap 60 W. Every "Clocks Event Reasons" field is readable in WSL
- [ ] Driver: update from 566.07 / CUDA 12.7 (Oct 2024) to the current NVIDIA driver, on Windows only, never inside WSL
- [ ] RAM: 16 GB total, Windows at 96% used, WSL gets about 7.7 GB. Watch this for ShareGPT parsing and model loading
- [ ] Python environment inside WSL with pinned torch, triton, transformers, safetensors, numpy, pytest, matplotlib
- [ ] Sanity checks: `torch.cuda.is_available()`, bf16 matmul, Triton vector-add
- [ ] HF cache on the WSL filesystem, with Qwen2.5-0.5B-Instruct downloaded
- [ ] Repo skeleton from the CLAUDE.md layout, plus an empty `docs/design-log.md`
- vLLM gets its own separate venv in Phase 6, because it pins its own torch version

## Phase 0: Measurement and roofline

- [ ] **M0.1 Engine protocol and fake engine** (`bench/engines/base.py`, `fake.py`)
  - One `step()` = one forward pass. For a batch-1 engine, a prefill step returns 1 token.
  - Fake engine with parameters a (s/token of prefill) and b (s/decode step), batch-1 FIFO.
- [ ] **M0.2 Runner** (`bench/runner.py`)
  - Open-loop replay. After each step, admit every request with `arrival_s ≤ now` and record its *scheduled* arrival time.
  - Idle wait: sleep until about 1 ms before the next arrival, then spin. Measure how late the sleep wakes up on WSL.
  - `time.perf_counter()` everywhere. Log queue depth every step.
- [ ] **M0.3 Metrics** (`bench/metrics.py`)
  - Write raw `requests.jsonl` first. Summaries come from the raw records.
  - To decide: drop the warmup 10% by arrival order or by completion order? How to define "queue still growing" for the overload flag?
- [ ] **M0.4 Harness self-test** (`tests/test_harness.py`)
  - TPOT ≈ b, TTFT ≈ a·P, mean inter-arrival time ≈ 1/λ.
  - Overload flag set at ρ = 1.2, not set at ρ = 0.4. Tolerances based on measured sleep jitter.
- [ ] **M0.5 Traces** (`bench/trace.py`)
  - ShareGPT V3, Qwen tokenizer, filters (prompt ≤ 1024, output 4–512, total ≤ 2048). One fixed sample → several rates plus a burst trace, with a sidecar file.
  - Open question: HF has S ≈ 4–6 s, so 0.3× capacity is about 0.06 req/s, and 200 requests would take about 55 min. Choose n per rate and possibly cap `output_len`. Record the choice in the design log.
- [ ] **M0.6 Roofline** (`bench/roofline.py`)
  - Bandwidth: 512 MB copy. Expect about 170–200 GB/s.
  - bf16 matmul peak: expect about 10–13 TFLOP/s (GeForce Ampere runs fp32 accumulation at half rate), so the ridge is about 55–65 FLOP/B.
  - Decode sweep K = 896, N = 4864: I ≈ 1 at M = 1 and about 59 at M = 64, so it should cross the ridge near M ≈ 64.
  - Launch overhead: per-launch µs × profiled kernel count per decode step (estimate: about 850 launches, about 5 ms).
- [ ] **M0.7 HF baseline** (`bench/engines/hf.py`)
  - Manual prefill/decode loop, `.item()` before taking the timestamp.
  - Burst trace → S → 0.3×, 0.7×, 1.2× capacity, each run 3 times, each with `env.json`.
- [ ] Design log: measured BW, peak, launch overhead, S, chosen rates

## Phase 1: Forward pass and contiguous KV cache

- [ ] safetensors loader + config parsing (check `tie_word_embeddings`)
- [ ] RMSNorm (computed in fp32)
- [ ] RoPE with the `rotate_half` pairing, tested on its own against HF
- [ ] One decoder layer matching HF layer 0 (Q/K/V biases, no o_proj bias, GQA g = 7)
- [ ] Full model. **Gate:** invariant 1 (fp32 within ~1e-4)
- [ ] Contiguous cache `[H_kv, T_max, d]` per layer, keys stored after RoPE. The forward pass takes `positions` explicitly.
- [ ] Generate. **Gate:** invariant 2 (50 tokens, cached == uncached)
- [ ] Measure batch-1 TPOT in the harness against HF

## Phase 2: Paged KV cache and block manager

- [ ] `BlockManager` (`can_allocate`, `allocate`, `append_slot`, `free`, `num_free_blocks`, `ref_count`). **Gate:** invariant 3
- [ ] Builders for block tables, context lengths and slot mapping
- [ ] Gather + SDPA paged attention. **Gate:** invariant 4 (lengths 1, 15, 16, 17, 33)
- [ ] Measure the gather cost against contiguous attention

## Phase 3: Triton paged-attention decode

- [ ] Triton tutorials (vector add, fused softmax, matmul, fused attention)
- [ ] Online softmax in NumPy (the (2, 5, 1) example)
- [ ] Single-pass kernel: one program per (seq, KV head), g padded to 16. **Gate:** invariant 5
- [ ] Split-K + combine kernel. **Gate:** invariant 5, and faster at batch 1 with long context
- [ ] Sweeps: bandwidth vs. context (128–8k) × batch (1, 8, 32) × block size (8, 16, 32), with measured BW as the ceiling

## Phase 4: Continuous batching scheduler

- [ ] Request states (waiting, running, preempted, finished)
- [ ] Flat packed batch with `cu_seqlens`
- [ ] Admission with a watermark. **Gate:** invariant 6
- [ ] Recompute preemption. **Gate:** invariant 7
- [ ] Chunked prefill with a token budget. Compare p99 ITL with and without chunking
- [ ] Throughput and p99 TPOT vs. rate plots
- [ ] (Stretch) CUDA graphs, measured separately

## Phase 5: INT8 KV cache

- [ ] Quantize/dequantize reference implementation and error measurement
- [ ] INT8 per-token K/V kernel (scale folded into scores and P)
- [ ] Per-channel K (quantized per block, bf16 partial block) + per-token V
- [ ] Chunked-prefill perplexity, capacity, and serving runs on 1.5B
- [ ] **Gate:** the three-config table

## Phase 6: Benchmark vs. vLLM and write-up

- [ ] Separate vLLM venv, memory-matched configs (eager and CUDA graphs)
- [ ] One script that rebuilds every plot from raw logs
- [ ] README with design, results, and limitations. Go through Heiser's checklist first

---

## Early decisions

- Stay on 0.5B through Phase 4. Switch to 1.5B for the memory-pressure experiments.
- Results folders are `results/<YYYYMMDD-HHMMSS>_<engine>_<trace>/`, created by the runner and never overwritten.
- The forward pass takes explicit `positions` from Phase 1 on.

## Changelog

- 2026-10-10: Plan created.
- 2026-10-10: Ubuntu 24.04 installed, GPU visible in WSL, repo moved to `~/Small_Inference`, git initialized. Decided to update the driver.
