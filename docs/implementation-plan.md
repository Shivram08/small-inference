# Implementation Plan

A living document that we update as each milestone lands. Status: `[ ]` todo, `[~]` in progress, `[x]` done (its gating test passes).

Every milestone runs through the same loop: concept discussion with real numbers → interface design → implementation → gating test passes → measure with the harness → design-log entry → only then compare against nano-vllm or vLLM.

---

## Step 0: Environment

- [x] `git init -b main`, `.gitignore` (weights, venvs, profiler output; `traces/` and `results/` tracked), `.gitattributes` (LF everywhere)
- [x] git identity (Shivram08) + first commit (8ec1175). GitHub auth exists on Windows only; set up gh in WSL before the first push
- [x] Repo moved to WSL ext4 at `~/Small_Inference` (checksums verified, CRLF converted to LF). `E:\Small_Inference` is the stale original, to delete once we've confirmed the move
- [x] Ubuntu 24.04 installed in WSL2 and set as the default distro
- [x] GPU access inside WSL: `nvidia-smi` and `nvidia-smi -q -d PERFORMANCE` work. Vivobook Pro 15 N6506MJ (Optimus: display on the Intel Arc iGPU, dGPU at 0 MiB used, so all 6 GB is ours). The NVML failure earlier was transient (dGPU idle-powered-off), not a setting. Power limit 75 W when plugged in with Best performance + fan Performance (60 W default + about 15 W Dynamic Boost; 30 W in Balanced mode). Every "Clocks Event Reasons" field is readable in WSL
- [x] Driver: updated on Windows from 566.07 / CUDA 12.7 to 617.42 / CUDA 13.4 (`nvidia-smi` in WSL reports userspace 615.78.05, KMD 617.42)
- [ ] RAM: 16 GB total, Windows at 96% used, WSL gets about 7.7 GB. Watch this for ShareGPT parsing and model loading
- [x] Python environment inside WSL with pinned torch, triton, transformers, safetensors, numpy, pytest, matplotlib. Managed by uv 0.13.0 (`~/.local/bin/uv`; no sudo, and system Python lacks `ensurepip`). `.venv/` on system Python 3.12.3. Direct deps pinned with `==` in `pyproject.toml`, transitive deps in `uv.lock`. Pins: torch 2.14.1+cu130 (from the PyTorch cu130 index; arch list includes sm_86), triton 3.8.0, transformers 5.19.0, safetensors 0.8.0, huggingface-hub 2.2.0, numpy 2.5.3, matplotlib 3.11.2, pytest 9.1.1. Recreate with `uv sync`
- [x] Sanity checks in `tests/test_env.py`: `torch.cuda.is_available()`, bf16 matmul, Triton vector-add (2²⁴, 2²⁴ + 17, 1). cc 8.6, **20 SMs** (not 16), 5.03 / 6.00 GiB free with a CUDA context open, bf16 4096³ relative error 2.88e-3. Triton needed `build-essential python3.12-dev` (gcc 13.3) to build its launcher stub
- [x] HF cache at the default `~/.cache/huggingface` (ext4), with Qwen2.5-0.5B-Instruct downloaded (954 MB, snapshot 7ae5576). `tie_word_embeddings: true`, so there is no separate lm_head weight
- [x] Repo skeleton from the CLAUDE.md layout (`engine/`, `bench/engines/` packages, `traces/` and `results/` tracked via `.gitkeep`), plus an empty `docs/design-log.md`
- vLLM gets its own separate venv in Phase 6, because it pins its own torch version

## Phase 0: Measurement and roofline

- [x] **M0.1 Engine protocol and fake engine** (`bench/engines/base.py`, `fake.py`). Gate: `tests/test_fake_engine.py` (8 pass)
  - One `step()` = one forward pass. For a batch-1 engine, a prefill step returns 1 token.
  - Fake engine with parameters a (s/token of prefill) and b (s/decode step), batch-1 FIFO.
  - Waits by sleeping until 1 ms before the deadline, then spinning. Measured on WSL across 5 runs of 40 × 5 ms steps: median lateness 3 µs, worst 0.35 ms (preemption). Prefill 30 ms → 30.01 ms
- [x] **M0.2 Runner** (`bench/runner.py`, `bench/env.py`). Mechanics gate: `tests/test_runner.py` (9 pass). The statistical gate is M0.4
  - Open-loop replay. After each step, admit every request with `arrival_s ≤ now` and record its *scheduled* arrival time.
  - Idle wait: sleep until about 1 ms before the next arrival, then spin. Measured on WSL: idle admission lag 16 µs mean, 23 µs max.
  - `time.perf_counter()` everywhere. Log queue depth every step.
  - Queue depth is counted by scheduled arrival, from the runner side (the protocol stays unchanged): `outstanding` = arrived − finished, `waiting` = arrived − has a first token.
  - 2 warmup requests (negative ids, output capped at 8) run before t0 and aren't recorded. Every run drains fully. Contract violations raise.
  - Output: `requests.jsonl` (with `output_ids`, for comparing engines token by token later), `steps.jsonl`, `env.json` (static env + `nvidia-smi` before/after, including the cumulative throttle counters), `run.json`. Folders are never overwritten.
  - Runner overhead between steps (end of step k to start of step k+1): median 2.3–2.5 µs over thousands of steps in M0.4. The 17–39 µs seen in an early demo came from its first few steps
- [x] **M0.3 Metrics** (`bench/metrics.py`). Formula gate: `tests/test_metrics.py` (8 pass, exact answers on simulated FIFO timelines). The real-clock gate is M0.4
  - Write raw `requests.jsonl` first. Summaries come from the raw records and go to `summary.json`, which can be regenerated.
  - Decided: drop the first 10% **by arrival** (under overload, completion order favours short requests).
  - Decided: rates come from the kept requests' spans. Offered = (k−1)/arrival span, achieved = (k−1)/finish span. This is unbiased when the engine keeps up (finishes ≈ arrivals + S), and on a burst trace it gives capacity 1/S directly.
  - Decided: overload = achieved < 0.95 × offered, OR the least-squares slope of `waiting` over the second half of the kept arrival window > 0.05 × offered. Null for burst traces. Both tolerances are recorded in `summary.json`.
  - Output throughput = all tokens with timestamps inside the kept finish window ÷ window length. Goodput uses strict `<` against the SLO (TTFT 2 s, TPOT 100 ms).
- [x] **M0.4 Harness self-test** (`tests/test_harness.py`, run with `pytest -m slow`, about 33 s). Gate passes, stable over 3 runs
  - TPOT ≈ b, TTFT ≈ a·P, mean inter-arrival time ≈ 1/λ.
  - Overload flag set at ρ = 1.2, not set at ρ = 0.4. Tolerances based on measured sleep jitter.
  - Fake timings scaled 10× down (a = 10 µs/token, b = 1 ms, P = 300, n = 21 → S = 23 ms, capacity 43.5 req/s), so runs take seconds.
  - Unloaded (50 ms gaps): TTFT 3.039–3.047 ms (a·P = 3.0). TPOT 1.0123–1.0165 ms (b = 1.0); the excess is runner overhead + spin overshoot + preemption (TPOT p99 1.10–1.13 ms). Tolerance overhead + 50 µs.
  - Burst (150 requests): 42.98–43.22 req/s vs 1/S = 43.48 (−0.6 to −1.2%, from about 3 µs per-step overshoot × 21 steps).
  - ρ = 0.4 (250 Poisson requests, seed 42): achieved = offered = 17.76, queue slope ≈ 0.005 req/s against a 0.89 threshold, so no flag. ρ = 1.2 (150 requests): offered 58.5 (realized Poisson; nominal 52.2), achieved 41.7–43.2 ≈ capacity, slope 10–12 req/s, both conditions fire.
  - `bench/trace.py` started with `poisson_arrivals` and `write_trace`. Mean gap within 4 SE of 1/λ at n = 20,000.
  - The default `pytest` skips slow tests (`addopts = -m 'not slow'`). Rerun `-m slow` after any change to the runner, metrics, or fake engine
- [x] **M0.5 Traces** (`bench/trace.py`). Gate: `tests/test_trace.py` (10 pass, synthetic ShareGPT + real tokenizer)
  - ShareGPT V3, Qwen tokenizer, filters (prompt ≤ 1024, output 4–512, total ≤ 2048). One fixed sample → several rates plus a burst trace, with a sidecar file.
  - Only conversations whose first two turns are human → gpt (58,787 of 94,145). The 33,826 that start with gpt are later chunks of split conversations, with no first prompt.
  - Prompt = human turn in the Qwen chat template (default system prompt + generation prompt, +29 tokens). `output_len` = raw token count of the gpt reply.
  - Arrivals at every rate come from the same arrival seed (sample seed + 1), so traces at different rates are exact rescalings of each other.
  - `json.load` of the 673 MB file: 1.51 GiB peak, 4.7 s. A full build: 2.3 GB peak, 17 s. No streaming parser needed.
  - Decided: **n = 100 per trace, seed 0, no output cap.** Estimated HF S ≈ 5 s → about 54 min per repetition of burst + 3 rates, about 2.7 h for 3 repetitions. The rate traces are built in M0.7, once S is measured.
  - Built `traces/sharegpt_n100_seed0_burst`: prompt mean 124 / p50 52 / max 927, output mean 230 / p50 242 / p90 417 / max 494 tokens
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
- 2026-10-10: Python env created with uv (torch 2.14.1+cu130). Picked cu130 over cu128 because the driver supports CUDA 13.4 and cu128 wheels stop at torch 2.11.
- 2026-10-10: Step 0 complete: sanity checks green, Qwen2.5-0.5B downloaded, skeleton in place. Found 20 SMs (CLAUDE.md corrected). M0.1 fake engine done.
