# Implementation Plan

A living document that we update as each milestone lands. Long or heavy runs that are written but not yet run are queued in `docs/deferred-runs.md`. Status: `[ ]` todo, `[~]` in progress, `[x]` done (its gating test passes).

Every milestone runs through the same loop: concept discussion with real numbers → interface design → implementation → gating test passes → measure with the harness → design-log entry → only then compare against nano-vllm or vLLM.

---

## Step 0: Environment

- [x] `git init -b main`, `.gitignore` (weights, venvs, profiler output; `traces/` and `results/` tracked), `.gitattributes` (LF everywhere)
- [x] git identity (Shivram08) + first commit (8ec1175)
- [x] GitHub: private repo https://github.com/Shivram08/small-inference, `main` tracks `origin/main`. Pushing from WSL fails on large uploads (`curl 55 … GnuTLS recv error`, likely the WSL2 virtual-network MTU), so push from Windows git, using the Windows `gh` login:
  `git -c safe.directory=* -c "credential.https://github.com.helper=" -c "credential.https://github.com.helper=!gh auth git-credential" -C \\wsl.localhost\Ubuntu-24.04\home\nekka\Small_Inference push`
- [x] Repo moved to WSL ext4 at `~/Small_Inference` (checksums verified, CRLF converted to LF). The stale `E:\Small_Inference` original (CLAUDE.md and PLAN.MD identical to 8ec1175, an older implementation-plan.md) was deleted on 2026-10-19
- [x] Ubuntu 24.04 installed in WSL2 and set as the default distro
- [x] GPU access inside WSL: `nvidia-smi` and `nvidia-smi -q -d PERFORMANCE` work. Vivobook Pro 15 N6506MJ (Optimus: display on the Intel Arc iGPU, dGPU at 0 MiB used, so all 6 GB is ours). The NVML failure earlier was transient (dGPU idle-powered-off), not a setting. Power limit 75 W when plugged in with Best performance + fan Performance (60 W default + about 15 W Dynamic Boost; 30 W in Balanced mode). Every "Clocks Event Reasons" field is readable in WSL
- [x] Driver: updated on Windows from 566.07 / CUDA 12.7 to 617.42 / CUDA 13.4 (`nvidia-smi` in WSL reports userspace 615.78.05, KMD 617.42)
- [x] RAM: 16 GB total, WSL gets 7.4 GiB + 2 GiB swap (no `.wslconfig`). Every run now records `/proc/meminfo` before and after (`env.json`) and the process's peak RSS (`run.json`). The Phase 1 loader streams tensors one at a time to the GPU, so the 1.5B model (3.1 GB) never sits fully in CPU RAM. Heavy runs (campaigns) only when the laptop is otherwise idle: Phase 0 campaigns crashed it while it was in use
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
- [x] **M0.6 Roofline** (`bench/roofline.py`, `tests/test_roofline.py`). 3 runs at f9366ad: `results/roofline/20261011-162332`, `-162407`, `-162439`. Plugged in, Best performance, enforced power limit 75–82 W, no throttle-counter increase during any run, GPU 64–67 °C after. Runs 2 and 3 have a false `git_dirty: true` (untracked `results/` was counted; fixed in env.py afterwards)
  - Bandwidth: 512 MB copy. Expected about 170–200 GB/s. **Measured 156.2 GB/s** (3-run spread 0.07; within-run 153.6–156.6). The theoretical number for this SKU is 96-bit × 14 Gbps = 168 GB/s, not 190–220, so this is 93% of theoretical.
  - bf16 matmul peak: expected about 10–13 TFLOP/s. **Measured 18.66 TFLOP/s** (spread 0.18, best at n = 6144; 15.0 at n = 1024). That matches 20 SMs × 512 FLOP/clk (fp32 accumulate) × ~1.8 GHz. **Ridge = 119.5 FLOP/B**, not 55–65.
  - Decode sweep K = 896, N = 4864: I ≈ 1 at M = 1 and about 59 at M = 64. It crosses the ridge near **M ≈ 128–192**, not 64.
    - M = 1–2: 131–133 GB/s (84–85% of BW), a GEMV-like kernel.
    - **M = 4–16: falls to 73–89 GB/s.** M = 4 takes 120 µs vs 66 µs at M = 2 for the same bytes; cuBLAS switches kernels. Batch 4–16 decode GEMMs are worse per step than batch 2.
    - M = 32–128: 111–114 GB/s, about 72% of BW.
    - M = 192: a step from 92 to 139 µs (tile/wave quantization). M = 256–512: 14.3–15.8 TFLOP/s, 77–85% of peak (K = 896 is a short reduction).
  - Launch overhead: per-launch µs × profiled kernel count per decode step. Estimated about 850 launches and 5 ms. **Measured 11.3 µs per eager launch** (`x.add_`, spread 0.9) × **1006 kernels per HF decode step** (prompt 128, batch 1; about 42 per layer) = **11.4 ms**.
  - HF decode step wall time: **25.4–30.7 ms** (run medians), versus the 6.3 ms bandwidth bound (988 MB of weights / 156 GB/s → **158 tok/s**). HF batch 1 runs at about 25% of the bound and is CPU-bound: launch overhead alone is about 1.8× the GPU time, and the rest is Python module overhead. S estimate for M0.7: 230 tokens × ~26 ms ≈ 6 s
- [x] **M0.7 HF baseline** (`bench/engines/hf.py`, `bench/baseline.py`). Gate: `tests/test_hf_engine.py` (4 pass, token-identical to greedy `generate()`). Campaign `results/20261011-163739_hf_baseline/manifest.json`, 12 runs at 3fe4be1, all clean, 16:37–20:46
  - Manual prefill/decode loop, `.item()` before taking the timestamp.
  - Burst trace → S → 0.3×, 0.7×, 1.2× capacity, each run 3 times, each with `env.json`.
  - Rep-1 burst: capacity 0.1469 req/s (S = 6.81 s) → rates 0.0441 / 0.103 / 0.176. Over 3 reps, burst capacity was 0.139–0.169.
  - Results (mean over 3 reps): 0.3× TTFT p50 0.20 s, p99 7.9 s, goodput 0.82; 0.7× TTFT p50 1.75 s, goodput 0.51; 1.2× TTFT p50 31 s, goodput 0.06, overload flagged in all 3 reps. TPOT p50 24–28 ms everywhere (bound: 6.3 ms).
  - Low-load reps are repeatable (achieved rate within 0.1%, TPOT p50 within 1–2 ms). 1.2× drifted slower over the evening: TPOT p50 25.0 → 28.2 → 31.2 ms. CPU-bound timing; no CPU telemetry yet.
  - The generation_config defaults (sampling, repetition_penalty 1.1) don't apply to the raw argmax. In the reference test, `generate(eos_token_id=None)` still stops at the model's EOS, so the test points EOS at an untrained padded-vocab id.
- [x] Design log: measured BW, peak, launch overhead, S, chosen rates
- [~] Measurement follow-ups before the next campaign (2026-10-19). Each one is a measurement change, so Phase 0 results are not comparable with runs after it. Code and unit tests done (`tests/test_telemetry.py`, `tests/test_profile_decode.py`); the campaign that uses them is pending:
  - [x] Host CPU clocks in `env.json` (`cpu_before`/`cpu_after`, from Windows `Get-Counter` through interop, no admin). **CPU temperature dropped** (decided 2026-10-19): it needs a Windows-side monitoring tool. The counters give per logical CPU a nominal frequency (1400 MHz P-cores = logical 0, 1, 10–19; 900 E-cores = 2–9; 700 LP E-cores = 20, 21), % performance (effective clock = nominal × perf / 100), and % utility
  - [x] `bench/telemetry.py`: nvidia-smi sampled every 1 s and CPU counters every 2 s during each run (`runner.run(telemetry=True)`, always on in campaigns) → `telemetry_gpu.csv`, `telemetry_cpu.csv`, `telemetry.json`. Throttle counters summed over increases only, counting resets. Found while testing: at idle, the SW power-cap and SW thermal counters run at 1 µs/µs (active mask 0x24), so the summary also gives each reason's active fraction over busy samples (idle bit clear). Caveat: 1 s polling may keep the dGPU from idling down, which could shrink the Phase 0 cold-start prefill penalty
  - [x] Campaign order shuffled per repetition (`--order-seed`, default 0, orders saved in the manifest). Cooldown waits for the GPU to reach its idle temperature at campaign start + 3 °C (min 30 s, max 600 s), recorded per run in the manifest
  - [~] `bench/profile_decode.py` (`--engine hf|contiguous`): torch.profiler on batch-1 decode steps → wall, GPU busy (kernel-interval union), kernel and launch counts, launch API time, sync wait, remainder. Smoke-tested only, in **Balanced mode** (30 W GPU limit): step 47 ms unprofiled; 1008 kernels and 1006 launches per step; GPU busy 11.2 ms. Not a valid measurement (`results/profile/20261019-124857_hf`). Rerun in Best performance mode
  - [~] `bench/cpu_probe.py`: pure-Python fixed workload + CPU telemetry, no GPU, to test the P/E-core hypothesis. One 90 s run in Balanced mode with the laptop in use (`results/cpu_probe/20261019-124946`): chunk time is block-bimodal too (≈1.05 ms vs ≈1.8 ms, slow blocks up to 10 s), but the busiest core was a P-core in every sample, and slow windows had the busiest core at 2.2 GHz vs 3.4 GHz. Preliminary: points at P-core clock drops, not E-core migration. Rerun in Best performance mode with the laptop idle
- [ ] Next HF campaign with the new measurement code: merged into the HF + Phase 1 comparison campaign (`bench/baseline.py --engines hf,contiguous --rates ...`, which reuses the Phase 0 rate traces). Queued in `docs/deferred-runs.md`

## Phase 1: Forward pass and contiguous KV cache

- [x] safetensors loader + config parsing (`engine/config.py`, `engine/loader.py`). `tie_word_embeddings: true`, so lm_head is the embedding tensor (no copy). Strict: missing or unexpected names raise. Streams one tensor at a time to the GPU and casts there, so CPU RAM peaks at one tensor (272 MB, the embedding). Gate: tensor fingerprints equal HF's
- [x] RMSNorm (computed in fp32, cast back before the weight multiply). Gate: bit-identical to `Qwen2RMSNorm` in fp32 and bf16
- [x] RoPE with the `rotate_half` pairing. Gate: bit-identical to HF at positions up to 32,767 (fp32, bf16), plus relative-position and pairing property tests. Sign of rotate_half folded into the sin table; cos/sin gathered once per forward; q and k rotated together as one [T, 16, 64] view
- [x] One decoder layer matching HF layer 0. Gate: bit-identical with `fused_matmuls = False`; 1.26e-5 fused (output magnitude 3)
- [x] Full model (`engine/model.py`). **Gate (invariant 1), fused 2.4e-4 accepted (2026-10-19):** with `fused_matmuls = False`, fp32 logits are **bit-identical** to HF eager on all three prompts (36, 247, 1 tokens). With fusion (the default), max |diff| 2.4e-4 at logit magnitude 30 (8e-6 relative), entirely from cuBLAS reduction order in the fused QKV / gate-up matmuls (confirmed: unfused → 0.0); argmax identical everywhere. Test bound: 5e-4 absolute and 2e-5 relative. bf16 argmax agreement with HF fp32 over 817 positions: 95.5%, same as HF bf16 eager (95.6%); HF bf16 SDPA gets 97.4%
- [x] Contiguous cache `[H_kv, T_max, d]` per layer (`engine/kv_cache.py`), keys after RoPE. The forward pass takes `positions` explicitly; with a cache they come from `cache.next_positions(n)` (device-side arange, no sync, no copy). Chunked prefill (64-token chunks) matches one-shot within 1.1e-4
- [x] Generate (`engine/generate.py`). **Gate (invariant 2):** 50 greedy fp32 tokens identical cached vs uncached for the chat, 247-token, and 1-token prompts, and identical to HF's own cached greedy decode
- [x] Engine (`engine/engine.py`, `ContiguousEngine`): batch-1 FIFO like the HF baseline, one preallocated 2,048-token cache reused per request. Gate: `tests/test_engine.py` (protocol, FIFO, equals `greedy_generate`). Runner: `--engine contiguous`
- [~] Measure batch-1 TPOT in the harness against HF. Done so far: launches per decode step 1381 → **905** (HF: 1008) after removing a hidden K/V copy (broadcast `matmul` expanded K and V 7×) and per-layer RoPE table work. Pending: items 2, 3, and 5 in `docs/deferred-runs.md`

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
- 2026-10-11: M0.2–M0.7 done: runner, metrics, harness self-test, ShareGPT traces, roofline (3 runs), HF baseline campaign (12 runs) and its analysis. Phase 0 gates all pass. GitHub remote added (push from Windows).
- 2026-10-19: Measurement follow-ups built (telemetry, shuffled order, temperature cooldown, decode profiler, CPU probe); the campaign that uses them waits for an idle laptop. Stale `E:\Small_Inference` deleted. Run policy: only short (< 5 min), non-saturating tests while the laptop is in use. Phase 1 started.
