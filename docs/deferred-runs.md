# Deferred runs

Tests and measurements that are written but not yet run, because they are long or load the machine. They run only when the laptop is otherwise idle. Everything else (the default `pytest` suite: < 1 min, at most a few GB of GPU memory, one CPU core) runs while the laptop is in use.

When an item has run: record the result folder and the outcome in `docs/implementation-plan.md`, tick the item there, and move the row to "Done" below.

## Before any of these

- [ ] Plugged in. Windows power mode **Best performance**, fan mode **Performance**. (In Balanced, the GPU is capped at 30 W and an HF decode step took 47 ms instead of 24–31 ms.)
- [ ] `nvidia-smi --query-gpu=enforced.power.limit --format=csv` shows about 75 W (not 30 W)
- [ ] Heavy apps closed: browser tabs with video, games, IDE indexing
- [ ] `free -m` shows at least 5 GB available in WSL

## Queue (run in this order)

| # | What | Command | Time | Load | Unlocks |
|---|---|---|---|---|---|
| 1 | Harness self-test, after the runner change of 2026-10-19 (memory snapshots, telemetry hook) | `.venv/bin/python -m pytest -m slow` | ~35 s | 1 core | M0.4 still valid with the new runner |
| 2 | Decode-step profile, HF | `.venv/bin/python -m bench.profile_decode --engine hf` | < 1 min | 1 core, GPU | Breakdown of the 24–31 ms HF step (GPU vs launches vs Python); measurement follow-up item |
| 3 | Decode-step profile, Phase 1 engine | `.venv/bin/python -m bench.profile_decode --engine contiguous` | < 1 min | 1 core, GPU | Same breakdown for 905 launches/step |
| 4 | CPU probe, laptop idle | `.venv/bin/python -m bench.cpu_probe --duration 120` | 2 min | 1 core | Tests the P/E-core vs P-core-clock explanation of bimodal TPOT |
| 4b | Paged-gather cost vs contiguous attention (Phase 2) | `.venv/bin/python -m bench.paged_gather` | < 1 min | 1 core, GPU | Phase 2 "measure the gather cost"; the baseline the Phase 3 kernel must beat |
| 4c | Triton decode-kernel sweep (Phase 3): context 128–8k × batch 1/8/32 × block 8/16/32, single pass and split-K, with figures | `.venv/bin/python -m bench.decode_sweep` | ~1–2 min | 1 core, GPU | Phase 3 gates "bandwidth-vs-context plot" and "split-K faster at batch 1, long context" |
| 5 | **Comparison campaign**: HF and Phase 1 engine, burst + 3 rates, 3 shuffled reps, telemetry on | `.venv/bin/python -m bench.baseline --engines hf,contiguous --rates 0.0441,0.103,0.176` | **~8 h** (overnight) | 1 core, GPU at full power, ~2.5 GB GPU memory | Phase 1 "batch-1 TPOT vs HF"; a new HF baseline measured with the same code (Phase 0 numbers aren't comparable after the measurement change) |

| 6 | **Phase 4 capacity + rate sweep**, continuous engine: burst → capacity → rate traces at 0.2–1.2× → runs (n = 1000: builds the n = 1000 burst and rate traces) | `.venv/bin/python -m bench.baseline --engines continuous --n 1000 --fractions 0.2,0.4,0.6,0.8,1.0,1.2 --reps 1` | ~45–60 min | 1 core, GPU, ~3 GB GPU memory; building the new rate traces parses the 640 MB ShareGPT JSON (several GB of RAM for a minute) | Phase 4 capacity; the rate traces for item 7 |
| 7 | **Chunked vs unchunked prefill**, 3 shuffled reps on item 6's traces | `.venv/bin/python -m bench.baseline --engines continuous,continuous-nochunk --n 1000 --rates <item 6's rates>` | ~4–5 h | as item 6 (one shared model, two 1.5 GiB pools) | Phase 4 "p99 ITL with and without chunking" and "throughput and p99 TPOT vs rate" |
| 8 | Phase 4 figures from items 6–7 (and item 5 for HF / Phase 1 on their traces) | `.venv/bin/python -m bench.rate_plots results/<item 7 campaign>/manifest.json --out docs/figures/phase4` | seconds | 1 core | Phase 4 plots |

| 9 | Download Qwen2.5-1.5B-Instruct (3.1 GB, network; disk has 946 GB free) | `.venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct` | a few min | network | Phase 5 on 1.5B (items 10–11) |
| 10 | Phase 5 three-config table: quality (64-token chunked prefill: perplexity, KL, top-1, quantization error, K outliers), capacity, and decode-kernel timing per format | `.venv/bin/python -m bench.kv_quant_eval --timing` then `--model Qwen/Qwen2.5-1.5B-Instruct --timing` | ~2 + ~5 min | 1 core, GPU; 1.5B: ~4.5 GB GPU memory | Phase 5 gate (the table); 0.5B quality already measured, timing pending |
| 11 | Phase 5 serving on 1.5B under memory pressure (1.5 GiB KV: 56k tokens bf16 vs ~110k INT8): bf16 capacity + rate traces first, then each INT8 format on the same traces, one process per format | 11a: `.venv/bin/python -m bench.baseline --model Qwen/Qwen2.5-1.5B-Instruct --engines continuous --kv-gib 1.5 --max-seqs 256 --n 1000 --fractions 0.3,0.6,0.9,1.2 --reps 1`; 11b: `.venv/bin/python -m bench.process_reps --engines continuous,continuous-int8,continuous-int8k --reps 3 -- --model Qwen/Qwen2.5-1.5B-Instruct --kv-gib 1.5 --max-seqs 256 --n 1000 --rates 1.45,2.89,4.34,5.79` | ~2–4 h | 1 core, GPU at full power, ~5 GB GPU memory | Phase 5 serving results: throughput and latency per format at matched memory |
| 12 | Phase 5 figures | `.venv/bin/python -m bench.rate_plots <item 11 manifests> --out docs/figures/phase5` | seconds | 1 core | Phase 5 plots |

| 13 | Create the vLLM venv (network, several GB; vLLM pins its own torch). Record the installed version in the plan | `uv venv .venv-vllm --python 3.12 && VIRTUAL_ENV=.venv-vllm uv pip install vllm` | 5–15 min | network, disk | Phase 6 |
| 14 | vLLM adapter test (Engine contract, exact lengths) | `PYTHONPATH=. .venv-vllm/bin/python -m pytest tests/test_vllm_engine.py -p no:cacheprovider -o addopts=""` | ~1 min | 1 core, GPU | Phase 6 harness |
| 15 | vLLM comparison on item 6's traces: first pick `VLLM_GPU_MEMORY_UTILIZATION` so vLLM's reported KV cache is close to 1.5 GiB (our pool), then run CUDA graphs and eager; our `continuous` runs come from item 7 | `PYTHONPATH=. VLLM_GPU_MEMORY_UTILIZATION=<u> .venv-vllm/bin/python -m bench.baseline --engines vllm,vllm-eager --rates <item 6's rates>` | ~1–1.5 h | 1 core, GPU at full power | Phase 6 comparison ("X% of vLLM eager throughput at Y req/s") |
| 16 | Rebuild every figure and table after adding the new result folders to `docs/figures/sources.json`; then write the README | `.venv/bin/python -m bench.make_figures` | ~1 min | 1 core | Phase 6 gates |

Item 5's time: each rate trace lasts about n/λ of arrivals (100 / 0.0441 req/s ≈ 38 min at 0.3×), so one repetition of both engines is about 2.5 h plus cooldowns. To shorten it, run `--reps 1` first as a check (~2.7 h), then the full campaign.

After item 5: rerun `bench/plots.py` on the new campaign, and compare TPOT, TTFT, and goodput per rate between the two engines.

## Done

| # | What | When | Result |
|---|---|---|---|
| 1 | Harness self-test | 2026-10-19 | 4 passed (35.7 s): M0.4 still valid with the new runner |
| 2–3 | Decode-step profiles, 3 interleaved runs each, plus the continuous engine (`results/profile/20261019-18*`) | 2026-10-19 | Unprofiled step p50: HF 26.2 / 25.6 / 24.7 ms (1006 launches); contiguous 32.0 / 29.8 / 18.4 ms (903); continuous 17.7 / 27.3 / 18.9 ms (784). Bimodal: see the GPU DVFS row |
| 4 | CPU probe, idle laptop (`results/cpu_probe/20261019-180631`) | 2026-10-19 | Unimodal: all 480 windows 0.8–1.1 ms, busiest core a P-core at ~4.2 GHz. The earlier bimodal probe was Balanced mode + laptop in use |
| — | Locked-clock probes (`results/launch_variance_probe_20261019-1845`) | 2026-10-19 | With the SM locked at 1,927 MHz, decode is still bimodal; the extra time is CPU-side launch enqueue through WSL (GPU wait and Python speed unchanged). Campaigns proceed; reported as a limitation |
| 4b | Paged gather vs contiguous attention, 3 runs (`results/paged_gather/20261019-1853*`; the wall-time-only runs `-185052/-185112/-185117` used the old method) | 2026-10-19 | GPU kernel time: gather costs 1.5–2.0× contiguous attention (stable to ~1% across runs). Per 24-layer step: +0.15 ms (batch 1, 128 tokens) to +0.89 ms (batch 1, 4k), +7.3 ms (batch 8, 4k); 2 extra launches per sequence per layer. Wall time is launch-bound and swung 2× between runs, so the method was changed to report kernel time |
| 4c | Decode-kernel sweep, 3 runs (`results/decode_sweep/20261019-1853*`, `-185423`, `-185431`) + batch 2/4 | 2026-10-19 | Peak **154 GB/s = 99% of measured copy BW** (batch 32, 8k). Split-K at batch 1: 3.2× faster at 8k, 1.6× at 4k, but 1.3–1.9× *slower* at ≤ 2k, never helpful at batch 8. `choose_splits` changed to split only at ≤ 8 programs and ≥ 3,072 tokens |
| 10 | Phase 5 table on 0.5B and 1.5B with timing (`results/kv_quant/20261019-185822_*`, `-185928_*`) | 2026-10-19 | Quality: 1.5B perplexity bf16 6.953, int8 7.565 (+8.8%), int8-kchannel 6.992 (+0.56%); K outliers up to 68× median (layer 0). **Timing is a negative result**: INT8 per-token only ties bf16 (bandwidth halved, so the kernel is no longer memory-bound), and per-channel K is 3.2× slower on 0.5B and catastrophic on 1.5B (d = 128: 0.68 s per layer at 32 × 4k, likely register spills). Needs a kernel fix before item 11 |
| 6 (n = 100, exploratory) | Continuous capacity + 6 rates, 1 rep, old kernel (`results/20261019-190320_continuous_campaign`) | 2026-10-19 | Burst capacity 4.87 req/s (HF: 0.147, 33×); TTFT p50 0.04–0.07 s at every rate. All rates were flagged overloaded, a measurement artifact: on 17–100 s traces the achieved rate reads up to 30% low from generation-time spread. Fixed in metrics (allowance for generation-time spread) and by moving the batching engines to n = 1000 traces; item 6 is redone at n = 1000 |
| 6 | Continuous capacity + 6 rates at n = 1000, fixed kernel (`results/20261019-192347_continuous_campaign`) | 2026-10-19 | **Capacity 8.65 req/s** (59× HF batch 1; the n = 100 burst read 4.87 because of its drain tail). Rates 1.73 / 3.46 / 5.19 / 6.92 / 8.65 / 10.4 req/s: TTFT p50 0.06–0.08 s up to 1.0×; overload flagged only at 1.2× (TTFT p50 8.9 s) |
| 7 | Chunked vs unchunked prefill, 6 rates × 3 shuffled reps, n = 1000 (`results/20261019-195616_continuous_continuous-nochunk_campaign`, figure `docs/figures/phase4/rate_curves.png`) | 2026-10-19 | At 1.0× (8.65 req/s) chunked keeps up (TTFT p50 0.09 s, 1,746 tok/s) and unchunked is overloaded (TTFT p50 14 s, 1,477 tok/s). p99 ITL at 0.8–1.0×: chunked 63–82 ms, unchunked 80–99 ms (+~40%). Same at 0.2×. Under 1.2× overload chunked has the worse TTFT (16.4 vs 9.6 s): decodes first means admissions wait. Throughput saturates near 1,750 tok/s |
| 11a | 1.5B bf16 capacity + rate traces, 1.5 GiB pool, up to 256 sequences, n = 1000 (`results/20261019-231137_continuous_campaign`) | 2026-10-19 | Burst capacity 4.82 req/s; rates 1.45 / 2.89 / 4.34 / 5.79: TTFT p50 0.07–0.13 s at all four, TPOT p50 36–74 ms, not overloaded even at 1.2× |
| 11b (first run, reference INT8 writes) | 1.5B, 3 formats × 3 reps, one process each (`results/20261019-234449_process_reps`) | 2026-10-20 | Burst (pool binds): INT8 1,316–1,348 tok/s vs bf16 1,187–1,201 (+11–13%), 0 preemptions vs 416. At offered rates the pool never binds (bf16 peak 20–55%), and INT8 TPOT was ~1.9× bf16 (64 vs 34 ms at 1.45 req/s): the torch quantize ops added several hundred launches per step. Superseded by the rerun with fused writes; kept as the record of the unfused path |
| 11b rerun (partial) | Fused INT8 writes, 3 formats, order seed 1 (`results/20261020-044302_process_reps`) | 2026-10-20 | Repetition 1 complete for all three formats (`-044306_continuous-int8`, `-051459_continuous-int8k`, `-054700_continuous`). Repetition 2 was interrupted at 06:46 by a Windows Update restart during its first process (`continuous-int8k`, partial; not to be used). Remaining: repetitions 2–3 (6 processes, ~3.3 h), after re-locking clocks |
| 9 | 1.5B download | 2026-10-19 | Qwen2.5-1.5B-Instruct, 2.9 GB, snapshot 989aa79 |
| — | GPU DVFS probe: 90 s of contiguous decode with 0.5 s GPU / 1 s CPU telemetry (`results/gpu_dvfs_probe_20261019-180916`) | 2026-10-19 | **The bimodal TPOT is GPU clocks, not the CPU.** Slow mode: driver reports "GPU Idle", SM ~1,200 MHz, 17 W, step ~32 ms. Fast: SM ~1,940 MHz, 43 W, "Reliability" (0x400), step ~19.6 ms. CPU clocks were slightly *higher* in the slow mode. Launch-bound decode leaves gaps the governor reads as idle |
