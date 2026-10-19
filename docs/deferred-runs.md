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

| 6 | **Phase 4 capacity + rate sweep**, continuous engine: burst → capacity → rate traces at 0.2–1.2× → runs | `.venv/bin/python -m bench.baseline --engines continuous --fractions 0.2,0.4,0.6,0.8,1.0,1.2 --reps 1` | ~20–30 min | 1 core, GPU, ~3 GB GPU memory; building the new rate traces parses the 640 MB ShareGPT JSON (several GB of RAM for a minute) | Phase 4 capacity; the rate traces for item 7 |
| 7 | **Chunked vs unchunked prefill**, 3 shuffled reps on item 6's traces | `.venv/bin/python -m bench.baseline --engines continuous,continuous-nochunk --rates <item 6's rates>` | ~1–1.5 h | as item 6 (one shared model, two 1.5 GiB pools) | Phase 4 "p99 ITL with and without chunking" and "throughput and p99 TPOT vs rate" |
| 8 | Phase 4 figures from items 6–7 (and item 5 for HF / Phase 1 on their traces) | `.venv/bin/python -m bench.rate_plots results/<item 7 campaign>/manifest.json --out docs/figures/phase4` | seconds | 1 core | Phase 4 plots |

| 9 | Download Qwen2.5-1.5B-Instruct (3.1 GB, network; disk has 946 GB free) | `.venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct` | a few min | network | Phase 5 on 1.5B (items 10–11) |
| 10 | Phase 5 three-config table: quality (64-token chunked prefill: perplexity, KL, top-1, quantization error, K outliers), capacity, and decode-kernel timing per format | `.venv/bin/python -m bench.kv_quant_eval --timing` then `--model Qwen/Qwen2.5-1.5B-Instruct --timing` | ~2 + ~5 min | 1 core, GPU; 1.5B: ~4.5 GB GPU memory | Phase 5 gate (the table); 0.5B quality already measured, timing pending |
| 11 | Phase 5 serving on 1.5B under memory pressure (1.5 GiB KV: 56k tokens bf16 vs ~110k INT8): bf16 capacity + rate traces first, then each INT8 format on the same traces, one process per format | `.venv/bin/python -m bench.baseline --model Qwen/Qwen2.5-1.5B-Instruct --engines continuous --fractions 0.3,0.6,0.9,1.2 --reps 1`, then for E in continuous-int8 continuous-int8k: `... --engines $E --rates <those rates> --reps 3` (and `continuous` again with `--rates`, 3 reps) | ~2–4 h | 1 core, GPU at full power, ~5 GB GPU memory | Phase 5 serving results: throughput and latency per format at matched memory |
| 12 | Phase 5 figures | `.venv/bin/python -m bench.rate_plots <item 11 manifests> --out docs/figures/phase5` | seconds | 1 core | Phase 5 plots |

Item 5's time: each rate trace lasts about n/λ of arrivals (100 / 0.0441 req/s ≈ 38 min at 0.3×), so one repetition of both engines is about 2.5 h plus cooldowns. To shorten it, run `--reps 1` first as a check (~2.7 h), then the full campaign.

After item 5: rerun `bench/plots.py` on the new campaign, and compare TPOT, TTFT, and goodput per rate between the two engines.

## Done

| # | What | When | Result |
|---|---|---|---|
