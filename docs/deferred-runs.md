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
| 5 | **Comparison campaign**: HF and Phase 1 engine, burst + 3 rates, 3 shuffled reps, telemetry on | `.venv/bin/python -m bench.baseline --engines hf,contiguous --rates 0.0441,0.103,0.176` | **~8 h** (overnight) | 1 core, GPU at full power, ~2.5 GB GPU memory | Phase 1 "batch-1 TPOT vs HF"; a new HF baseline measured with the same code (Phase 0 numbers aren't comparable after the measurement change) |

Item 5's time: each rate trace lasts about n/λ of arrivals (100 / 0.0441 req/s ≈ 38 min at 0.3×), so one repetition of both engines is about 2.5 h plus cooldowns. To shorten it, run `--reps 1` first as a check (~2.7 h), then the full campaign.

After item 5: rerun `bench/plots.py` on the new campaign, and compare TPOT, TTFT, and goodput per rate between the two engines.

## Done

| # | What | When | Result |
|---|---|---|---|
