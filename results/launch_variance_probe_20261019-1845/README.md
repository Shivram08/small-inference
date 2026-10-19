Locked SM clock (1,927 MHz, set by the user), quiet machine, Phase 1 contiguous engine, batch-1 decode, 90 s each:

- `decode_telemetry.py` -> `locked_clock_telemetry/`: still bimodal, per-second median step p10 18.5 / p50 20.6 / p90 34.4 ms with the SM at 1,927 MHz in all 210 samples. GPU power is lower in slow seconds (34-36 W vs 43 W): a consequence of less work per second, not a cause.
- `decode_cpu_canary.py`: a fixed ~0.17 ms pure-Python chunk between steps: 164 us in fast seconds, 170 us in slow (step 21.1 vs 25.9 ms). CPU execution speed is unchanged.
- `decode_split.py`: step = CPU enqueue (forward returns) + GPU wait (.item()). Fast 19.0 = 17.3 + 1.7 ms; slow 22.6 = 20.9 + 1.7 ms. The extra time is entirely in enqueueing ~900 launches through WSL's GPU paravirtualization path.
- The slow share varies run to run: 36%, 89%, 4% of seconds in three consecutive runs.
