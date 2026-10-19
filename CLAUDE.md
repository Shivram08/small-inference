# Inference Engine Build

A minimal LLM serving engine built from scratch, as both a learning project and a portfolio piece. It includes a hand-written Qwen2 forward pass, a paged KV cache, Triton paged-attention decode kernels, a continuous-batching scheduler, and one original contribution: an INT8 KV cache with fused dequantization. The final deliverable is an honest benchmark against vLLM on the same hardware and workload.

The point is understanding, not just a working engine. Every component should be built, tested against a reference, and measured before moving on.

## How to work with me on this repo

- **Explain with concrete numbers first**, then the formal version. Use real shapes and sizes from the models below, not abstract `N` and `M`.
- **Go deep.** Assume I know tokens, logits, attention, and FlashAttention basics. Don't simplify the systems reasoning.
- **When showing code in chat, give complete files** rather than diffs.
- **Never claim a component works** without running its correctness test. "Done" means the test passes.
- **Don't consult vLLM or nano-vllm source** to design a component until my version passes its tests. Comparing designs afterward is encouraged.
- If a change touches measurement code, say so explicitly. Results from before and after a measurement change are not comparable.

## Hardware and environment

- **GPU:** RTX 3050 Laptop, 6 GB GDDR6, Ampere (compute capability 8.6), 20 SMs (full GA107, measured; the 4 GB variant has 16).
  - bf16 and Triton are supported. **FP8 is not** (needs Ada/Hopper), so quantization uses INT8.
  - Memory bandwidth: 168 GB/s theoretical (96-bit GDDR6 at 14 Gbps). **Measured 156 GB/s** (`results/roofline/20261011-*`). Use the measured value for all analysis.
  - Measured bf16 matmul peak: 18.7 TFLOP/s, so the ridge point is about 120 FLOP/B.
- **OS:** Linux or WSL2. Triton and vLLM target Linux.
- **Laptop benchmarking rules:** run plugged in, in the highest power mode (Balanced caps the GPU at 30 W instead of 75 W; an HF decode step took 47 ms in Balanced vs 24–31 ms in Phase 0). Warm up before timing. Report at least 3 repetitions as mean and spread. Check throttle reasons (`nvidia-smi -q -d PERFORMANCE`) before and after each run.

## Models

Develop on the 0.5B model. Run final experiments on the 1.5B model, where KV memory is scarce on 6 GB.

| Model | Layers | Q heads | KV heads | head_dim | hidden | KV bytes/token (bf16) | Weights (bf16) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 24 | 14 | 2 | 64 | 896 | 12,288 | ~0.99 GB |
| Qwen2.5-1.5B-Instruct | 28 | 12 | 2 | 128 | 1536 | 28,672 | ~3.1 GB |

Notes:
- KV bytes/token = 2 (K, V) × layers × KV heads × head_dim × bytes/elem.
- INT8 with one fp16 scale per (token, head, K-or-V): 2 × layers × KV heads × (head_dim + 2). That gives 6,336 B for 0.5B and 14,560 B for 1.5B.
- Qwen2 quirks: the Q/K/V projections have biases, and o_proj does not. `rope_theta` = 1,000,000. RoPE uses the HF `rotate_half` pairing (i, i + d/2), not adjacent pairs. Check `tie_word_embeddings` in `config.json`.
- GQA group size g = Q heads / KV heads: 7 for 0.5B, 6 for 1.5B.

## Repository layout

```
engine/            # the serving engine (Phases 1–5)
bench/
  trace.py         # build trace files
  runner.py        # open-loop trace replay against an engine
  metrics.py       # raw timestamps -> summary metrics
  roofline.py      # bandwidth, matmul peak, decode-shaped sweep, launch overhead
  telemetry.py     # GPU (nvidia-smi) and host-CPU (Windows counters) sampling during a run
  profile_decode.py # torch.profiler breakdown of a batch-1 decode step (HF or this engine)
  cpu_probe.py     # CPU-only timing probe for host-side variance
  paged_gather.py  # paged gather + attention vs contiguous attention, per layer
  engines/
    base.py        # Engine protocol
    hf.py          # HF baseline adapter
    fake.py        # synthetic-timing engine for testing the harness
tests/             # correctness tests for every component
traces/            # generated .jsonl traces (committed, never edited)
results/           # one folder per run (generated, never edited by hand)
docs/design-log.md # decisions, failed attempts, and why
```

## Phases

| Phase | Weeks | Goal | Done when |
|---|---|---|---|
| 0. Measurement and roofline | 1 | A trusted benchmark harness and a measured roofline | Fake-engine tests pass. Bandwidth, matmul peak, and launch overhead are measured. HF baseline is run at 3 rates. |
| 1. Forward pass and contiguous KV cache | 2–3 | Qwen2 in plain PyTorch, logits matching HF | fp32 logits are within ~1e-4 of HF. Cached and uncached greedy generation match exactly for 50 tokens. |
| 2. Paged KV cache and block manager | 4 | Fixed-size blocks, block tables, slot mapping | The randomized alloc/append/free test shows no leaks. Outputs match Phase 1 exactly. |
| 3. Triton paged-attention decode | 5–6 | Kernel reading the paged cache in place, plus split-K | Matches the reference on all edge cases. Bandwidth-vs-context plot is made. Split-K improves batch-1 long context. |
| 4. Continuous batching scheduler | 7–8 | Iteration-level scheduling, admission, recompute preemption, chunked prefill | Batch-invariance and forced-preemption tests pass. Throughput and p99 TPOT vs. rate plots are made. |
| 5. INT8 KV cache (contribution) | 9 | Halve KV memory with dequantization fused into the kernel | Three-config table is complete: bf16, INT8 per-token K/V, INT8 per-channel-K (block-grouped) with per-token V. |
| 6. Benchmark vs. vLLM and write-up | 10 | A fair comparison and an honest README | Every plot reproduces from raw logs with one script. |

Alternative Phase 5, if I switch: prefix caching with hashed blocks and reference counts, evaluated on multi-turn and agent-style traces.

## Phase 0 spec (current phase)

### Traces (`bench/trace.py`)
- JSONL with one request per row: `{"id", "arrival_s", "prompt_ids", "output_len"}`.
- **Store token ids, not text.** Tokenize once with the Qwen tokenizer so every engine sees an identical workload.
- **Fixed output lengths and EOS ignored.** Every engine generates exactly `output_len` tokens, and `output_len >= 2`.
- Source: ShareGPT. Use the first human turn as the prompt and the first assistant reply's token length as `output_len`. Filters: prompt ≤ 1024, output 4–512, total ≤ 2048.
- Poisson arrivals: `np.cumsum(rng.exponential(1/rate, n))` with a recorded seed.
- Build several rates from the same request sample, plus a **burst** trace with every arrival at t = 0 to measure max throughput.
- Record seed, rate, filters, and tokenizer in a sidecar file.

### Engine interface (`bench/engines/base.py`)
```python
class Engine(Protocol):
    def add_request(self, req_id: int, prompt_ids: list[int], output_len: int) -> None: ...
    def step(self) -> list[tuple[int, int, bool]]: ...   # (req_id, token_id, finished)
    def has_unfinished(self) -> bool: ...
```
- One `step()` is one forward pass worth of work. The Phase 4 engine must fit this interface unchanged.
- **HF baseline (`hf.py`):** FIFO queue at batch 1. Use a manual prefill/decode loop with `past_key_values`, not `generate()`. Load bf16 with `attn_implementation="sdpa"`, use greedy sampling, and ignore EOS.

### Runner (`bench/runner.py`)
- **Open-loop replay** in wall-clock time. Single process, single thread, with no asyncio.
- Arrival is recorded as the **scheduled** time, not the admission time.
- Token timestamps are taken **after** the sampled token reaches the CPU (`.item()`/`.tolist()` syncs). Never timestamp right after launching GPU work.
- Log queue depth on every step.

### Metrics (`bench/metrics.py`)
- Write raw per-request records first (`results/<run>/requests.jsonl`: arrival, every token time, finish). Compute summaries separately.
- TTFT = t1 − arrival. TPOT = (tn − t1)/(n − 1). ITL is the pooled distribution of gaps. E2E = tn − arrival.
- Report mean, p50, p90, and p99. Also report output throughput, achieved request rate, and goodput against a fixed SLO (TTFT < 2 s, TPOT < 100 ms, unless changed in the design log).
- Drop the first ~10% of requests as warmup.
- **Overload flag:** raise it when the achieved rate is below the offered rate or the queue is still growing at the end.

### Choosing rates for the baseline
Measure the mean service time S from the burst trace. Capacity is about 1/S. Run at about 0.3×, 0.7×, and 1.2× capacity.

### Roofline (`bench/roofline.py`)
All timings use `torch.cuda.Event`, after warmup, as the median of ~20 repetitions.
- **Bandwidth:** `dst.copy_(src)` on a ~512 MB bf16 tensor, where BW = 2 × bytes / time.
- **Peak matmul:** square bf16 matmuls from 1024 to 8192, where FLOP/s = 2MNK / time. Take the max.
- **Decode-shaped sweep:** K = 896, N = 4864, M from 1 to 512, with I = 2MKN / (2(MK + KN + MN)). Plot achieved FLOP/s vs. I on log-log axes against min(peak, I × BW).
- **Launch overhead:** 1000 tiny kernels with one sync at the end, giving per-launch µs. Multiply by the kernel count per decode step (from the PyTorch profiler).

### Environment record
Every run folder gets `env.json` with the GPU, driver, CUDA, torch, git commit, trace hash, and `nvidia-smi` clocks, temperature, power, and throttle reasons before and after, plus WSL memory. Campaign runs also get host CPU clocks before and after, and `telemetry.json` from GPU (1 s) and CPU (2 s) sampling during the run. Throttle counters run while the GPU is idle, so judge throttling by the active-reason mask over busy samples.

### Harness self-test (`bench/engines/fake.py`)
The fake engine sleeps a·P for prefill and b per decode step. Tests must verify three things:
- Measured TPOT ≈ b and unloaded TTFT ≈ a·P.
- Mean inter-arrival ≈ 1/λ.
- The overload flag fires above 1/S.

## Correctness invariants (keep these tests green forever)

1. **Logit match:** fp32 within ~1e-4 of HF. In bf16, argmax should agree on nearly all positions.
2. **Cache consistency:** cached and uncached greedy generation are token-identical (fp32).
3. **Block manager:** randomized operations leave every block free exactly once, with no leaks.
4. **Paged equals contiguous:** paged outputs match Phase 1 outputs, including lengths that aren't multiples of the block size and length 1.
5. **Kernel equals reference:** this covers shuffled block tables, ragged lengths within a batch, non-multiple lengths, and length 1.
6. **Batch invariance:** each request's greedy fp32 output is identical alone or in any batch mix.
7. **Forced preemption:** with a tiny block pool, all requests finish with unchanged outputs and no leaked blocks.

## Key reasoning to keep in mind

- **Decode linear layers have arithmetic intensity ≈ batch size**, so batching is what makes them efficient. With a ridge of about 120 FLOP/B, they stay memory-bound up to batch ≈ 128–192.
- **cuBLAS dip at small batch:** for [M, 896] × [896, 4864], M = 1–2 gets 131–133 GB/s, but **M = 4–16 drops to 73–89 GB/s** (M = 4 takes 120 µs vs 66 µs at M = 2, for the same bytes), because cuBLAS switches kernels. M = 32–128 recovers to about 112 GB/s. A throughput dip at batch 4–16 in Phase 4 is cuBLAS, not the scheduler.
- **Decode attention has arithmetic intensity = g (GQA group size)** regardless of batch, because each sequence reads its own cache. Judge the attention kernel by achieved bandwidth against the measured BW, aiming for 60–80%.
- **Batch-1 upper bound** is bandwidth ÷ weight bytes: **158 tok/s for 0.5B** (988 MB / 156 GB/s = 6.3 ms) and about 50 tok/s for 1.5B. For small models, Python and launch overhead exceed the GPU time. HF eager does 1006 kernels per decode step at 11.3 µs per launch, which is 11.4 ms of launch overhead alone, and a measured step of 25–31 ms. That motivates CUDA graphs (a Phase 4 stretch goal).
- **Store keys after RoPE.** Positions are absolute.
- **Quantization eval pitfall:** single-shot prefill attends to fresh bf16 K/V and hides quantization error. Evaluate perplexity through chunked prefill (32–64 token chunks).

## Benchmarking and reporting rules

- Use the same trace file, model, dtype, max context, and sampling settings for every engine compared.
- **Matching memory budget for vLLM:** set `gpu_memory_utilization` so its KV cache is close to mine, and report both sizes.
- Compare against vLLM both with CUDA graphs and with `enforce_eager`. Eager vs. eager isolates scheduling and kernel quality.
- Never overwrite a results folder. Every plot is regenerated from raw logs by a script.
- Report honestly, in the form "X% of vLLM eager throughput at Y req/s". Don't claim to beat vLLM overall. State limitations plainly.

## Design log

After each phase, add 2–3 lines to `docs/design-log.md` covering what was chosen, what was tried and failed, and why. It becomes the README's design section.

## Primary references by phase

- **P0:** kipply, "Transformer Inference Arithmetic"; Horace He, "Making Deep Learning Go Brrrr From First Principles"; *How to Scale Your Model* (rooflines and inference chapters).
- **P1:** HF `modeling_qwen2.py`; RoFormer; the GQA paper (Ainslie et al. 2023).
- **P2:** Kwon et al., PagedAttention (SOSP 2023); OSTEP paging chapters.
- **P3:** Triton tutorials; Milakov & Gimelshein, online softmax; FlashAttention-2; the Flash-Decoding blog post.
- **P4:** Orca (OSDI 2022); Sarathi-Serve (OSDI 2024).
- **P5:** KIVI (2024); KVQuant (2024).
- **P6:** Heiser, "Systems Benchmarking Crimes".