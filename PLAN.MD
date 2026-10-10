1. [00Your laptop GPU](#setup)
2. [··How to study](#method)
3. [P0Measurement and the roofline](#p0)
4. [P1Forward pass and KV cache math](#p1)
5. [P2Paged KV cache](#p2)
6. [P3Triton attention kernels](#p3)
7. [P4Continuous batching](#p4)
8. [P5INT8 KV cache](#p5)
9. [P6Benchmark and write-up](#p6)

# Building a small LLM serving engine, one phase at a time

A study guide for an RTX 3050 Laptop (6 GB). Each phase lists what to learn, the math worked through with real numbers, what to build, how to check yourself, and what to read. It assumes you already know tokens, logits, and what attention computes.

## Before you startWorking with a 6 GB laptop GPU

The RTX 3050 Laptop is an Ampere card (compute capability 8.6). That matters in four ways for this project.

- **It supports bf16 and Triton, but not FP8.** FP8 tensor cores start with Ada/Hopper, so Phase 5 uses INT8 for the quantized KV cache. That is a fine choice, and arguably the more instructive one.
- **Memory bandwidth is roughly 190–220 GB/s**, depending on the laptop's memory clock. You will measure the real number in Phase 0. Almost every performance argument in this guide runs through that one figure.
- **6 GB is a feature, not a limitation.** On a big GPU, KV-cache pressure only shows up with large models. On yours it shows up with a 1.5B model, so scheduling, preemption, and quantization all have measurable effects.
- **Use Linux or WSL2.** Triton and vLLM target Linux. On Windows, do everything inside WSL2 with the NVIDIA driver's WSL support.

Develop on **Qwen2.5-0.5B-Instruct** (fast iteration, about 1 GB of weights in bf16). Run the final experiments on **Qwen2.5-1.5B-Instruct**, where KV memory is genuinely scarce. Use the calculator to see why.

KV-cache budget on your card

Model

KV-cache dtype

GPU memory (GB)

Reserved for CUDA context, activations, fragmentation (GB)

Context length per sequence (tokens)

KV bytes per token–

Weights (bf16)–

Left for KV cache–

Tokens of KV that fit–

Full-length sequences that fit–

weights

reserved

KV cache

bytes/token = 2 (K and V) × layers × KV heads × head_dim × bytes per element. For INT8, each (token, layer, head, K-or-V) also stores a 2-byte scale. GB here means 109 bytes.

Two laptop-specific habits will save you from bad data later. Benchmark plugged in, in the highest power mode, with the lid open. Then check for thermal throttling with `nvidia-smi -q -d PERFORMANCE`, which lists active throttle reasons. Laptop GPUs can drop clocks after a minute of sustained load, which silently turns a 2-minute benchmark into a different experiment from a 10-second one.

## How to use this guideThe study loop for every phase

Each phase follows the same loop. It is built around working a concrete numeric example before reading the formal version, then letting tests decide when you understand something.

1. **Read the concepts here**, working each example by hand or in a few lines of NumPy.
2. **Read the primary paper or post** listed for the phase. With the example already in your head, the notation will land faster.
3. **Build the component yourself** before looking at nano-vllm or vLLM source.
4. **Test it against a reference** (HF transformers, PyTorch SDPA, or your own earlier phase). A component is done when the test passes, not when the code looks right.
5. **Measure it** with the Phase 0 harness, and only then read the reference implementation to compare designs.
6. **Write two or three lines in a design log:** what you chose, what you tried that failed, and why. This becomes your README and your interview stories.

Rough timing is 9–10 weeks part-time. Phases 1 and 3 take the longest, because they hold the most new material.

## Phase 0 · week 1Measurement and the roofline

Goal: a benchmark harness you trust, and a quantitative model of why decoding is slow.

### What to learn

#### 1. The serving metrics

For one request with prompt length \\(P\\) that produces \\(n\\) output tokens, with timestamps for arrival, first token, and last token:

- **TTFT** (time to first token) = first-token time − arrival time. It is dominated by queueing plus prefill.
- **TPOT** (time per output token) = \\(\\dfrac{\\text{E2E} - \\text{TTFT}}{n - 1}\\). This is the steady decode speed a user feels.
- **ITL** (inter-token latency) is the full distribution of gaps between consecutive tokens. Its p99 exposes stalls that average TPOT hides, for example when a long prefill blocks decodes.
- **Throughput** is output tokens per second across all requests. **Goodput** is the request rate you can sustain while still meeting a latency target such as p99 TPOT under 50 ms.

#### 2. Generating a realistic workload

Real requests arrive at random, not in neat batches. Model arrivals as a Poisson process with rate \\(\\lambda\\) requests/second. The gaps between arrivals are exponential, and you can sample one with \\(\\Delta = -\\ln(U)/\\lambda\\) where \\(U \\sim \\text{Uniform}(0,1)\\). Draw prompt and output lengths from a real distribution, such as ShareGPT conversations (the dataset vLLM's serving benchmark uses), rather than fixed lengths.

**Little's law** connects load to memory: average concurrency \\(N = \\lambda \\cdot W\\), where \\(W\\) is the average time a request spends in the system.

worked example

At 2 requests/s with an average of 10 s per request, about 20 sequences are in flight at once. If each holds about 1,000 tokens of context, that is 20,000 tokens of KV cache, which is 246 MB for Qwen2.5-0.5B at 12,288 bytes/token. This is how you will choose request rates that actually stress your 6 GB card.

#### 3. The roofline model

Any kernel is limited either by arithmetic or by memory traffic. Define **arithmetic intensity** \\(I\\) as FLOPs performed per byte moved from GPU memory. Then

\\\[ \\text{attainable FLOP/s} = \\min\\big(\\text{peak FLOP/s},\\; I \\times \\text{bandwidth}\\big). \\\]

The crossover \\(I^\* = \\text{peak} / \\text{bandwidth}\\) is the **ridge point**. Below it you are memory-bound; above it, compute-bound.

#### 4. Why decode is memory-bound: the core calculation

A linear layer with weight \\(W \\in \\mathbb{R}^{m \\times n}\\) applied to \\(B\\) tokens does \\(2Bmn\\) FLOPs (one multiply and one add per weight per token). It reads \\(2mn\\) bytes of bf16 weights, and the activations are small by comparison. So

\\\[ I\_{\\text{linear}} \\approx \\frac{2Bmn}{2mn} = B \\;\\text{FLOP/byte}. \\\]

During decode each sequence contributes one token per step, so \\(B\\) is the batch size. At batch 1, \\(I \\approx 1\\), far below any GPU's ridge point (typically 100+ FLOP/byte for tensor-core matmuls). The GPU spends its time streaming weights. **Batching is how you raise \\(I\\)**, because every token in the batch reuses the same weight read. That single fact is the reason continuous batching (Phase 4) exists.

worked example: your speed limit

Qwen2.5-0.5B has about 0.49B parameters, so about 0.99 GB in bf16. Each decode step must read all of it at least once. At 192 GB/s, that gives

\\\[ \\frac{192 \\text{ GB/s}}{0.99 \\text{ GB/step}} \\approx 194 \\text{ steps/s} \\\]

as an upper bound at batch 1, about 5.2 ms per token. For the 1.5B model (≈3.1 GB) the bound is about 62 tokens/s. Your HF baseline will land well below this. The gap is partly kernel launch overhead: 24 layers × roughly 10–15 kernels each, at several µs per launch, adds a few milliseconds of CPU time, which is the same order as the 5 ms of memory time. Small models are often limited by Python and launch overhead rather than by the GPU. That is why CUDA graphs appear in Phase 4.

### What to build

- A load generator that replays a trace (arrival time, prompt, output length) against any engine with a common interface. Use the same trace file for every engine you ever compare.
- A metrics recorder for per-token timestamps that reports TTFT, TPOT, ITL p50/p99, throughput, and goodput.
- An HF `generate` baseline at batch 1.
- **Your own roofline.** Measure bandwidth with a large device-to-device copy (bytes moved = 2 × tensor size, read plus write) and peak matmul throughput with large bf16 `torch.matmul` calls, both timed with `torch.cuda.Event` after warmup. These two numbers replace the spec-sheet guesses for the rest of the project.

### Check yourself

Why does doubling the batch size roughly double decode throughput at small batch sizes, but not forever?

At small \\(B\\), each step's cost is dominated by reading weights once, and that cost is shared across all \\(B\\) sequences. Throughput grows almost linearly with \\(B\\) until either \\(I \\approx B\\) approaches the ridge point (compute-bound), or the per-sequence costs that do not amortize, chiefly reading each sequence's own KV cache in attention, start to dominate.

Your measured TTFT spikes at high request rates even though prefill time is constant. Why?

Queueing. TTFT includes waiting time before the request is scheduled. As utilization approaches capacity, queueing delay grows sharply, much like in an M/M/1 queue, where waiting time scales with \\(\\rho/(1-\\rho)\\).

#### Done when

- The harness reports all metrics for the HF baseline at three request rates.
- You have your measured bandwidth and matmul TFLOP/s written in your design log.
- You can explain why the batch-1 decode speed bound for the 0.5B model is about 190 tokens/s.

### Reading

- kipply, “Transformer Inference Arithmetic”. The clearest treatment of FLOPs vs. bytes for inference, with worked numbers. Read this first.
- Horace He, “Making Deep Learning Go Brrrr From First Principles”. Compute vs. memory vs. overhead as three separate bottlenecks. The overhead section explains your small-model numbers.
- Google DeepMind, *How to Scale Your Model*, the rooflines chapter and the inference chapter. A rigorous roofline treatment applied to transformers.
- Databricks, “LLM Inference Performance Engineering: Best Practices”. Practical metric definitions (TTFT, TPOT) and how batching trades latency for throughput.
- vLLM's serving benchmark (`vllm bench serve`) and its source. Read this after building your own harness, to compare metric definitions.

## Phase 1 · weeks 2–3Your own forward pass, and the KV cache from first principles

Goal: a from-scratch Qwen2 forward pass whose logits match Hugging Face, with a contiguous KV cache you fully understand.

### What to learn

#### 1. One Qwen2 decoder layer

For a hidden state \\(x\\) (dimension 896 for the 0.5B model), each of the 24 layers computes:

\\\[ h = x + \\text{Attn}\\big(\\text{RMSNorm}(x)\\big), \\qquad y = h + \\text{MLP}\\big(\\text{RMSNorm}(h)\\big). \\\]

After the last layer there is a final RMSNorm and the LM head. Open `config.json` and note `tie_word_embeddings`: when it is true, the LM head reuses the embedding matrix. Also note that Qwen2's Q, K, and V projections have biases, while the output projection does not. Forgetting those biases is a common first bug.

#### 2. RMSNorm

\\\[ \\text{RMSNorm}(x) = \\frac{x}{\\sqrt{\\tfrac{1}{D}\\sum\_{i=1}^{D} x_i^2 + \\varepsilon}} \\odot g. \\\]

Unlike LayerNorm, it does not subtract the mean and has no bias. Compute it in fp32 even when the model runs in bf16, then cast back.

#### 3. The SwiGLU MLP

\\\[ \\text{MLP}(x) = W\_{\\text{down}}\\big(\\text{SiLU}(W\_{\\text{gate}}x) \\odot W\_{\\text{up}}x\\big), \\qquad \\text{SiLU}(z) = z\\,\\sigma(z). \\\]

For the 0.5B model the intermediate size is 4,864. The MLP holds most of the layer's parameters, which is why it dominates the weight traffic you computed in Phase 0.

#### 4. Rotary position embeddings (RoPE)

Start with the concrete version. Take a 4-dimensional query \\(q = (q_0, q_1, q_2, q_3)\\) at position \\(p\\). RoPE groups the dimensions into pairs and rotates each pair by an angle proportional to the position, with a different speed per pair:

\\\[ \\theta_i = b^{-2i/d}, \\quad i = 0, \\dots, \\tfrac{d}{2}-1, \\qquad \\begin{pmatrix} q'\_a \\\\ q'\_b \\end{pmatrix} = \\begin{pmatrix} \\cos p\\theta_i & -\\sin p\\theta_i \\\\ \\sin p\\theta_i & \\cos p\\theta_i \\end{pmatrix} \\begin{pmatrix} q_a \\\\ q_b \\end{pmatrix}. \\\]

The same rotation is applied to keys. Since rotations compose, \\(\\langle R_m q, R_n k\\rangle = \\langle q, R\_{n-m} k\\rangle\\): the attention score depends only on the relative offset \\(n - m\\). Qwen2.5 uses base \\(b = 1{,}000{,}000\\) (`rope_theta`).

pitfall: which dimensions form a pair

The original RoFormer formulation pairs adjacent dimensions \\((x_0, x_1), (x_2, x_3), \\dots\\). The Hugging Face implementation (`rotate_half`) pairs \\(x_i\\) with \\(x\_{i + d/2}\\). The checkpoint weights were trained with the HF layout, so the interleaved version gives plausible-looking but wrong logits. If your logits are close but not matching, check this first.

#### 5. Grouped-query attention (GQA)

Qwen2.5-0.5B has \\(H = 14\\) query heads but only \\(H\_{kv} = 2\\) key/value heads, so each KV head is shared by a group of \\(g = H/H\_{kv} = 7\\) query heads. Query head \\(h\\) uses KV head \\(\\lfloor h / g \\rfloor\\). With head dimension \\(d = 64\\), the K and V projections output only \\(2 \\times 64 = 128\\) values per token, against \\(14 \\times 64 = 896\\) for Q. GQA exists almost entirely to shrink the KV cache, which you will quantify below.

#### 6. Causal attention, written out

For one head, with token positions \\(i\\) (query) and \\(j\\) (key):

\\\[ s\_{ij} = \\frac{q_i \\cdot k_j}{\\sqrt{d}}, \\qquad a\_{ij} = \\frac{e^{s\_{ij}}}{\\sum\_{j' \\le i} e^{s\_{ij'}}} \\;\\text{ for } j \\le i, \\qquad o_i = \\sum\_{j \\le i} a\_{ij}\\, v_j. \\\]

The constraint \\(j \\le i\\) (the causal mask) is what makes the KV cache possible.

#### 7. Why the KV cache is exact, not an approximation

Follow the dependencies. The key at position \\(j\\) in layer \\(\\ell\\) is \\(k_j^{(\\ell)} = \\text{RoPE}\_j\\big(W_K\\, \\text{RMSNorm}(x_j^{(\\ell)}) + b_K\\big)\\). The hidden state \\(x_j^{(\\ell)}\\) depends only on tokens \\(0, \\dots, j\\), because of the causal mask in every earlier layer. When you append token \\(t+1\\), nothing at positions \\(\\le t\\) can change. So every \\(k_j\\) and \\(v_j\\) can be computed once, stored, and reused forever.

Meanwhile, the output at the newest position needs only its own query \\(q_t\\). Old queries are never used again, which is why there is no Q cache.

worked example: how much work the cache saves

Prompt of \\(P = 500\\) tokens, generating \\(n = 200\\). Without a cache, step \\(t\\) re-runs the full forward pass over \\(P + t\\) tokens: \\(\\sum\_{t=0}^{199}(500 + t) = 119{,}900\\) token-forward passes. With a cache: 500 tokens of prefill, then 200 single-token steps, for 700 total. That is about 171× less linear-layer work. Attention still scales with context length at every step, because each new query reads all cached keys and values.

#### 8. Prefill and decode are two different computations

Look at tensor shapes for one sequence in one layer:

|  | Prefill (P prompt tokens) | Decode (1 new token) |
| --- | --- | --- |
| Q | `[P, 14, 64]` | `[1, 14, 64]` |
| New K, V | `[P, 2, 64]` each, all written to cache | `[1, 2, 64]` each, appended to cache |
| K, V read in attention | the same P tokens just computed | all \\(T\\) cached tokens |
| Linear layers | matrix × matrix, compute-bound | matrix × vector, memory-bound |

This asymmetry drives most design decisions in Phases 3 and 4.

#### 9. How the cache is stored, and how big it is

The simplest layout: for each layer, preallocate `k_cache` and `v_cache` of shape `[batch, H_kv, T_max, d]` and write new entries at index `pos`. Two details matter:

- **Store keys after RoPE.** Each cached key is already rotated for its absolute position, so it never needs touching again.
- **Positions are absolute.** During decode, the new token's RoPE angle uses its true position \\(T\\), not 0. Getting this wrong produces fluent text for a few tokens and then degradation.

The size per token follows directly from the shapes:

\\\[ \\text{bytes per token} = \\underbrace{2}\_{K,\\,V} \\times \\underbrace{L}\_{\\text{layers}} \\times H\_{kv} \\times d \\times \\underbrace{s}\_{\\text{bytes/elem}}. \\\]

worked example: three models on your card

| Model | L | Hkv | d | bytes/token (bf16) | 32 seqs × 2,048 tokens |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 24 | 2 | 64 | 12,288 | 0.81 GB |
| Qwen2.5-1.5B | 28 | 2 | 128 | 28,672 | 1.88 GB |
| Llama-3.2-1B | 16 | 8 | 64 | 32,768 | 2.15 GB |
| 0.5B if it had no GQA | 24 | 14 | 64 | 86,016 | 5.64 GB |

The 1.5B model's weights take about 3.1 GB, so 32 sequences at 2,048 tokens of context would not fit in 6 GB alongside them. That is the memory pressure Phases 2, 4, and 5 deal with. The last row shows what GQA buys: without it, the 0.5B model's cache would be 7× larger.

### What to build

- A `Qwen2Model` in plain PyTorch that loads weights from the safetensors checkpoint (use `safetensors` directly, not `AutoModel`).
- A `generate` function with a contiguous KV cache: one prefill call, then a decode loop. Use greedy sampling for now.

#### Tests that prove it works

- **Logit match.** In fp32, your logits should match HF's within about 1e-4 max absolute difference. In bf16, expect larger differences, but the argmax should agree on nearly every position.
- **Cache consistency.** Generating 50 tokens with the cache must produce exactly the same tokens (fp32, greedy) as recomputing the full sequence at every step without a cache. This is the single best test of your position handling.

### Check yourself

Why is there a KV cache but no Q cache?

Each query is used only once, at the step its token is generated. Keys and values are read by every later query, so they are worth storing.

Does the KV cache make attention itself cheaper per step?

No. It removes the recomputation of K and V for old tokens (and the linear-layer work for them), but the new query still attends over all \\(T\\) cached positions, so attention work per step still grows linearly with context.

Why would applying RoPE to keys at read time (instead of before caching) be worse?

You would re-rotate every cached key at every decode step, adding compute and memory traffic proportional to context length for no benefit, since a key's position never changes.

#### Done when

- Logit-match and cache-consistency tests pass.
- You can derive the bytes-per-token formula from tensor shapes without looking.
- Your engine's batch-1 decode speed is measured in the Phase 0 harness against the HF baseline.

### Reading

- Hugging Face transformers, `modeling_qwen2.py`. Your ground truth. Read it alongside writing your own; it is short.
- Sebastian Raschka's article on coding a KV cache from scratch, and *Build a Large Language Model (From Scratch)*. Reading-first, code-along treatment of exactly this phase.
- Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embedding” (2021), and EleutherAI's blog post “Rotary Embeddings: A Relative Revolution”. The blog is the gentler entry; the paper has the derivation.
- Shazeer, “Fast Transformer Decoding: One Write-Head is All You Need” (2019), and Ainslie et al., “GQA” (2023). MQA and GQA, framed exactly as a KV-cache memory problem.
- Zhang & Sennrich, “Root Mean Square Layer Normalization” (2019), and Shazeer, “GLU Variants Improve Transformer” (2020). Short; skim for the definitions.

## Phase 2 · week 4Paged KV cache and the block manager

Goal: replace the contiguous cache with fixed-size blocks managed like virtual memory, with outputs identical to Phase 1.

### What to learn

#### 1. What goes wrong with contiguous caches

A contiguous cache must reserve space for the longest possible sequence, because you don't know in advance when generation stops.

worked example

Reserve `T_max = 2048` per sequence when the average sequence actually reaches 600 tokens. Then \\(1 - 600/2048 \\approx 71\\%\\) of the reserved KV memory holds nothing. In the same memory, allocating only what each sequence actually uses would fit about 3.4× as many sequences. On top of this internal waste, sequences of varying reservation sizes fragment the free space between them.

#### 2. Paging: the core idea

Split KV storage into fixed-size **blocks** of \\(B\\) tokens (16 is typical). All blocks live in one preallocated pool. Each sequence keeps a **block table**: a list of physical block ids, in logical order. Blocks can be anywhere in the pool; only the table knows the order. This is exactly how an operating system's page table maps virtual pages to physical frames.

Address translation for the token at logical position \\(p\\):

\\\[ \\text{physical block} = \\text{table}\\big\[\\lfloor p / B \\rfloor\\big\], \\qquad \\text{offset} = p \\bmod B, \\qquad \\text{slot} = \\text{physical block} \\cdot B + \\text{offset}. \\\]

worked example

With \\(B = 4\\), a sequence of 11 tokens has the block table `[7, 2, 9]`. Token 10 lives in logical block \\(\\lfloor 10/4 \\rfloor = 2\\), which is physical block 9, at offset 2, so slot \\(9 \\cdot 4 + 2 = 38\\). The sequence's last block holds tokens 8, 9, 10, and one empty slot. When token 11 arrives it fills that slot. Token 12 needs a new block, so the allocator pops one from the free list and appends it to the table.

#### 3. The physical layout

One straightforward choice, per layer:

```
k_cache[layer]: [num_blocks, B, H_kv, d]   # bf16
v_cache[layer]: [num_blocks, B, H_kv, d]
```

One block id indexes the same slot range in every layer, so the block manager tracks blocks once, not per layer. The per-token byte count from Phase 1 already includes all layers, so

\\\[ \\text{num\\\_blocks} = \\left\\lfloor \\frac{\\text{KV budget}}{B \\times \\text{bytes per token}} \\right\\rfloor. \\\]

worked example

Qwen2.5-0.5B, \\(B = 16\\): one block costs \\(16 \\times 12{,}288 = 196{,}608\\) bytes across all layers. A 3 GiB budget gives \\(3 \\cdot 2^{30} / 196{,}608 = 16{,}384\\) blocks, or 262,144 tokens. Each sequence now wastes at most \\(B - 1 = 15\\) slots, about 7.5 on average, instead of hundreds.

Production engines use other layouts (for example, splitting the head dimension so a kernel's vector loads align). Start with the simple one and only change it if your Phase 3 profiling shows a reason.

#### 4. Writing new K/V: slot mapping

For each new token in a batch, the scheduler computes its slot. The write is then a single scatter:

```
k_flat = k_cache[layer].view(-1, H_kv, d)   # [num_blocks * B, H_kv, d]
k_flat[slot_mapping] = k_new                # k_new: [num_new_tokens, H_kv, d]
```

This is one of the first places to watch for silent bugs: a wrong slot writes over another sequence's cache, and the result is plausible but wrong text.

#### 5. Allocation, freeing, and sharing

- **Allocate** \\(\\lceil P / B \\rceil\\) blocks at admission, then one more each time a sequence's length crosses a multiple of \\(B\\).
- **Free** all of a sequence's blocks when it finishes or is preempted. A free list used as a stack is enough.
- **Reference counts** let several sequences share a block, for example a common system prompt. A shared block that needs to be written is first copied (copy-on-write). You don't need sharing until the prefix-caching variant of Phase 5, but design the block manager with a `ref_count` field now.

#### 6. Choosing the block size

Small blocks waste less memory but mean longer block tables and smaller, less efficient memory loads inside the kernel. Large blocks are the reverse. You will measure this tradeoff in Phase 3 by sweeping \\(B \\in \\{8, 16, 32\\}\\).

#### 7. Attention on paged memory, the slow version

Before writing a kernel, gather each sequence's blocks into a contiguous tensor (`k_cache[layer][block_table]`, then reshape and trim to its length) and call PyTorch's SDPA. It is correct and simple, but it copies the whole cache on every step, which doubles memory traffic. Measure how slow it is: that number is the motivation for Phase 3.

### What to build

- A `BlockManager` with `can_allocate`, `allocate`, `append_slot`, `free`, and `num_free_blocks`.
- Per-step metadata construction: block tables (padded into a 2D tensor), context lengths, and the slot mapping.
- The gather-based paged attention described above.

#### Tests that prove it works

- Unit tests for the block manager, plus a randomized test: thousands of random allocate, append, and free operations, after which every block is free exactly once. A leak here becomes a mysterious out-of-memory in Phase 4.
- End-to-end: greedy outputs match Phase 1 exactly for sequences whose lengths are not multiples of \\(B\\), including a prompt of length 1.

### Check yourself

A sequence has 33 tokens with B = 16. How many blocks does it hold, and how many slots are empty?

\\(\\lceil 33/16 \\rceil = 3\\) blocks, holding 48 slots, so 15 are empty. That is the worst case: a sequence that just crossed a block boundary.

Why does one block id cover all layers instead of each layer allocating separately?

Every layer stores K/V for exactly the same token positions, so their allocation patterns are identical. Sharing one id means one block table per sequence instead of one per layer, and one allocator decision per block instead of 24.

#### Done when

- The randomized block-manager test passes and outputs match Phase 1.
- You have measured the gather-based attention's cost against Phase 1's contiguous attention.

### Reading

- Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention” (SOSP 2023), sections 3–4. The primary source for this phase. The memory-waste measurements in section 3 are worth reading closely.
- *Operating Systems: Three Easy Pieces* (free online), the chapters “Paging: Introduction” and “Free-Space Management”. The OS ideas PagedAttention borrows, explained clearly.
- The vLLM launch blog post, “vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention” (2023). Short, with good diagrams of block tables and sharing.
- nano-vllm's block manager on GitHub. Read after yours works; it is small enough to read in one sitting.

## Phase 3 · weeks 5–6Triton paged-attention decode kernels

Goal: a decode attention kernel that reads the paged cache in place, reaches a large fraction of your measured bandwidth, and handles long contexts at batch 1.

### What to learn

#### 1. The GPU model you actually need

Your card has 16 streaming multiprocessors (SMs). Each runs many threads in groups of 32 (warps). Memory forms a hierarchy: large and slow GDDR6 (6 GB, \~200 GB/s), a few MB of L2 cache, then fast on-chip shared memory and registers per SM. For a memory-bound kernel, three things decide performance:

- **Coalescing.** Neighboring threads should read neighboring addresses, so one memory transaction serves a whole warp. Loading a K tile of shape `[B, d]` row by row, with `d` contiguous, is coalesced.
- **Reuse.** Every byte brought on-chip should be used as many times as possible before being discarded.
- **Enough parallel work.** If the launch grid has fewer programs than SMs, most of the chip sits idle no matter how good the kernel is.

#### 2. Triton's programming model

In Triton you write the code for one *program instance*, which handles a block of data. You get your index from `tl.program_id`, build vectors of offsets with `tl.arange`, and use masked `tl.load`/`tl.store` for edges. `tl.dot` runs small matrix multiplies on tensor cores but needs each dimension to be at least 16. Triton handles the thread-level details (warps, shared memory staging) that you would write by hand in CUDA.

#### 3. Online softmax: the trick everything depends on

A naive softmax needs the maximum over all scores before computing any exponential, which means reading all keys twice. Online softmax processes scores in one pass, keeping a running max \\(m\\), a running normalizer \\(\\ell\\), and an unnormalized output accumulator \\(o\\). For each new score \\(s_j\\) with value \\(v_j\\):

\\\[ m_j = \\max(m\_{j-1}, s_j), \\quad \\ell_j = \\ell\_{j-1}\\, e^{m\_{j-1} - m_j} + e^{s_j - m_j}, \\quad o_j = o\_{j-1}\\, e^{m\_{j-1} - m_j} + e^{s_j - m_j}\\, v_j. \\\]

At the end, the output is \\(o_n / \\ell_n\\). The factor \\(e^{m\_{j-1} - m_j}\\) rescales everything accumulated so far whenever a new maximum appears.

worked example: three scores, scalar values

Scores \\(s = (2, 5, 1)\\), values \\(v = (1, 0, 2)\\). Start with \\(m = -\\infty,\\ \\ell = 0,\\ o = 0\\).

| step | s | new m | rescale | ℓ | o (unnormalized) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 2 | – | 1 | 1 × 1 = 1 |
| 2 | 5 | 5 | e−3 = 0.04979 | 0.04979 + 1 = 1.04979 | 1 × 0.04979 + 1 × 0 = 0.04979 |
| 3 | 1 | 5 | 1 | 1.04979 + e−4 = 1.06810 | 0.04979 + 0.01832 × 2 = 0.08642 |

Result: \\(0.08642 / 1.06810 = 0.08091\\). A direct softmax gives weights \\((0.0466, 0.9362, 0.0171)\\) and the same output, \\(0.0466 + 2 \\times 0.0171 = 0.0809\\). Write this in five lines of NumPy before writing any kernel.

In the kernel, “one score” becomes “one tile of \\(B\\) scores”: take the max over the tile, rescale once per tile, and accumulate the tile's \\(p \\cdot V\\) with `tl.dot`.

#### 4. Designing the paged decode kernel

Launch one program per (sequence, KV head). Each program:

1. Loads the \\(g\\) query heads that share this KV head, as a tile of shape `[g, d]`. Pad \\(g\\) up to 16 with a mask so `tl.dot` works (\\(g = 7\\) for the 0.5B model, 6 for the 1.5B).
2. Loops over the sequence's logical blocks. For each, it reads the physical id from the block table and loads the K tile `[B, d]` straight from the paged cache.
3. Computes scores `[g, B]`, multiplies by \\(1/\\sqrt{d}\\), masks positions past the context length, and applies the online softmax update.
4. Loads the V tile and accumulates \\(p \\cdot V\\) into a `[g, d]` fp32 accumulator.
5. Divides by \\(\\ell\\) and writes the output.

Processing the whole query group in one program is the key design choice. Each K/V tile is loaded once and used by \\(g\\) heads. Launching one program per query head instead would load the same tile 7 times.

#### 5. The roofline for decode attention

For one sequence with \\(T\\) cached tokens, attention does \\(2HdT\\) FLOPs for \\(QK^\\top\\) and \\(2HdT\\) for \\(PV\\). It reads \\(2 \\times H\_{kv} \\times d \\times T \\times 2\\) bytes of bf16 K and V. So

\\\[ I\_{\\text{attn}} = \\frac{4HdT}{4H\_{kv}dT} = \\frac{H}{H\_{kv}} = g. \\\]

For the 0.5B model, \\(I = 7\\) FLOP/byte, far below the ridge point, and that holds *regardless of batch size*, because each sequence reads its own cache. Batching makes linear layers efficient but does nothing for decode attention. The right performance metric for this kernel is therefore **achieved bandwidth**, bytes moved divided by kernel time, compared against the bandwidth you measured in Phase 0. Aim for 60–80%.

#### 6. Split-K (flash-decoding) for long contexts

At batch 1, the 0.5B model has 2 KV heads, so your grid has just 2 programs on a 16-SM chip. Fourteen SMs do nothing. The fix: split each sequence's context into \\(S\\) chunks, launch a program per (sequence, KV head, chunk), and have each emit a partial result \\((m_s, \\ell_s, o_s)\\), with \\(o_s\\) unnormalized. A small second kernel combines them:

\\\[ M = \\max_s m_s, \\qquad \\ell = \\sum_s e^{m_s - M}\\,\\ell_s, \\qquad o = \\frac{\\sum_s e^{m_s - M}\\, o_s}{\\ell}. \\\]

worked example: splitting the earlier scores

Split \\((2, 5, 1)\\) into chunk A \\(= (2, 5)\\) and chunk B \\(= (1)\\). Chunk A gives \\(m_A = 5,\\ \\ell_A = 1.04979,\\ o_A = 0.04979\\). Chunk B gives \\(m_B = 1,\\ \\ell_B = 1,\\ o_B = 2\\). Combine with \\(M = 5\\): \\(\\ell = 1.04979 + e^{-4} \\cdot 1 = 1.06810\\) and \\(o = (0.04979 + e^{-4} \\cdot 2)/1.06810 = 0.08091\\). Same answer as the single pass.

Choose \\(S\\) so the total program count is a few times the SM count, and skip splitting when the batch already provides enough parallelism.

#### 7. Measuring a kernel properly

Use `triton.testing.do_bench`, which handles warmup and repeats. Compute achieved bandwidth by hand from the bytes your kernel must read. Nsight Compute gives deeper detail (DRAM throughput, occupancy, stalls), but its support under WSL2 has been limited, so check NVIDIA's current documentation. Hand-computed bandwidth is enough to drive every decision in this phase.

### What to build

- First, the Triton tutorials in order: vector add, fused softmax, matrix multiplication, then fused attention. The last one is essentially FlashAttention, which you have built before, so read it for Triton idioms.
- The single-pass paged decode kernel, then the split-K version with its reduction kernel.
- A sweep: kernel time and achieved bandwidth versus context length (128 to 8k) at batch 1, 8, and 32, and versus block size (8, 16, 32).

#### Tests that prove it works

Compare against your Phase 2 gather-plus-SDPA path in fp32, using random block tables (non-contiguous, shuffled ids), ragged context lengths within a batch, lengths that are not multiples of \\(B\\), and length 1. Then confirm end-to-end generation still matches Phase 1.

### Check yourself

Why does a larger batch not improve your kernel's achieved bandwidth much, but a longer context does (up to a point)?

Both add work, but a longer context means each program runs a longer loop of streaming loads, which hides latency well. A larger batch adds more programs, which helps only until the SMs are full. Neither changes the arithmetic intensity, which stays at \\(g\\).

Why keep the accumulator in fp32 when K and V are bf16?

You sum hundreds or thousands of terms, and bf16 has only 8 bits of mantissa. Accumulating in bf16 loses small contributions and drifts, especially with the repeated rescaling by \\(e^{m\_{j-1}-m_j}\\).

When does split-K hurt?

When the batch already fills the GPU. Then the extra partial-result writes, the second kernel launch, and the reduction are pure overhead, while parallelism was never the problem.

#### Done when

- The kernel matches the reference on all edge cases above.
- You have a plot of achieved bandwidth versus context length, with your Phase 0 bandwidth drawn as the ceiling.
- Split-K measurably improves batch-1 long-context latency.

### Reading

- The official Triton tutorials (triton-lang.org). Work through them in order; the fused-softmax tutorial teaches the row-per-program pattern you will reuse.
- Milakov & Gimelshein, “Online normalizer calculation for softmax” (2018). Four pages; the origin of the trick in section 3.
- Dao et al., FlashAttention (2022) and Dao, FlashAttention-2 (2023). Re-read with decode in mind: what changes when the query length is 1?
- The “Flash-Decoding for long-context inference” blog post (Dao, Haziza, Massa, Sizov, 2023). The source for split-K decode, with a clear diagram.
- Simon Boehm, “How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance”. The best reading-based build-up of the GPU memory-hierarchy intuition, even though you are writing Triton.
- *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj), the chapters on memory architecture and performance considerations. Reference material when a concept in a blog post feels hand-wavy.

## Phase 4 · weeks 7–8Continuous batching and the scheduler

Goal: an engine loop that admits, runs, preempts, and retires requests every step, under a fixed memory budget.

### What to learn

#### 1. Why static batching wastes the GPU

worked example

Batch two requests that generate 20 and 500 tokens. With static batching, the batch runs until the longer one finishes. The first request's slot sits idle for 480 steps, and any request arriving meanwhile waits for the whole batch. Recall from Phase 0 that each step costs roughly the same whether the batch holds 1 or 8 sequences, so those idle slots are throughput you are paying for and not using.

#### 2. Iteration-level scheduling

The fix, from the Orca paper, is to make scheduling decisions after *every* forward step instead of every batch. Finished sequences leave immediately, and waiting ones join at the next step. Your engine becomes a loop:

```
while has_work():
    batch = scheduler.schedule()      # who runs this step, and how many tokens each
    meta  = build_metadata(batch)     # input ids, positions, slot mapping, block tables, context lengths
    logits = model.forward(meta)
    tokens = sample(logits)
    scheduler.update(batch, tokens)   # append tokens, retire finished, free their blocks
```

#### 3. Packing a mixed batch without padding

Sequences in a step contribute different numbers of new tokens: one for each decode, many for a prefill. Concatenate all new tokens into one flat tensor of shape `[total_tokens, hidden]` and track boundaries with cumulative lengths (`cu_seqlens`). Linear layers, norms, and the MLP don't care which sequence a token belongs to, so they run on the flat tensor as one large matmul. That is exactly where batching's efficiency comes from, since arithmetic intensity scales with total tokens. Only attention needs the per-sequence metadata.

#### 4. Admission control

Admit a waiting request only if the pool has \\(\\lceil P/B \\rceil\\) free blocks for its prompt, plus a small **watermark** reserve. Running sequences need a new block every \\(B\\) decode steps. If you admit right up to the limit, they will run out almost immediately and force preemptions.

#### 5. Preemption: recompute or swap

When a running sequence needs a block and none are free, evict a victim, usually the most recently admitted one, so older requests keep making progress. There are two options:

- **Recompute:** free the victim's blocks and put it back in the waiting queue. When it is readmitted, its prompt plus already-generated tokens are prefilled again. Cost ≈ \\(2 \\times N\_{\\text{params}} \\times T\\) FLOPs.
- **Swap:** copy its blocks to pinned CPU memory and copy them back later. Cost = KV bytes ÷ PCIe bandwidth, each way.

worked example: plug in your own numbers

A 0.5B-model sequence with \\(T = 1{,}500\\) tokens holds \\(1{,}500 \\times 12{,}288 \\approx 18.4\\) MB of KV. Recompute costs \\(2 \\times 0.49\\times10^9 \\times 1{,}500 \\approx 1.5\\) TFLOP, which is about 150 ms if you achieve 10 TFLOP/s. Swapping at an effective 10 GB/s costs about 1.8 ms each way. Measure your laptop's real pinned-memory copy speed and your achieved prefill TFLOP/s, then redo this. Swapping is cheaper here on paper, but it adds CPU memory management and copies that compete with the next step. Implement recompute first; it is simpler and always correct.

#### 6. Chunked prefill

A 2,000-token prefill can take tens of milliseconds. If it runs as one step, every decoding sequence stalls for that long, which shows up as p99 ITL spikes. Chunked prefill sets a **token budget** per step (say 512). Each step first includes all running decodes (one token each), then fills the remaining budget with a slice of a waiting prompt. Long prompts are spread across several steps, and decodes keep a steady rhythm.

The new requirement is attention for a prefill chunk. Queries at positions \\(\[c, c + n)\\) must attend to all cached positions \\(\[0, c)\\) plus, causally, to earlier positions within the chunk. Write the chunk's K/V into the cache first. Then either gather that sequence's blocks and call SDPA with the right mask (simple, slower), or write a variable-length prefill kernel as a stretch goal.

#### 7. CPU overhead and CUDA graphs

Recall from Phase 0 that for a model this small, Python overhead and kernel launches can take as long as the GPU work. Once the scheduler works, profile a decode step with the PyTorch profiler and see how much of the step the GPU is idle. The standard fix is CUDA graphs: capture the decode forward pass once for each of a few fixed batch sizes (1, 2, 4, 8, 16, 32), pad each step's batch up to the nearest captured size, and replay the graph. That requires static input buffers, so plan your metadata tensors accordingly. Treat this as a stretch goal, and measure it separately so you can say exactly what it contributed.

### What to build

- Request states (waiting, running, preempted, finished), the scheduler with token budget and watermark, recompute preemption, and chunked prefill.
- A connection to the Phase 0 harness, so the engine accepts requests at their trace arrival times.

#### Tests that prove it works

- **Batch invariance:** in fp32 with greedy sampling, each request's output must be identical whether it runs alone or alongside any mix of other requests. This one test catches nearly every slot-mapping, position, and masking bug.
- **Forced preemption:** run with a deliberately tiny block pool so preemption happens constantly. Every request must still finish with the same output, and no blocks may leak.

### Check yourself

Why evict the most recently admitted sequence rather than the longest one?

Evicting the longest frees the most memory but throws away the most work (expensive to recompute) and can starve long requests indefinitely. Evicting the newest wastes the least completed work and gives first-come, first-served fairness.

Chunked prefill raises TTFT for the request being chunked. Why accept that?

It trades a bit of one request's TTFT for steady TPOT for every running request. The prompt takes several steps instead of one, but no decode ever waits behind a long prefill, so p99 inter-token latency drops.

#### Done when

- Batch-invariance and forced-preemption tests pass.
- You have plots of throughput and p99 TPOT versus request rate, with the knee where latency starts climbing clearly visible.
- You can compare chunked and unchunked prefill on p99 ITL with the same trace.

### Reading

- Yu et al., “Orca: A Distributed Serving System for Transformer-Based Generative Models” (OSDI 2022). Iteration-level scheduling and selective batching. Skip the distributed parts.
- Agrawal et al., “Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve” (OSDI 2024). Chunked prefill and stall-free scheduling, with the latency analysis you will reproduce.
- The PagedAttention paper's section on scheduling and preemption. Recompute versus swap, with their measurements.
- Anyscale's blog post on continuous batching (2023). An accessible picture of static versus continuous batching, with benchmarks.
- PyTorch's “Accelerating PyTorch with CUDA Graphs” blog post. For the stretch goal; explains static buffers and capture rules.
- nano-vllm's scheduler and model runner. After yours works, compare how it builds per-step metadata.

## Phase 5 · week 9Your contribution: an INT8 KV cache

Goal: halve KV memory, fuse dequantization into your kernel, and measure the capacity-versus-quality tradeoff honestly.

### What to learn

#### 1. Symmetric INT8 quantization

For a vector \\(x\\), pick a scale \\(s = \\max_i |x_i| / 127\\), then store \\(\\tilde{x}\_i = \\text{clamp}\\big(\\text{round}(x_i / s), -127, 127\\big)\\) as int8. Reconstruct with \\(\\hat{x}\_i = s\\,\\tilde{x}\_i\\). Each element's error is at most \\(s/2\\).

worked example

\\(x = (0.8,\\ -2.4,\\ 1.2,\\ 0.1)\\), so \\(s = 2.4/127 = 0.018898\\). Dividing gives \\((42.33,\\ -127,\\ 63.5,\\ 5.29)\\), which rounds to \\((42,\\ -127,\\ 64,\\ 5)\\). Reconstruction gives \\((0.7937,\\ -2.4,\\ 1.2094,\\ 0.0945)\\). Every error is within \\(s/2 \\approx 0.0094\\). The cost of storing this vector drops from 8 bytes (bf16) to 4 bytes plus one 2-byte scale.

#### 2. Granularity, and why keys and values differ

One scale covers some group of numbers. The smaller the group, the better the accuracy, and the more scales you store. The common options:

- **Per token, per head:** one scale for each token's \\(d\\)-dimensional K (or V) vector in each head. This is simple and works naturally with a growing cache.
- **Per channel:** one scale for each of the \\(d\\) dimensions, computed across a group of tokens.

The KIVI paper's key observation is that **keys have outlier channels**: a few dimensions are consistently much larger than the rest. With per-token scaling, one outlier sets the scale and crushes every small value in that vector.

worked example: the outlier problem

A key vector \\((0.1,\\ 0.2,\\ 9.0,\\ -0.15)\\) gives \\(s = 9.0/127 = 0.0709\\). The small entries become \\((1,\\ 3,\\ \\cdot,\\ -2)\\) and reconstruct to \\((0.071,\\ 0.213,\\ \\cdot,\\ -0.142)\\), with relative errors of 29%, 6%, and 5%. A per-channel scale gives the large dimension its own scale, so the small dimensions keep their precision. Values do not show this channel structure, so per-token scaling suits them.

Per-channel scaling has a practical problem during decode: the scale for a channel depends on tokens that haven't been generated yet. KIVI's solution keeps the most recent tokens in full precision and quantizes them as a group once the group fills. With a paged cache, there is a natural fit: **the group is the block**. Keep a sequence's current partial block in bf16 and quantize it per-channel when it fills. Designing and evaluating that variant is what makes this phase your own work.

#### 3. Storage layout and memory savings

```
k_cache_q[layer]: [num_blocks, B, H_kv, d]  int8
k_scale[layer]:   [num_blocks, B, H_kv]     fp16   # per-token variant
v_cache_q[layer]: [num_blocks, B, H_kv, d]  int8
v_scale[layer]:   [num_blocks, B, H_kv]     fp16
```

\\\[ \\text{bytes per token} = 2 \\times L \\times H\_{kv} \\times (d + 2). \\\]

For the 0.5B model that is 6,336 bytes instead of 12,288 (1.94× capacity). For the 1.5B model, 14,560 instead of 28,672 (1.97×). Try the INT8 option in the calculator at the top to see what that means for full-length sequences on your card.

#### 4. Fusing dequantization into attention, cheaply

You never need to rebuild a full bf16 K tile. With per-token K scales \\(s_j\\):

\\\[ q \\cdot (s_j \\tilde{k}\_j) = s_j\\,(q \\cdot \\tilde{k}\_j), \\\]

so you compute dot products against the int8 keys (cast to bf16 in registers after loading) and multiply each score by one scale. With per-channel K scales \\(c\\), shared by all tokens in a block:

\\\[ q \\cdot (c \\odot \\tilde{k}\_j) = (q \\odot c) \\cdot \\tilde{k}\_j, \\\]

so you scale the query once per block. For values with per-token scales \\(s^V_j\\):

\\\[ \\sum_j p_j\\,(s^V_j \\tilde{v}\_j) = \\sum_j (p_j\\, s^V_j)\\, \\tilde{v}\_j, \\\]

so you fold the scale into the attention probabilities. In every case the extra arithmetic is tiny, and since Phase 3 showed this kernel is memory-bound, halving the bytes read should translate fairly directly into speed. Whether it does is one of your measurements.

pitfall: evaluating with prefill hides the error

During a single-shot prefill, attention uses the freshly computed bf16 K and V, not the quantized cache. A perplexity evaluation that prefills a whole document at once therefore barely touches the quantized path, and will look perfect. Feed evaluation text through your chunked-prefill path in small chunks (say 32–64 tokens), so most attention reads quantized history.

#### 5. What to measure

| Question | Measurement |
| --- | --- |
| How much error per layer? | Relative error and cosine similarity of attention outputs versus bf16, on real activations |
| How much quality loss? | Perplexity on WikiText-2 (chunked, as above), plus a small downstream task with a fixed prompt set |
| How much capacity gained? | Maximum concurrent sequences before preemption, on the 1.5B model |
| Does it help serving? | Throughput and p99 TPOT versus request rate under memory pressure, against the bf16 cache on the same trace |
| Is the kernel still efficient? | Achieved bandwidth and kernel time versus the bf16 kernel |

Run this for three configurations: bf16, INT8 per-token K and V, and INT8 per-channel-K (block-grouped) with per-token V. That table is the centerpiece of your write-up.

### Check yourself

Why use symmetric quantization (range −127 to 127) rather than asymmetric with a zero point?

K and V entries are roughly centered on zero, so a zero point buys little range, and symmetric scaling keeps the fused-dequantization algebra above clean: one multiply, no offset term inside the dot product.

Halving KV bytes halves attention kernel time. Why won't end-to-end decode throughput double?

Attention is only part of each step. The linear layers still read the same bf16 weights. The bigger win comes indirectly: more sequences fit in memory, so the batch is larger, and that is what raises linear-layer efficiency.

#### Done when

- The three-configuration table is filled in, with quality and system metrics on the same traces.
- You can explain each result in one sentence, including any that surprised you.

### Reading

- Liu et al., “KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache” (2024). The per-channel-key, per-token-value result and the residual full-precision window.
- Hooper et al., “KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization” (2024). A deeper analysis, including pre- versus post-RoPE quantization of keys, which is a good extra experiment.
- Dettmers et al., “LLM.int8()” (2022). The original account of outlier features in transformer activations; read for intuition.
- Gholami et al., “A Survey of Quantization Methods for Efficient Neural Network Inference” (2021), sections 2–3. Clean definitions of symmetric, asymmetric, and granularity choices.

#### Alternative contribution: prefix caching

If you would rather work on scheduling than kernels: hash each full block by its token contents plus its predecessor's hash, reuse blocks with matching hashes across requests (reference counts from Phase 2 now matter), and evaluate on multi-turn or agent-style traces with shared system prompts. Read the SGLang paper's RadixAttention section (Zheng et al., 2024) and vLLM's automatic prefix caching design docs.

## Phase 6 · week 10Benchmark against vLLM and write it up

Goal: an honest comparison and a write-up that reads like engineering, not marketing.

### What to learn

#### 1. Running a fair comparison

- **Same everything:** model, dtype, maximum context length, trace file, and sampling settings (greedy, fixed output lengths via ignoring end-of-sequence).
- **Same memory budget:** set vLLM's `gpu_memory_utilization` so its KV cache is close to yours in size, and report both numbers.
- **Separate engine design from launch overhead:** run vLLM both with CUDA graphs and in eager mode (`enforce_eager`). Comparing your eager engine against vLLM's eager mode isolates scheduling and kernel quality.
- **Laptop hygiene:** warmup runs, at least three repetitions reported as mean and spread, and a throttle-reason check after each run.

#### 2. Reporting what you found

You will not beat vLLM overall, and claiming to would cost you credibility with anyone who knows the field. Report where you are (“reaches X% of vLLM eager-mode throughput at 4 requests/s”) and where your contribution changes the picture (“INT8 KV fits 1.9× more concurrent sequences at a Y% perplexity increase”). Limitations stated plainly read as strength.

### What to build

- A single script that reproduces every plot in the write-up from raw benchmark logs.
- A README covering the problem, architecture diagram, key design decisions (from your design log), results, and limitations.
- Optionally, a blog post walking through one of the worked examples here with your own measurements. That is very effective as an interview conversation starter.

### Reading

- Gernot Heiser, “Systems Benchmarking Crimes”. A short checklist of the ways benchmark results mislead. Go through it before publishing your numbers.
- vLLM documentation for engine arguments and the serving benchmark. To configure the comparison correctly.

Numbers in worked examples use Qwen2.5 configurations and an assumed \~192 GB/s bandwidth. Replace the assumptions with your Phase 0 measurements as soon as you have them.