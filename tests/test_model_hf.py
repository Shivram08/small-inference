"""Phase 1 gates against HF on Qwen2.5-0.5B-Instruct: loader, RMSNorm, RoPE, layer 0, logits (invariant 1), and
greedy generation cached == uncached == HF (invariant 2).

With fused_matmuls off, the fp32 forward pass is bit-identical to HF eager, which pins every op (an eps or RoPE slip
would show up even below 1e-4). With fusion on (the engine default), cuBLAS reduction order differs by ~1 ulp per
fused matmul, which grows to about 2.4e-4 at logits of magnitude 30 (8e-6 relative); that is the ~1e-4 gate.

Memory plan (6 GB GPU, about 7.4 GiB WSL RAM): the `ref` fixture loads HF in fp32 (2 GB), records every reference
output, frees it, then does the same in bf16. Only then are this engine's models loaded (fp32 2 GB + bf16 1 GB).
About a minute in total, a few GB of GPU memory, one CPU core.
"""

import json

import pytest
import torch

from engine.generate import greedy_generate
from engine.loader import DEFAULT_MODEL
from engine.model import Qwen2Model, Rope, rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
TRACE = "traces/sharegpt_n100_seed0_burst.jsonl"
GEN_TOKENS = 50


def free_gpu():
    """Release cached allocator blocks after dropping a model."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def prompts():
    """A chat-templated question (36 tokens), the first three 200-400-token ShareGPT trace prompts, and 1 token."""
    from bench.trace import load_tokenizer
    tok = load_tokenizer()
    chat = tok(tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}],
                                       tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    with open(TRACE) as f:
        rows = [json.loads(line) for line in f]
    longs = [r["prompt_ids"] for r in rows if 200 <= len(r["prompt_ids"]) <= 400][:3]
    return {"chat": chat, "long": longs[0], "one": chat[:1], "long2": longs[1], "long3": longs[2]}


EXACT = ("chat", "long", "one")  # full fp32 logits kept for these; the bf16 pool also uses long2 and long3


def hf_greedy(model, prompt_ids, n):
    """n argmax tokens from HF with its own KV cache (manual loop, EOS ignored)."""
    ids = torch.tensor([prompt_ids], device="cuda")
    out = model(ids, use_cache=True)
    toks = []
    for _ in range(n):
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt.item())
        out = model(nxt, past_key_values=out.past_key_values, use_cache=True)
    return toks


@pytest.fixture(scope="module")
def ref(prompts):
    """Every HF reference output, computed with one HF model on the GPU at a time and none left afterwards."""
    from transformers import AutoModelForCausalLM
    try:
        hf = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float32, attn_implementation="eager",
                                                  local_files_only=True).cuda().eval()
    except OSError:
        pytest.skip("Qwen2.5-0.5B-Instruct not in the local HF cache")
    r = {"config": hf.config,
         "param_fingerprints": {n: (tuple(p.shape), p.double().sum().item(), p.double().abs().sum().item())
                                for n, p in hf.named_parameters()},
         "norm0_weight": hf.model.layers[0].input_layernorm.weight.detach().clone()}
    captured = {}
    hook = hf.model.layers[0].register_forward_hook(
        lambda mod, args, out: captured.__setitem__("layer0", (out[0] if isinstance(out, tuple) else out)[0]))
    with torch.inference_mode():
        r["logits32"] = {k: hf(torch.tensor([prompts[k]], device="cuda")).logits[0].float() for k in EXACT}
        r["argmax32"] = {k: (r["logits32"][k] if k in EXACT else hf(torch.tensor([p], device="cuda")).logits[0])
                         .argmax(-1) for k, p in prompts.items()}
        hf(torch.tensor([prompts["chat"]], device="cuda"))  # last call wins: layer 0 output for the chat prompt
        r["layer0"] = captured["layer0"].clone()
        hook.remove()
        r["greedy32"] = {k: hf_greedy(hf, prompts[k], GEN_TOKENS) for k in ("chat", "one")}
    del hf
    free_gpu()

    for attn in ("eager", "sdpa"):  # eager rounds attention probs to bf16 like this engine; SDPA keeps them fp32
        hf = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.bfloat16, attn_implementation=attn,
                                                  local_files_only=True).cuda().eval()
        with torch.inference_mode():
            r[f"argmax16_{attn}"] = {k: hf(torch.tensor([p], device="cuda")).logits[0].argmax(-1)
                                     for k, p in prompts.items()}
        del hf
        free_gpu()
    return r


class unfused:
    """Context manager: run a model with separate Q/K/V and gate/up matmuls, then restore fusion."""

    def __init__(self, model):
        """Remember the model."""
        self.model = model

    def __enter__(self):
        """Turn fusion off."""
        self.model.fused_matmuls = False

    def __exit__(self, *exc):
        """Turn fusion back on."""
        self.model.fused_matmuls = True


@pytest.fixture(scope="module")
def mine32(ref):
    """This engine's model in fp32, loaded after the HF references are done."""
    return Qwen2Model.from_pretrained(dtype=torch.float32)


@pytest.fixture(scope="module")
def mine16(ref):
    """This engine's model in bf16."""
    return Qwen2Model.from_pretrained(dtype=torch.bfloat16)


def test_config_and_weights_match_checkpoint(ref):
    """Config parsing gives the documented shapes, and the strict loader returns exactly HF's tensors (fp32)."""
    from engine.config import ModelConfig
    from engine.loader import load_weights, snapshot_dir
    d = snapshot_dir()
    cfg = ModelConfig.from_json(d / "config.json")
    assert (cfg.num_layers, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size) == (24, 14, 2, 64, 896)
    assert cfg.group_size == 7 and cfg.tie_word_embeddings and cfg.kv_bytes_per_token(2) == 12_288
    w = load_weights(d, cfg, "cuda", torch.float32)
    assert w["lm_head.weight"] is w["model.embed_tokens.weight"]  # tied, no copy
    for name, (shape, s, a) in ref["param_fingerprints"].items():
        t = w[name].double()
        assert tuple(t.shape) == shape, name
        assert t.sum().item() == pytest.approx(s, rel=1e-12, abs=1e-9), name
        assert t.abs().sum().item() == pytest.approx(a, rel=1e-12), name
    assert len(ref["param_fingerprints"]) == len(w) - 1  # HF lists the tied lm_head once, under embed_tokens
    del w
    free_gpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_matches_hf(ref, dtype):
    """Bit-identical to Qwen2RMSNorm, including the cast to the input dtype before the weight multiply."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    hf_norm = Qwen2RMSNorm(896, eps=1e-6).cuda().to(dtype)
    with torch.no_grad():
        hf_norm.weight.copy_(ref["norm0_weight"])
    x = torch.randn(37, 896, device="cuda", dtype=dtype) * 3
    torch.testing.assert_close(rms_norm(x, hf_norm.weight.detach(), 1e-6), hf_norm(x).detach(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_matches_hf(ref, dtype):
    """Bit-identical to Qwen2RotaryEmbedding + apply_rotary_pos_emb, at positions up to 32,767."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb
    from engine.config import ModelConfig
    from engine.loader import snapshot_dir
    cfg = ModelConfig.from_json(snapshot_dir() / "config.json")
    positions = torch.tensor([0, 1, 2, 3, 100, 1023, 1024, 4095, 20_000, 32_767], device="cuda")
    T = positions.numel()
    q = torch.randn(T, 14, 64, device="cuda", dtype=dtype)
    k = torch.randn(T, 2, 64, device="cuda", dtype=dtype)
    cos, sin = Qwen2RotaryEmbedding(ref["config"]).cuda()(q, positions[None])
    hq, hk = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
    rope = Rope(cfg, "cuda")
    torch.testing.assert_close(rope(q, positions), hq[0].transpose(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(rope(k, positions), hk[0].transpose(0, 1), rtol=0, atol=0)


def test_layer0_matches_hf(ref, mine32, prompts):
    """Decoder layer 0 on the chat prompt's embeddings: bit-identical to HF unfused, within ~1 ulp-scale fused."""
    ids = torch.tensor(prompts["chat"], device="cuda")
    x, pos = mine32.embed[ids], torch.arange(len(ids), device="cuda")
    with torch.inference_mode():
        with unfused(mine32):
            exact = mine32.layer_forward(0, x, pos, None)
        fused = mine32.layer_forward(0, x, pos, None)
    diff = (fused - ref["layer0"]).abs().max().item()
    print(f"\nlayer 0 fused max |diff| {diff:.2e} (output max |x| {ref['layer0'].abs().max().item():.1f})")
    torch.testing.assert_close(exact, ref["layer0"], rtol=0, atol=0)
    assert diff < 1e-4


def test_logits_fp32_match_hf(ref, mine32, prompts):
    """Invariant 1: fp32 logits over every position of the chat, long, and 1-token prompts. Unfused: bit-identical to HF. Fused:
    within 5e-4 (measured 2.4e-4) and 2e-5 of the logit range, with the same argmax everywhere."""
    for name in EXACT:
        p = prompts[name]
        ids, pos = torch.tensor(p, device="cuda"), torch.arange(len(p), device="cuda")
        theirs = ref["logits32"][name]
        with unfused(mine32):
            torch.testing.assert_close(mine32.forward(ids, pos), theirs, rtol=0, atol=0, msg=name)
        logits = mine32.forward(ids, pos)
        diff = (logits - theirs).abs().max().item()
        scale = theirs.abs().max().item()
        print(f"\n{name} ({len(p)} tokens): unfused bit-identical; fused max |diff| {diff:.2e} "
              f"({diff / scale:.1e} of max |logit| {scale:.1f})")
        assert diff < 5e-4 and diff / scale < 2e-5, name
        assert torch.equal(logits.argmax(-1), theirs.argmax(-1)), name


def test_bf16_argmax_agrees_with_hf(ref, mine16, prompts):
    """Invariant 1, bf16: next-token argmax agrees with the fp32 reference about as often as HF's own bf16 eager
    model does, pooled over all 817 positions of the five prompts (a standard error of about 0.7 points). Measured: this engine 95.5%, HF bf16 eager
    95.6%, HF bf16 SDPA 97.4% (SDPA keeps attention probabilities in fp32; eager and this engine round them to bf16)."""
    def pooled(argmaxes):
        """Fraction of all positions where an argmax set matches HF fp32."""
        hits = sum((argmaxes[k] == ref["argmax32"][k]).sum().item() for k in prompts)
        return hits / sum(len(p) for p in prompts.values())

    mine = {k: mine16.forward(torch.tensor(p, device="cuda"), torch.arange(len(p), device="cuda")).argmax(-1)
            for k, p in prompts.items()}
    a_mine, a_eager, a_sdpa = pooled(mine), pooled(ref["argmax16_eager"]), pooled(ref["argmax16_sdpa"])
    print(f"\nbf16 argmax agreement with HF fp32: this engine {a_mine:.1%}, HF eager {a_eager:.1%}, "
          f"HF SDPA {a_sdpa:.1%}")
    assert a_mine >= a_eager - 0.02
    assert a_mine >= 0.93


@pytest.mark.parametrize("name", ["chat", "one", "long"])
def test_cached_equals_uncached_fp32(mine32, prompts, name):
    """Invariant 2: 50 greedy tokens are identical with and without the KV cache, including a 1-token prompt."""
    p = prompts[name]
    assert greedy_generate(mine32, p, GEN_TOKENS, use_cache=True) == greedy_generate(mine32, p, GEN_TOKENS,
                                                                                    use_cache=False)


@pytest.mark.parametrize("name", ["chat", "one"])
def test_greedy_matches_hf_fp32(ref, mine32, prompts, name):
    """And both equal HF's own cached greedy decode in fp32."""
    assert greedy_generate(mine32, prompts[name], GEN_TOKENS) == ref["greedy32"][name]


def test_chunked_prefill_matches_one_shot_fp32(mine32, prompts):
    """Explicit positions: prefilling the long prompt in 64-token chunks through the cache matches one forward
    (measured 1.1e-4 fused)."""
    from engine.kv_cache import ContiguousKVCache
    p = torch.tensor(prompts["long"], device="cuda")
    full = mine32.forward(p, torch.arange(len(p), device="cuda"))
    cache = ContiguousKVCache(mine32.cfg, len(p), "cuda", torch.float32)
    parts = [mine32.forward(p[s:s + 64], cache.next_positions(len(p[s:s + 64])), cache)
             for s in range(0, len(p), 64)]
    diff = (torch.cat(parts) - full).abs().max().item()
    print(f"\nchunked vs one-shot max |diff| {diff:.2e}")
    assert diff < 5e-4  # different M per matmul, so different cuBLAS kernels: same reduction-order effect as fusion
