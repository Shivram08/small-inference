"""Phase 5 evaluation: the three-config table (bf16, INT8 per-token K/V, INT8 per-channel K + per-token V).

Quality: documents (ShareGPT trace prompts of 256-1024 tokens) are prefilled in 64-token chunks through each cache
format in lockstep, so every chunk after the first attends to quantized K/V (single-shot prefill would hide the
error). Per format: perplexity of each next token, mean KL divergence from the bf16 cache's distribution, and top-1
agreement with it. Then, on the bf16 cache's own K/V: each scheme's pure quantization error per layer (relative RMS),
and the K channel outlier ratio (largest channel absmax / median channel absmax) that motivates per-channel K.

Capacity: bytes per token and how many tokens fit in a KV budget, per format.

Speed (--timing): the Triton decode kernel's time per layer for each format at a few (batch, context) points; this
part needs the laptop in Best performance mode and idle.

Writes results/kv_quant/<stamp>_<model>/{results.json, table.md, env.json}. Quality on 0.5B with 20 documents takes
about a minute and ~2 GB of GPU memory.
"""

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import torch

from engine.block_manager import BlockManager
from engine.paged_cache import PagedKVCache, build_metadata
from engine.quant import dequantize, quantize_per_channel, quantize_per_token
from engine.quant_cache import FORMATS, QuantPagedKVCache, blocks_for_budget, kv_bytes_per_token

CHUNK = 64


def _per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token quantization of [H, S, d] with the scale broadcast back to [H, S, 1]."""
    q, s = quantize_per_token(x)
    return q, s[..., None]


def documents(trace: Path, n: int, lo: int = 256, hi: int = 1024) -> list[list[int]]:
    """The first n trace prompts with lo..hi tokens."""
    with open(trace) as f:
        docs = [r["prompt_ids"] for r in map(json.loads, f) if lo <= len(r["prompt_ids"]) <= hi]
    return docs[:n]


def make_cache(cfg, fmt: str, num_blocks: int, manager: BlockManager, device="cuda", dtype=torch.bfloat16):
    """A cache of `fmt` over `manager`, Triton backend (prefill rows use the gather reference either way)."""
    if fmt == "bf16":
        return PagedKVCache(cfg, num_blocks, 16, device, dtype, "triton")
    return QuantPagedKVCache(cfg, num_blocks, 16, device, dtype, "triton", fmt, manager, 8)


@torch.inference_mode()
def quality(model, docs: list[list[int]]) -> dict:
    """Perplexity, KL from bf16, top-1 agreement, reconstruction error, and K outlier ratios per format."""
    nll = {f: 0.0 for f in FORMATS}
    kl = {f: 0.0 for f in FORMATS}
    agree = {f: 0 for f in FORMATS}
    n_pred = 0
    L = model.cfg.num_layers
    recon = {f: {"k": [0.0] * L, "v": [0.0] * L, "norm_k": [0.0] * L, "norm_v": [0.0] * L} for f in FORMATS[1:]}
    outlier = [[] for _ in range(L)]
    for doc in docs:
        nb = -(-len(doc) // 16)
        managers = {f: BlockManager(nb, 16) for f in FORMATS}
        caches = {f: make_cache(model.cfg, f, nb, managers[f], model.device, model.dtype) for f in FORMATS}
        target = torch.tensor(doc[1:], device=model.device)
        for f in FORMATS:
            managers[f].allocate(0, 0)
        for s in range(0, len(doc), CHUNK):
            chunk = doc[s:s + CHUNK]
            ids = torch.tensor(chunk, device=model.device)
            logp = {}
            for f in FORMATS:
                managers[f].append_slots(0, len(chunk))
                meta = build_metadata(managers[f], [(0, s, len(chunk))], model.device)
                logp[f] = model.forward(ids, meta.positions, caches[f], meta=meta).float().log_softmax(-1)
            rows = min(len(chunk), len(doc) - 1 - s)  # the last token has no next token
            if rows <= 0:
                continue
            tgt = target[s:s + rows]
            ref = logp["bf16"][:rows]
            for f in FORMATS:
                lp = logp[f][:rows]
                nll[f] += -lp.gather(1, tgt[:, None]).sum().item()
                kl[f] += (ref.exp() * (ref - lp)).sum().item()
                agree[f] += (lp.argmax(-1) == ref.argmax(-1)).sum().item()
            n_pred += rows
        # Pure quantization error: quantize the bf16 cache's own K/V with each scheme (no upstream drift: comparing
        # against an INT8 cache's K/V would also count hidden-state changes from earlier layers' quantization).
        caches["bf16"].meta = build_metadata(managers["bf16"], [(0, len(doc) - 1, 1)], model.device)
        n_full = (len(doc) // 16) * 16
        for layer in range(L):
            k_ref, v_ref = (t.float() for t in caches["bf16"].gather(layer, 0))  # [H, S, d]
            ch = k_ref.abs().amax(dim=1)  # [H, d]: each channel's absmax over the document
            outlier[layer].append((ch.amax(-1) / ch.median(-1).values).mean().item())
            k_tok = dequantize(*_per_token(k_ref))
            v_tok = dequantize(*_per_token(v_ref))
            H, _, d = k_ref.shape
            blocks = k_ref[:, :n_full].reshape(H, -1, 16, d).bfloat16()
            kq, ks = quantize_per_channel(blocks, dim=2)
            k_ch = torch.cat([dequantize(kq, ks[:, :, None]).reshape(H, -1, d),
                              k_ref[:, n_full:].bfloat16().float()], dim=1)  # partial block: bf16 staging
            for f, k_hat in (("int8", k_tok), ("int8-kchannel", k_ch)):
                recon[f]["k"][layer] += (k_hat - k_ref).pow(2).sum().item()
                recon[f]["v"][layer] += (v_tok - v_ref).pow(2).sum().item()
                recon[f]["norm_k"][layer] += k_ref.pow(2).sum().item()
                recon[f]["norm_v"][layer] += v_ref.pow(2).sum().item()
        del caches
        torch.cuda.empty_cache()
    out = {"n_docs": len(docs), "n_predictions": n_pred, "formats": {}}
    for f in FORMATS:
        r = {"perplexity": math.exp(nll[f] / n_pred), "mean_kl_vs_bf16_nats": kl[f] / n_pred,
             "top1_agreement_vs_bf16": agree[f] / n_pred}
        if f != "bf16":
            k_err = [math.sqrt(recon[f]["k"][i] / recon[f]["norm_k"][i]) for i in range(L)]
            v_err = [math.sqrt(recon[f]["v"][i] / recon[f]["norm_v"][i]) for i in range(L)]
            r |= {"k_rel_error_by_layer": k_err, "v_rel_error_by_layer": v_err,
                  "k_rel_error_mean": statistics.fmean(k_err), "v_rel_error_mean": statistics.fmean(v_err)}
        out["formats"][f] = r
    out["k_outlier_ratio_by_layer"] = [statistics.fmean(x) for x in outlier]
    return out


def capacity(cfg, budget_bytes: int) -> dict:
    """Bytes per token and tokens per budget for each format (16-token blocks; format B with 128 staging slots)."""
    return {f: {"bytes_per_token": kv_bytes_per_token(cfg, f),
                "blocks": blocks_for_budget(cfg, f, budget_bytes, 16, 128 if f == "int8-kchannel" else 0),
                "tokens": 16 * blocks_for_budget(cfg, f, budget_bytes, 16, 128 if f == "int8-kchannel" else 0)}
            for f in FORMATS}


@torch.inference_mode()
def timing(cfg, points: list[tuple[int, int]], reps: int = 10, calls: int = 20) -> list[dict]:
    """Decode-kernel µs per layer per format at (batch, context) points, shuffled block tables."""
    from bench.decode_sweep import time_call_us
    from engine.triton_decode import paged_decode_attention
    rows = []
    for batch, S in points:
        nb = batch * -(-S // 16)
        for f in FORMATS:
            m = BlockManager(nb, 16)
            cache = make_cache(cfg, f, nb, m)
            for s in range(batch):
                m.allocate(s, 0)
                for start in range(0, S, 512):
                    n = min(512, S - start)
                    m.append_slots(s, n)
                    meta = build_metadata(m, [(s, start, n)], "cuda")
                    cache.begin(n, meta)
                    for layer in range(1):
                        k = torch.randn(n, cfg.num_kv_heads, cfg.head_dim, device="cuda", dtype=torch.bfloat16)
                        cache.write(layer, k, torch.randn_like(k))
            meta = build_metadata(m, [(s, S - 1, 1) for s in range(batch)], "cuda")
            cache.begin(batch, meta)
            q = torch.randn(batch, cfg.num_q_heads, cfg.head_dim, device="cuda", dtype=torch.bfloat16)
            us = time_call_us(lambda: paged_decode_attention(q, cache.k[0], cache.v[0], meta.block_tables,
                                                             meta.context_lens_t, cfg.head_dim ** -0.5, S,
                                                             quant=cache._kernel_quant(0)), calls, reps)
            rows.append({"batch": batch, "context_len": S, "format": f, "us_per_layer": us,
                         "bytes_read": batch * S * kv_bytes_per_token(cfg, f) / cfg.num_layers})
            del cache
            torch.cuda.empty_cache()
    return rows


def table_md(res: dict) -> str:
    """The three-config table as Markdown."""
    q, c = res["quality"]["formats"], res["capacity"]
    lines = ["| Format | Bytes/token | Tokens in budget | Perplexity | Mean KL vs bf16 (nats) | Top-1 agree | "
             "K rel. error | V rel. error |", "|---|---|---|---|---|---|---|---|"]
    for f in FORMATS:
        r = q[f]
        lines.append(f"| {f} | {c[f]['bytes_per_token']:,.0f} | {c[f]['tokens']:,} | {r['perplexity']:.3f} | "
                     f"{r['mean_kl_vs_bf16_nats']:.4f} | {r['top1_agreement_vs_bf16']:.1%} | "
                     f"{r.get('k_rel_error_mean', 0):.4f} | {r.get('v_rel_error_mean', 0):.4f} |")
    ratios = res["quality"]["k_outlier_ratio_by_layer"]
    lines += ["", f"K outlier ratio (max channel absmax / median, per layer): mean {statistics.fmean(ratios):.1f}, "
              f"max {max(ratios):.1f} (layer {ratios.index(max(ratios))}), min {min(ratios):.1f}"]
    if "timing" in res:
        lines += ["", "| Batch | Context | " + " | ".join(f"{f} µs/layer" for f in FORMATS) + " |",
                  "|---|---|" + "---|" * len(FORMATS)]
        pts = sorted({(r["batch"], r["context_len"]) for r in res["timing"]})
        for b, S in pts:
            t = {r["format"]: r["us_per_layer"] for r in res["timing"] if (r["batch"], r["context_len"]) == (b, S)}
            lines.append(f"| {b} | {S} | " + " | ".join(f"{t[f]:.1f}" for f in FORMATS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run quality (and optionally timing) for one model and write the table."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--docs", type=int, default=20)
    p.add_argument("--trace", type=Path, default=Path("traces/sharegpt_n100_seed0_burst.jsonl"))
    p.add_argument("--budget-gib", type=float, default=2.0)
    p.add_argument("--timing", action="store_true", help="also time the decode kernel (needs an idle laptop)")
    p.add_argument("--results-root", type=Path, default=Path("results/kv_quant"))
    args = p.parse_args()

    from bench.env import gpu_snapshot, static_env
    from engine.model import Qwen2Model
    model = Qwen2Model.from_pretrained(args.model, dtype=torch.bfloat16)
    out_dir = args.results_root / f"{datetime.now():%Y%m%d-%H%M%S}_{args.model.split('/')[-1]}"
    out_dir.mkdir(parents=True, exist_ok=False)
    env = {"static": static_env(args.trace), "gpu_before": gpu_snapshot()}
    res = {"model": args.model, "chunk": CHUNK, "budget_gib": args.budget_gib,
           "quality": quality(model, documents(args.trace, args.docs)),
           "capacity": capacity(model.cfg, int(args.budget_gib * 2**30))}
    if args.timing:
        res["timing"] = timing(model.cfg, [(1, 1024), (1, 4096), (8, 1024), (8, 4096), (32, 1024), (32, 4096)])
    env["gpu_after"] = gpu_snapshot()
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    (out_dir / "table.md").write_text(table_md(res))
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(table_md(res))
    print(out_dir)


if __name__ == "__main__":
    main()
