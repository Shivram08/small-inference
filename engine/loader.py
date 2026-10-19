"""Load Qwen2 weights from the local HF cache: safetensors read one tensor at a time, straight to the device.

Each tensor is read from the memory-mapped file, copied to the device in its stored dtype (bf16), and only then cast,
so CPU RAM holds at most one tensor at a time. The largest is embed_tokens, 151,936 x 896 bf16 = 272 MB for 0.5B.
WSL has about 7.4 GiB, which matters for the 1.5B model (3.1 GB of weights).
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from engine.config import ModelConfig

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def snapshot_dir(model_id: str = DEFAULT_MODEL) -> Path:
    """Local snapshot folder of an already-downloaded model (never downloads)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, local_files_only=True))


def expected_names(cfg: ModelConfig) -> set[str]:
    """Every tensor name a Qwen2ForCausalLM checkpoint must contain (lm_head only when not tied)."""
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        names |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight", p + "self_attn.o_proj.weight",
                  p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight"}
        names |= {p + f"self_attn.{proj}.{kind}" for proj in ("q_proj", "k_proj", "v_proj")
                  for kind in ("weight", "bias")}
    if not cfg.tie_word_embeddings:
        names.add("lm_head.weight")
    return names


def safetensor_files(model_dir: Path) -> list[Path]:
    """The checkpoint's .safetensors files, from the shard index if there is one."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [model_dir / s for s in shards]
    return [model_dir / "model.safetensors"]


def load_weights(model_dir: Path, cfg: ModelConfig, device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """All checkpoint tensors by HF name, on `device` in `dtype`. Strict: a missing or unexpected name raises.

    With tied embeddings, "lm_head.weight" is added as the same tensor as the embedding table (no copy).
    """
    want = expected_names(cfg)
    weights: dict[str, torch.Tensor] = {}
    for path in safetensor_files(model_dir):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for name in f.keys():
                if cfg.tie_word_embeddings and name == "lm_head.weight":
                    continue  # some tied checkpoints store a duplicate; the embedding is authoritative
                if name not in want:
                    raise KeyError(f"unexpected tensor {name} in {path.name}")
                weights[name] = f.get_tensor(name).to(device).to(dtype)
    missing = want - weights.keys()
    if missing:
        raise KeyError(f"missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}")
    if cfg.tie_word_embeddings:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    return weights
