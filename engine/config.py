"""Model hyperparameters from a Hugging Face config.json, restricted to what the Qwen2 forward pass supports."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int  # 151,936
    hidden_size: int  # 896 (0.5B), 1536 (1.5B)
    intermediate_size: int  # 4864 (0.5B), 8960 (1.5B)
    num_layers: int  # 24, 28
    num_q_heads: int  # 14, 12
    num_kv_heads: int  # 2, 2
    head_dim: int  # 64, 128
    rms_norm_eps: float  # 1e-6
    rope_theta: float  # 1,000,000
    max_position_embeddings: int  # 32,768
    tie_word_embeddings: bool  # True for 0.5B and 1.5B: lm_head reuses embed_tokens

    @property
    def group_size(self) -> int:
        """GQA group g: query heads per KV head (7 for 0.5B, 6 for 1.5B)."""
        return self.num_q_heads // self.num_kv_heads

    @property
    def q_size(self) -> int:
        """Width of the query projection output (896 for 0.5B)."""
        return self.num_q_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        """Width of each of the K and V projection outputs (128 for 0.5B)."""
        return self.num_kv_heads * self.head_dim

    def kv_bytes_per_token(self, bytes_per_elem: int = 2) -> int:
        """K and V for one token across all layers: 12,288 B for 0.5B in bf16."""
        return 2 * self.num_layers * self.kv_size * bytes_per_elem

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        """Parse config.json, refusing features this forward pass does not implement."""
        c = json.loads(Path(path).read_text())
        if c.get("model_type") != "qwen2":
            raise ValueError(f"expected model_type qwen2, got {c.get('model_type')}")
        if c.get("hidden_act", "silu") != "silu":
            raise ValueError(f"unsupported hidden_act {c['hidden_act']}")
        if c.get("use_sliding_window", False):
            raise ValueError("sliding-window attention is not implemented")
        if c.get("rope_scaling") not in (None, {}):
            raise ValueError(f"RoPE scaling is not implemented: {c['rope_scaling']}")
        head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        cfg = cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_q_heads=c["num_attention_heads"],
            num_kv_heads=c["num_key_value_heads"],
            head_dim=head_dim,
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
        )
        if cfg.num_q_heads % cfg.num_kv_heads:
            raise ValueError("query heads must be a multiple of KV heads")
        return cfg
