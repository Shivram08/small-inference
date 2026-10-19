"""Contiguous KV cache for one sequence: per layer, K and V as [H_kv, T_max, d], keys stored after RoPE.

Sizes for Qwen2.5-0.5B in bf16: one token is 2 (K, V) x 24 layers x 2 heads x 64 x 2 B = 12,288 B, so a 2,048-token
cache is 24 MiB. The [H_kv, T, d] layout makes read() a zero-copy slice that is already the [Hkv, S, d] shape
attention wants. Phase 2 replaces this with fixed-size blocks.

Bookkeeping lives on the CPU, so nothing here syncs with the GPU. New tokens always extend the sequence: callers get
their positions from next_positions(n) (a view of a device-side arange, so no host-to-device copy either), and the
forward pass calls begin(n) once before the layers write at [start, start + n).

Interface shared with the paged cache (engine/paged_cache.py), which is all the model uses: begin(n, meta) once per
forward, then per layer write(layer, k, v) and attend(layer, q, positions, scale).
"""

import torch

from engine.attention import attention
from engine.config import ModelConfig


class ContiguousKVCache:
    def __init__(self, cfg: ModelConfig, max_len: int, device: str | torch.device, dtype: torch.dtype):
        """Preallocate K and V of shape [layers, H_kv, max_len, d]. Contents are garbage until written."""
        shape = (cfg.num_layers, cfg.num_kv_heads, max_len, cfg.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.length = 0  # positions [0, length) hold valid K/V once the current forward's writes are done
        self.start = 0  # where the current forward's tokens are written
        self._arange = torch.arange(max_len, device=device)

    def reset(self) -> None:
        """Forget the sequence; the memory is reused as is."""
        self.length = self.start = 0

    def next_positions(self, n: int) -> torch.Tensor:
        """Positions [n] for the next n tokens: length, ..., length + n - 1."""
        if self.length + n > self.max_len:
            raise ValueError(f"{n} more tokens exceed cache capacity {self.max_len} (length {self.length})")
        return self._arange[self.length:self.length + n]

    def advance(self, n: int) -> None:
        """Start a forward over n new tokens: they will be written at [length, length + n)."""
        if self.length + n > self.max_len:
            raise ValueError(f"{n} more tokens exceed cache capacity {self.max_len} (length {self.length})")
        self.start, self.length = self.length, self.length + n

    def begin(self, n_tokens: int, meta: object = None) -> None:
        """Start a forward over n_tokens new tokens (no metadata needed: they extend the one sequence)."""
        self.advance(n_tokens)

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store this forward's k, v [n, H_kv, d] for one layer."""
        end = self.start + k.shape[0]
        self.k[layer, :, self.start:end] = k.transpose(0, 1)
        self.v[layer, :, self.start:end] = v.transpose(0, 1)

    def read(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K, V [H_kv, length, d] for one layer: views, no copy."""
        return self.k[layer, :, :self.length], self.v[layer, :, :self.length]

    def positions(self) -> torch.Tensor:
        """Absolute positions of the cached keys, [length]."""
        return self._arange[:self.length]

    def attend(self, layer: int, q: torch.Tensor, positions: torch.Tensor, scale: float) -> torch.Tensor:
        """Attention of this forward's queries q [T, Hq, d] over every cached position. Returns [T, Hq * d]."""
        k, v = self.read(layer)
        return attention(q, k, v, positions, self.positions(), scale)
