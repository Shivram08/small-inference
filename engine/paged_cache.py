"""Paged KV cache: a pool of fixed-size blocks, per-forward metadata, and gather-based reference attention.

Pool layout per layer: K and V as [H_kv, num_blocks, block_size, d], i.e. [2, N, 16, 64] for 0.5B. One (head, block)
chunk is 16 x 64 x 2 B = 2 KiB contiguous, which is what a decode kernel program reads per step. A token's slot
(block * 16 + offset) addresses position [:, block, offset] in every layer.

Reference attention (Phase 2): for each sequence, gather its blocks, k[layer][:, block_ids] -> [2, nb, 16, 64], which
reshapes for free to [2, nb * 16, 64]; slice to the context length and call the same attention() as the contiguous
cache. Same values, same kernels, so outputs match Phase 1 bit for bit. The gather is a full copy of the sequence's
K and V per layer per step; the Phase 3 Triton kernel reads the blocks in place instead.

Several sequences can share one forward: their tokens are packed flat ([T_total, ...]), and PagedMetadata says which
rows belong to which sequence. The linear layers then run once over all of them.

attn_backend="triton" (Phase 3) sends every decoding sequence (one new token) through the Triton kernel in a single
launch per layer, reading blocks in place; sequences with several new tokens (prefill, chunks) still use the gather
reference.
"""

from dataclasses import dataclass

import torch

from engine.attention import attention
from engine.block_manager import BlockManager
from engine.config import ModelConfig

BACKENDS = ("gather", "triton")


@dataclass
class PagedMetadata:
    """Everything a forward needs to read and write the paged cache. CPU lists for Python-side loops, device tensors
    for indexing; built once per forward and shared by all layers."""
    num_tokens: int
    query_starts: list[int]  # [num_seqs + 1]: sequence i's new tokens are rows query_starts[i]:query_starts[i+1]
    context_lens: list[int]  # [num_seqs]: tokens in the cache after this forward's writes
    positions: torch.Tensor  # [T] absolute position of each new token
    slot_mapping: torch.Tensor  # [T] int64 slot of each new token
    slot_blocks: torch.Tensor  # [T] slot // block_size
    slot_offsets: torch.Tensor  # [T] slot % block_size
    block_ids: list[torch.Tensor]  # per sequence: its first ceil(context_len / B) blocks, int64 on device
    block_tables: torch.Tensor  # [num_seqs, max_blocks] int32, padded with -1 (the kernel-facing form)
    context_lens_t: torch.Tensor  # [num_seqs] int32 on device
    last_token_indices: torch.Tensor  # [num_seqs] row of each sequence's last new token

    @property
    def num_seqs(self) -> int:
        """Sequences in this forward."""
        return len(self.context_lens)


def build_metadata(manager: BlockManager, batch: list[tuple[int, int, int]],
                   device: str | torch.device) -> PagedMetadata:
    """Metadata for a forward over `batch` = [(seq_id, start, n), ...]: n new tokens per sequence at positions
    start .. start + n - 1. The manager must already hold their slots (allocate / append_slots); afterwards the
    sequence's context is start + n tokens."""
    B = manager.block_size
    positions, slots, query_starts, context_lens, tables = [], [], [0], [], []
    for seq_id, start, n in batch:
        if n < 1:
            raise ValueError(f"sequence {seq_id}: a forward needs at least one new token")
        ctx = start + n
        if manager.seq_len(seq_id) < ctx:
            raise ValueError(f"sequence {seq_id} holds {manager.seq_len(seq_id)} tokens, forward needs {ctx}")
        positions += range(start, ctx)
        slots += manager.slots(seq_id, start, ctx)
        query_starts.append(query_starts[-1] + n)
        context_lens.append(ctx)
        tables.append(manager.block_table(seq_id)[:manager.blocks_for(ctx)])
    width = max(len(t) for t in tables)
    padded = [t + [-1] * (width - len(t)) for t in tables]
    as_long = lambda xs: torch.tensor(xs, dtype=torch.int64, device=device)  # noqa: E731
    return PagedMetadata(
        num_tokens=len(positions),
        query_starts=query_starts,
        context_lens=context_lens,
        positions=as_long(positions),
        slot_mapping=as_long(slots),
        slot_blocks=as_long([s // B for s in slots]),
        slot_offsets=as_long([s % B for s in slots]),
        block_ids=[as_long(t) for t in tables],
        block_tables=torch.tensor(padded, dtype=torch.int32, device=device),
        context_lens_t=torch.tensor(context_lens, dtype=torch.int32, device=device),
        last_token_indices=as_long([e - 1 for e in query_starts[1:]]),
    )


class PagedKVCache:
    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, device: str | torch.device,
                 dtype: torch.dtype, attn_backend: str = "gather"):
        """Allocate the pool: K and V each [layers, H_kv, num_blocks, block_size, d]. Contents are garbage until
        written."""
        if attn_backend not in BACKENDS:
            raise ValueError(f"attn_backend must be one of {BACKENDS}")
        self.attn_backend = attn_backend
        shape = (cfg.num_layers, cfg.num_kv_heads, num_blocks, block_size, cfg.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.cfg, self.num_blocks, self.block_size = cfg, num_blocks, block_size
        self.meta: PagedMetadata | None = None
        self._arange = torch.arange(num_blocks * block_size, device=device)

    @staticmethod
    def blocks_for_budget(cfg: ModelConfig, budget_bytes: int, block_size: int, bytes_per_elem: int = 2) -> int:
        """How many blocks fit in budget_bytes: 2 GiB -> 10,922 blocks of 16 tokens for 0.5B in bf16."""
        return budget_bytes // (block_size * cfg.kv_bytes_per_token(bytes_per_elem))

    def begin(self, n_tokens: int, meta: PagedMetadata | None) -> None:
        """Start a forward over n_tokens new tokens described by meta."""
        if meta is None or meta.num_tokens != n_tokens:
            raise ValueError("a paged forward needs metadata for exactly its tokens")
        self.meta = meta
        self._decode = None
        if self.attn_backend == "triton":
            dec = [i for i in range(meta.num_seqs) if meta.query_starts[i + 1] - meta.query_starts[i] == 1]
            self._prefill = [i for i in range(meta.num_seqs) if meta.query_starts[i + 1] - meta.query_starts[i] > 1]
            if dec:
                # Once per forward, shared by all layers: the decode rows and their slice of the metadata.
                if len(dec) == meta.num_seqs:
                    rows, bt, ctx = None, meta.block_tables, meta.context_lens_t
                else:
                    idx = torch.tensor(dec, device=meta.positions.device)
                    rows, bt, ctx = meta.last_token_indices[idx], meta.block_tables[idx], meta.context_lens_t[idx]
                self._decode = (rows, bt, ctx, max(meta.context_lens[i] for i in dec))

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter this forward's k, v [T, H_kv, d] into their slots."""
        m = self.meta
        self.k[layer][:, m.slot_blocks, m.slot_offsets] = k.transpose(0, 1)
        self.v[layer][:, m.slot_blocks, m.slot_offsets] = v.transpose(0, 1)

    def gather(self, layer: int, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequence seq's K, V as contiguous [H_kv, context_len, d] copies."""
        m, (Hkv, d) = self.meta, (self.cfg.num_kv_heads, self.cfg.head_dim)
        ctx, blocks = m.context_lens[seq], m.block_ids[seq]
        k = self.k[layer][:, blocks].view(Hkv, -1, d)[:, :ctx]
        v = self.v[layer][:, blocks].view(Hkv, -1, d)[:, :ctx]
        return k, v

    def _attend_gather(self, layer: int, i: int, q: torch.Tensor, positions: torch.Tensor,
                       scale: float) -> torch.Tensor:
        """Reference attention for sequence i's rows of q, over its gathered context."""
        m = self.meta
        a, b = m.query_starts[i], m.query_starts[i + 1]
        k, v = self.gather(layer, i)
        return attention(q[a:b], k, v, positions[a:b], self._arange[:m.context_lens[i]], scale)

    def attend(self, layer: int, q: torch.Tensor, positions: torch.Tensor, scale: float) -> torch.Tensor:
        """Attention for every sequence in the forward over its own context. Returns [T, Hq * d]."""
        m = self.meta
        if self._decode is None:
            outs = [self._attend_gather(layer, i, q, positions, scale) for i in range(m.num_seqs)]
            return outs[0] if len(outs) == 1 else torch.cat(outs)
        from engine.triton_decode import paged_decode_attention

        rows, bt, ctx, max_ctx = self._decode
        T, Hq, d = q.shape
        q_dec = q if rows is None else q[rows]
        dec = paged_decode_attention(q_dec, self.k[layer], self.v[layer], bt, ctx, scale, max_ctx)
        if rows is None:
            return dec.view(T, Hq * d)
        out = torch.empty(T, Hq * d, device=q.device, dtype=q.dtype)
        out[rows] = dec.view(-1, Hq * d)
        for i in self._prefill:
            out[m.query_starts[i]:m.query_starts[i + 1]] = self._attend_gather(layer, i, q, positions, scale)
        return out
