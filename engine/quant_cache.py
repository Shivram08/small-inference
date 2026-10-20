"""INT8 paged KV caches (Phase 5): the same pool and interface as PagedKVCache, about half the bytes.

Format "int8" (A, per token): K and V int8 [layers, H_kv, N, B, d] with one fp16 scale per (token, head) each,
[layers, H_kv, N, B]. Written on arrival; 6,336 B/token for 0.5B instead of 12,288.

Format "int8-kchannel" (B): V as in A. K int8 with one fp16 scale per (block, head, channel), [layers, H_kv, N, d],
computed over the block's 16 tokens. A block's channel scales are only known once all 16 tokens exist, so each
sequence's partial last block of K lives in a bf16 staging slot ([layers, H_kv, n_stage, B, d]); the forward that
writes a block's 16th token quantizes the whole block from staging into the pool and frees the slot. A block that
one forward writes whole (a prefill chunk's interior blocks) is quantized straight from that forward's K and never
staged, so a sequence needs at most 2 slots at once (the partial block its chunk starts in and the one it ends in):
2 x max_num_seqs slots, 12.6 MB for 64 sequences. 6,624 B/token for full blocks.

Decode reads through the Triton kernel with dequantization fused in (KV_MODE 1 / 2). The gather path (prefill rows)
dequantizes into the model dtype and runs the reference attention. Every token, including those written in the same
forward, is attended to through its quantized form, so even one-shot prefill sees quantization error (the Phase 5
evaluation still uses chunked prefill, as the plan's pitfall note asks).
"""

import torch

from engine.block_manager import BlockManager
from engine.config import ModelConfig
from engine.paged_cache import PagedKVCache, PagedMetadata
from engine.quant import dequantize, quantize_per_channel, quantize_per_token
from engine.triton_decode import QuantKV

FORMATS = ("bf16", "int8", "int8-kchannel")


def kv_bytes_per_token(cfg: ModelConfig, fmt: str, block_size: int = 16) -> float:
    """Pool bytes per token: 12,288 / 6,336 / 6,624 for 0.5B (bf16 / int8 / int8-kchannel, 16-token blocks)."""
    per_head_layer = {"bf16": 2 * cfg.head_dim * 2,
                      "int8": 2 * (cfg.head_dim + 2),
                      "int8-kchannel": (cfg.head_dim + 2) + cfg.head_dim + cfg.head_dim * 2 / block_size}[fmt]
    return cfg.num_layers * cfg.num_kv_heads * per_head_layer


def blocks_for_budget(cfg: ModelConfig, fmt: str, budget_bytes: int, block_size: int, n_stage: int = 0) -> int:
    """Blocks that fit in budget_bytes, after the bf16 staging slots (format B)."""
    stage = n_stage * cfg.num_layers * cfg.num_kv_heads * block_size * cfg.head_dim * 2
    return int((budget_bytes - stage) // (block_size * kv_bytes_per_token(cfg, fmt, block_size)))


class QuantPagedKVCache(PagedKVCache):
    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, device: str | torch.device,
                 dtype: torch.dtype, attn_backend: str = "gather", fmt: str = "int8",
                 manager: BlockManager | None = None, n_stage: int = 128):
        """`dtype` is the model's activation dtype. Format B needs the BlockManager, to release a freed sequence's
        staging slot, and n_stage slots (see the module docstring for the bound)."""
        if fmt not in ("int8", "int8-kchannel"):
            raise ValueError(f"fmt must be int8 or int8-kchannel, got {fmt}")
        self.fmt, self.n_stage = fmt, n_stage
        super().__init__(cfg, num_blocks, block_size, device, dtype, attn_backend)
        # Fused Triton quantize-and-store (engine/triton_quant.py) on GPU pools; the torch reference otherwise. The
        # two produce identical int8 values and scales; the fused path saves ~20 launches per layer.
        self.fused_writes = torch.device(device).type == "cuda"
        if fmt == "int8-kchannel":
            if manager is None:
                raise ValueError("int8-kchannel needs the BlockManager to track partial blocks")
            self._stage_of: dict[int, int] = {}  # block id -> staging slot
            self._free_stage = list(range(n_stage - 1, -1, -1))
            manager.add_free_listener(self._release_blocks)

    def _allocate(self, shape: tuple, device: str | torch.device, dtype: torch.dtype) -> None:
        """int8 K/V, fp16 scales, and (format B) the bf16 staging slots."""
        L, H, N, B, D = shape
        self.k = torch.zeros(shape, device=device, dtype=torch.int8)
        self.v = torch.zeros(shape, device=device, dtype=torch.int8)
        self.v_scale = torch.zeros(L, H, N, B, device=device, dtype=torch.float16)
        if self.fmt == "int8":
            self.k_scale = torch.zeros(L, H, N, B, device=device, dtype=torch.float16)
        else:
            self.k_scale = torch.zeros(L, H, N, D, device=device, dtype=torch.float16)
            self.stage = torch.zeros(L, H, self.n_stage, B, D, device=device, dtype=torch.bfloat16)

    def _release_blocks(self, blocks: list[int]) -> None:
        """BlockManager listener: freed blocks give back their staging slots."""
        for b in blocks:
            slot = self._stage_of.pop(b, None)
            if slot is not None:
                self._free_stage.append(slot)

    def begin(self, n_tokens: int, meta: PagedMetadata | None) -> None:
        """Base bookkeeping, then (format B) staging slots for every block written this forward."""
        super().begin(n_tokens, meta)
        if self.fmt != "int8-kchannel":
            return
        B, dev = self.block_size, meta.positions.device
        by_block: dict[int, list[tuple[int, int]]] = {}  # block -> [(token row, offset)], in token order
        for row, slot in enumerate(meta.slots_cpu):
            b, off = divmod(slot, B)
            by_block.setdefault(b, []).append((row, off))
        staged_rows, staged_slots, staged_offs, done, direct, direct_rows = [], [], [], [], [], []
        for b, items in by_block.items():
            if b not in self._stage_of and items[0][1] == 0 and len(items) == B:
                direct.append(b)  # written whole by this forward: quantize straight from k, no staging
                direct_rows += [r for r, _ in items]
                continue
            if b not in self._stage_of:
                if not self._free_stage:
                    raise RuntimeError("out of INT8 staging slots; raise n_stage")
                self._stage_of[b] = self._free_stage.pop()
            staged_rows += [r for r, _ in items]
            staged_slots += [self._stage_of[b]] * len(items)
            staged_offs += [o for _, o in items]
            if items[-1][1] == B - 1:
                done.append(b)
        as_long = lambda xs: torch.tensor(xs, dtype=torch.int64, device=dev)  # noqa: E731
        self._staged = (as_long(staged_rows), as_long(staged_slots), as_long(staged_offs)) if staged_rows else None
        self._done = (as_long(done), as_long([self._stage_of[b] for b in done])) if done else None
        self._direct = (as_long(direct), as_long(direct_rows)) if direct else None
        for b in done:  # free now; nothing reassigns a slot before the next forward's begin()
            self._free_stage.append(self._stage_of.pop(b))
        # Per sequence: the staging slot of its partial last block (-1 if the last block is full).
        tails = [self._stage_of[meta.block_ids_cpu[i][-1]] if ctx % B else -1
                 for i, ctx in enumerate(meta.context_lens)]
        self._tails = torch.tensor(tails, dtype=torch.int32, device=dev)
        if self._decode is not None and len(self._decode_seqs) != meta.num_seqs:
            self._tails_dec = self._tails[torch.tensor(self._decode_seqs, device=dev)]
        else:
            self._tails_dec = self._tails

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Quantize and store this forward's k, v [T, H_kv, d]."""
        m = self.meta
        if self.fused_writes:
            self._write_fused(layer, k, v)
            return
        vq, vs = quantize_per_token(v)
        self.v[layer][:, m.slot_blocks, m.slot_offsets] = vq.transpose(0, 1)
        self.v_scale[layer][:, m.slot_blocks, m.slot_offsets] = vs.t()
        if self.fmt == "int8":
            kq, ks = quantize_per_token(k)
            self.k[layer][:, m.slot_blocks, m.slot_offsets] = kq.transpose(0, 1)
            self.k_scale[layer][:, m.slot_blocks, m.slot_offsets] = ks.t()
            return
        # Both paths quantize the bf16-rounded K, so a block's int8 contents don't depend on how it was chunked.
        if self._staged is not None:
            rows, slots, offs = self._staged
            self.stage[layer][:, slots, offs] = k[rows].transpose(0, 1).to(torch.bfloat16)
        if self._done is not None:
            blocks, slots = self._done
            kq, ks = quantize_per_channel(self.stage[layer][:, slots], dim=2)  # [H, n, B, d] over the B tokens
            self.k[layer][:, blocks] = kq
            self.k_scale[layer][:, blocks] = ks
        if self._direct is not None:
            blocks, rows = self._direct
            H, d = k.shape[1], k.shape[2]
            full = k[rows].view(-1, self.block_size, H, d).permute(2, 0, 1, 3).to(torch.bfloat16)  # [H, n, B, d]
            kq, ks = quantize_per_channel(full, dim=2)
            self.k[layer][:, blocks] = kq
            self.k_scale[layer][:, blocks] = ks

    def _write_fused(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """write() through the fused kernels: V (and format A's K) per token in one launch each; format B's K is
        staged, and blocks completed this forward are quantized from staging in one launch."""
        from engine.triton_quant import quantize_store_blocks, quantize_store_tokens

        m = self.meta
        quantize_store_tokens(v, m.slot_blocks, m.slot_offsets, self.v[layer], self.v_scale[layer])
        if self.fmt == "int8":
            quantize_store_tokens(k, m.slot_blocks, m.slot_offsets, self.k[layer], self.k_scale[layer])
            return
        if self._staged is not None:
            rows, slots, offs = self._staged
            self.stage[layer][:, slots, offs] = k[rows].transpose(0, 1).to(torch.bfloat16)
        if self._done is not None:
            blocks, slots = self._done
            quantize_store_blocks(self.stage[layer], slots, blocks, self.k[layer], self.k_scale[layer])
        if self._direct is not None:  # prefill-chunk interiors: rarer, kept on the reference path
            blocks, rows = self._direct
            H, d = k.shape[1], k.shape[2]
            full = k[rows].view(-1, self.block_size, H, d).permute(2, 0, 1, 3).to(torch.bfloat16)
            kq, ks = quantize_per_channel(full, dim=2)
            self.k[layer][:, blocks] = kq
            self.k_scale[layer][:, blocks] = ks

    def gather(self, layer: int, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequence seq's dequantized K, V as [H_kv, context_len, d] in the model dtype."""
        m, (H, d), B = self.meta, (self.cfg.num_kv_heads, self.cfg.head_dim), self.block_size
        ctx, blocks = m.context_lens[seq], m.block_ids[seq]
        v = dequantize(self.v[layer][:, blocks], self.v_scale[layer][:, blocks][..., None], self.dtype)
        if self.fmt == "int8":
            k = dequantize(self.k[layer][:, blocks], self.k_scale[layer][:, blocks][..., None], self.dtype)
        else:
            k = dequantize(self.k[layer][:, blocks], self.k_scale[layer][:, blocks][:, :, None], self.dtype)
            if ctx % B:
                k[:, -1] = self.stage[layer][:, self._stage_of[m.block_ids_cpu[seq][-1]]].to(self.dtype)
        return k.view(H, -1, d)[:, :ctx], v.view(H, -1, d)[:, :ctx]

    def _kernel_quant(self, layer: int) -> QuantKV:
        """Scales (and format B's staging) for the decode kernel."""
        if self.fmt == "int8":
            return QuantKV(1, self.k_scale[layer], self.v_scale[layer])
        return QuantKV(2, self.k_scale[layer], self.v_scale[layer], self.stage[layer], self._tails_dec)
