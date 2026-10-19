"""Block manager for the paged KV cache: which fixed-size blocks each sequence owns. Pure CPU bookkeeping.

Sizes for Qwen2.5-0.5B in bf16 with 16-token blocks: one block holds 16 tokens x 12,288 B = 192 KiB across all 24
layers, so a 2 GiB pool is 10,922 blocks (about 175k tokens). A sequence of n tokens owns ceil(n / 16) blocks; at
most 15 slots of its last block are unused, so waste is under one block per sequence instead of the max_len
reservation a contiguous cache needs.

A token's slot is block_id * block_size + offset. Blocks come off a LIFO free list, so a freed block is the next one
handed out (its memory is likely still in L2) and the block tables of long-lived sequences end up shuffled, which
the attention reference and kernels must handle. Reference counts are kept per block so that blocks can later be
shared (prefix caching); without sharing they are always 0 or 1.
"""


class OutOfBlocks(RuntimeError):
    """Not enough free blocks for the request; nothing was changed."""


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16):
        """All blocks start free."""
        if num_blocks < 1 or block_size < 1:
            raise ValueError("need at least one block of at least one token")
        self.num_blocks, self.block_size = num_blocks, block_size
        self._free = list(range(num_blocks - 1, -1, -1))  # stack: pop() hands out block 0 first
        self._ref = [0] * num_blocks
        self._tables: dict[int, list[int]] = {}
        self._lengths: dict[int, int] = {}
        self._free_listeners: list = []

    def add_free_listener(self, fn) -> None:
        """Call fn(blocks) with the blocks a free() returns to the free list (e.g. to drop per-block side state)."""
        self._free_listeners.append(fn)

    def num_free_blocks(self) -> int:
        """Blocks not owned by any sequence."""
        return len(self._free)

    def blocks_for(self, n_tokens: int) -> int:
        """Blocks needed to hold n_tokens: ceil(n_tokens / block_size)."""
        return -(-n_tokens // self.block_size)

    def can_allocate(self, n_tokens: int, watermark_blocks: int = 0) -> bool:
        """Whether a new sequence of n_tokens fits while leaving watermark_blocks free (headroom for decoding)."""
        return self.blocks_for(n_tokens) + watermark_blocks <= len(self._free)

    def _take(self, n_blocks: int) -> list[int]:
        """Pop n_blocks free blocks and give them reference count 1."""
        blocks = [self._free.pop() for _ in range(n_blocks)]
        for b in blocks:
            assert self._ref[b] == 0, f"block {b} on the free list with ref count {self._ref[b]}"
            self._ref[b] = 1
        return blocks

    def allocate(self, seq_id: int, n_tokens: int) -> list[int]:
        """Create a sequence holding n_tokens (n_tokens >= 0) and return their slots."""
        if seq_id in self._tables:
            raise ValueError(f"sequence {seq_id} already allocated")
        if n_tokens < 0:
            raise ValueError("n_tokens must be >= 0")
        if not self.can_allocate(n_tokens):
            raise OutOfBlocks(f"{self.blocks_for(n_tokens)} blocks needed, {len(self._free)} free")
        self._tables[seq_id] = self._take(self.blocks_for(n_tokens))
        self._lengths[seq_id] = n_tokens
        return self.slots(seq_id, 0, n_tokens)

    def new_blocks_needed(self, seq_id: int, n: int = 1) -> int:
        """Blocks that appending n tokens to seq_id would take from the free list."""
        return self.blocks_for(self._lengths[seq_id] + n) - len(self._tables[seq_id])

    def can_append(self, seq_id: int, n: int = 1) -> bool:
        """Whether n more tokens fit for seq_id."""
        return self.new_blocks_needed(seq_id, n) <= len(self._free)

    def append_slots(self, seq_id: int, n: int = 1) -> list[int]:
        """Extend seq_id by n tokens (one decode step, or a prefill chunk) and return their slots."""
        need = self.new_blocks_needed(seq_id, n)
        if need > len(self._free):
            raise OutOfBlocks(f"sequence {seq_id}: {need} blocks needed, {len(self._free)} free")
        self._tables[seq_id] += self._take(need)
        start = self._lengths[seq_id]
        self._lengths[seq_id] = start + n
        return self.slots(seq_id, start, start + n)

    def append_slot(self, seq_id: int) -> int:
        """Extend seq_id by one token and return its slot."""
        return self.append_slots(seq_id, 1)[0]

    def free(self, seq_id: int) -> None:
        """Release a sequence; each of its blocks returns to the free list when its reference count reaches 0."""
        released = []
        for b in reversed(self._tables.pop(seq_id)):  # reversed: the stack then hands them out in table order
            self._ref[b] -= 1
            assert self._ref[b] >= 0, f"block {b} freed twice"
            if self._ref[b] == 0:
                self._free.append(b)
                released.append(b)
        del self._lengths[seq_id]
        for fn in self._free_listeners:
            fn(released)

    def ref_count(self, block: int) -> int:
        """Number of sequences holding `block`."""
        return self._ref[block]

    def block_table(self, seq_id: int) -> list[int]:
        """The sequence's blocks in token order (a copy)."""
        return list(self._tables[seq_id])

    def seq_len(self, seq_id: int) -> int:
        """Tokens held by the sequence."""
        return self._lengths[seq_id]

    def seq_ids(self) -> list[int]:
        """Live sequences."""
        return list(self._tables)

    def slots(self, seq_id: int, start: int, end: int) -> list[int]:
        """Slots of the sequence's tokens at positions [start, end)."""
        if not 0 <= start <= end <= self._lengths[seq_id]:
            raise ValueError(f"positions [{start}, {end}) outside sequence {seq_id} of length {self._lengths[seq_id]}")
        table, B = self._tables[seq_id], self.block_size
        return [table[p // B] * B + p % B for p in range(start, end)]

    def check_invariants(self) -> None:
        """Assert the bookkeeping is consistent: every block is free xor owned, reference counts equal the number of
        owning tables, and each table has exactly the blocks its length needs. Raises AssertionError otherwise."""
        free = set(self._free)
        assert len(free) == len(self._free), "duplicate block on the free list"
        owners = [0] * self.num_blocks
        for seq_id, table in self._tables.items():
            assert len(table) == self.blocks_for(self._lengths[seq_id]), f"sequence {seq_id}: table size wrong"
            for b in table:
                owners[b] += 1
        for b in range(self.num_blocks):
            assert owners[b] == self._ref[b], f"block {b}: ref {self._ref[b]} but {owners[b]} owners"
            assert (b in free) == (owners[b] == 0), f"block {b}: free={b in free} with {owners[b]} owners"
