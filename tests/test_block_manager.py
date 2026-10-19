"""Phase 2 block manager: unit behavior and invariant 3 (randomized operations leave every block free exactly once)."""

import random

import pytest

from engine.block_manager import BlockManager, OutOfBlocks


def test_allocate_append_free_slots():
    """Slots follow the block table; a new block is taken exactly when the last one fills."""
    m = BlockManager(num_blocks=8, block_size=4)
    assert m.allocate(0, 6) == [0, 1, 2, 3, 4, 5]  # blocks 0, 1
    assert m.block_table(0) == [0, 1] and m.num_free_blocks() == 6
    assert m.append_slot(0) == 6  # block 1, offset 2
    assert m.append_slot(0) == 7
    assert m.new_blocks_needed(0) == 1
    assert m.append_slot(0) == 8  # block 2, offset 0: a new block
    assert m.seq_len(0) == 9 and m.block_table(0) == [0, 1, 2]
    assert m.append_slots(0, 5) == [9, 10, 11, 12, 13]  # finishes block 2, takes block 3
    m.free(0)
    assert m.num_free_blocks() == 8
    m.check_invariants()


def test_lifo_reuse_shuffles_tables():
    """Freed blocks are handed out first, so a later sequence gets a non-contiguous, non-monotonic table."""
    m = BlockManager(num_blocks=8, block_size=4)
    for s in range(4):
        m.allocate(s, 8)  # 2 blocks each: 0-1, 2-3, 4-5, 6-7
    m.free(1)
    m.free(3)
    m.allocate(9, 16)
    table = m.block_table(9)
    assert sorted(table) == [2, 3, 6, 7] and table != sorted(table)
    m.check_invariants()


def test_zero_length_and_errors():
    """Zero-token sequences own nothing; failed requests change nothing."""
    m = BlockManager(num_blocks=2, block_size=4)
    assert m.allocate(0, 0) == [] and m.block_table(0) == []
    assert m.append_slots(0, 8) == [0, 1, 2, 3, 4, 5, 6, 7]
    with pytest.raises(OutOfBlocks):
        m.append_slot(0)
    with pytest.raises(OutOfBlocks):
        m.allocate(1, 1)
    with pytest.raises(ValueError):
        m.allocate(0, 1)
    assert m.seq_len(0) == 8 and m.num_free_blocks() == 0
    assert not m.can_allocate(1) and not m.can_append(0)
    m.free(0)
    assert m.can_allocate(8) and not m.can_allocate(8, watermark_blocks=1)
    m.check_invariants()


def test_check_invariants_catches_corruption():
    """The checker itself works: a double-listed or leaked block is reported."""
    m = BlockManager(num_blocks=4, block_size=4)
    m.allocate(0, 4)
    m._free.append(m.block_table(0)[0])  # block both owned and free
    with pytest.raises(AssertionError):
        m.check_invariants()
    m = BlockManager(num_blocks=4, block_size=4)
    m._free.pop()  # leaked: neither free nor owned
    with pytest.raises(AssertionError):
        m.check_invariants()


@pytest.mark.parametrize("seed", range(50))
def test_randomized_no_leaks(seed):
    """Invariant 3: 400 random allocate / append / chunk-append / free operations, some failing for lack of blocks.
    After every operation the bookkeeping is consistent, live sequences' slots are disjoint and match a shadow model,
    and after freeing everything every block is free exactly once."""
    rng = random.Random(seed)
    B = rng.choice([1, 4, 16])
    m = BlockManager(num_blocks=rng.randint(4, 64), block_size=B)
    shadow: dict[int, list[int]] = {}  # seq_id -> its slots, in position order
    next_id = 0
    for _ in range(400):
        op = rng.random()
        if op < 0.3 or not shadow:
            n = rng.randint(0, 3 * B)
            if m.can_allocate(n):
                shadow[next_id] = m.allocate(next_id, n)
            else:
                with pytest.raises(OutOfBlocks):
                    m.allocate(next_id, n)
            next_id += 1
        elif op < 0.8:
            s, n = rng.choice(list(shadow)), rng.choice([1, 1, 1, rng.randint(1, 2 * B)])
            if m.can_append(s, n):
                shadow[s] += m.append_slots(s, n)
            else:
                with pytest.raises(OutOfBlocks):
                    m.append_slots(s, n)
        else:
            s = rng.choice(list(shadow))
            m.free(s)
            del shadow[s]
        m.check_invariants()
        live = [slot for slots in shadow.values() for slot in slots]
        assert len(live) == len(set(live)), "two live tokens share a slot"
        for s, slots in shadow.items():
            assert m.seq_len(s) == len(slots) and m.slots(s, 0, len(slots)) == slots
        assert m.num_free_blocks() == m.num_blocks - sum(m.blocks_for(len(v)) for v in shadow.values())
    for s in list(shadow):
        m.free(s)
    m.check_invariants()
    assert sorted(m._free) == list(range(m.num_blocks))
    assert all(m.ref_count(b) == 0 for b in range(m.num_blocks))
