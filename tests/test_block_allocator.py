import pytest

from llm_runtime.block_allocator import BlockAllocator
from llm_runtime.continuous_batcher import ContinuousBatcher
from llm_runtime.request import RequestState
from runtime_fakes import FakeModel, FakeTokenizer


def test_growth_multiple_requests_and_snapshot_isolation():
    pool = BlockAllocator(4, 2)
    assert pool.reserve("a", 1) == [0]
    assert pool.reserve("a", 2) == [0]
    assert pool.reserve("a", 3) == [0, 1]
    assert pool.reserve("b", 4) == [2, 3]
    assert pool.stats()["utilization"] == 1
    snapshot = pool.stats()
    snapshot["request_block_tables"]["a"].clear()
    assert pool.reserve("a", 1) == [0, 1]  # No implicit shrink.
    with pytest.raises(MemoryError):
        pool.reserve("a", 5)
    with pytest.raises(MemoryError):
        pool.reserve("c", 1)
    assert pool.stats()["request_block_tables"] == {"a": [0, 1], "b": [2, 3]}


def test_fragmented_reuse_is_deterministic_and_double_free_rejected():
    pool = BlockAllocator(4, 1)
    for name in "abcd":
        pool.reserve(name, 1)
    pool.free("b")
    pool.free("d")
    assert pool.reserve("e", 2) == [1, 3]  # Noncontiguous physical IDs are fine.
    pool.free("e")
    with pytest.raises(ValueError):
        pool.free("e")
    pool.free("a")
    pool.free("c")
    assert pool.stats()["free_blocks"] == 4
    assert pool.reserve("f", 4) == [1, 3, 0, 2]


@pytest.mark.parametrize("blocks,size", [(0, 2), (2, 0), (-1, 1)])
def test_invalid_pool(blocks, size):
    with pytest.raises(ValueError):
        BlockAllocator(blocks, size)


def test_zero_and_invalid_reservations():
    pool = BlockAllocator(2, 2)
    assert pool.reserve("zero", 0) == [] and not pool.owns("zero")
    for name, count in [("", 1), ("a", -1)]:
        with pytest.raises(ValueError):
            pool.reserve(name, count)


def test_continuous_lifecycle_growth_completion_and_cancel():
    pool = BlockAllocator(3, 2)
    engine = ContinuousBatcher(FakeModel(), FakeTokenizer(), 2, pool)
    a, b = RequestState("a", 3, "A"), RequestState("b", 1, "B")
    engine.add_request(a)
    engine.add_request(b)
    first = engine.step()
    assert first["logical_kv_before_forward"]["request_block_tables"] == {"A": [0], "B": [1]}
    assert b.kv_block_ids == [] and pool.stats()["request_block_tables"] == {"A": [0]}
    engine.step()
    assert a.kv_block_ids == [0, 2]
    engine.cancel_request("A")
    assert pool.stats()["free_blocks"] == 3 and a.kv_block_ids == []
    zero = RequestState("a", 0, "zero")
    engine.add_request(zero)
    engine.step()
    assert zero.is_finished and pool.stats()["allocated_blocks"] == 0


def test_pool_exhaustion_fails_cohort_without_leaking_blocks():
    pool = BlockAllocator(1, 1)
    engine = ContinuousBatcher(FakeModel(), FakeTokenizer(), block_allocator=pool)
    state = RequestState("a", 2, "A")
    engine.add_request(state)
    with pytest.raises(MemoryError):
        engine.step()
    assert state.stop_reason == "error" and pool.stats()["free_blocks"] == 1
    engine.cancel_request("A")


def test_forward_error_releases_reserved_blocks():
    class BrokenModel(FakeModel):
        def __call__(self, **kwargs):
            raise RuntimeError("broken")
    pool = BlockAllocator(4, 2)
    engine = ContinuousBatcher(BrokenModel(), FakeTokenizer(), block_allocator=pool)
    engine.add_request(RequestState("a", 2, "A"))
    with pytest.raises(RuntimeError):
        engine.step()
    assert pool.stats()["allocated_blocks"] == 0
    assert engine.requests["A"].stop_reason == "error"
