"""Logical fixed-size KV capacity, not physical tensors or a paged attention kernel."""

from collections import deque


class BlockAllocator:
    """Own numbered blocks. Growth is atomic; freed IDs rejoin a FIFO free list."""

    def __init__(self, total_blocks: int, block_size: int):
        if total_blocks < 1 or block_size < 1:
            raise ValueError("total_blocks and block_size must be positive.")
        self.total_blocks = total_blocks
        self.block_size = block_size
        self._free = deque(range(total_blocks))
        self._tables: dict[str, list[int]] = {}

    def reserve(self, request_id: str, token_count: int) -> list[int]:
        """Allocate/grow to cover processed context tokens; never shrink implicitly."""
        if not request_id or token_count < 0:
            raise ValueError("A nonempty request ID and nonnegative token count are required.")
        required = (token_count + self.block_size - 1) // self.block_size
        current = self._tables.get(request_id, [])
        extra = max(0, required - len(current))
        if extra > len(self._free):
            raise MemoryError("Logical KV block pool exhausted.")
        if required or request_id in self._tables:
            table = self._tables.setdefault(request_id, [])
            table.extend(self._free.popleft() for _ in range(extra))
            return list(table)
        return []  # Zero tokens do not create an owner.

    def free(self, request_id: str) -> None:
        """Unknown/double frees are errors, not duplicate free-list entries."""
        if request_id not in self._tables:
            raise ValueError("Request does not own any blocks.")
        self._free.extend(self._tables.pop(request_id))

    def owns(self, request_id: str) -> bool:
        return request_id in self._tables

    def stats(self) -> dict:
        allocated = self.total_blocks - len(self._free)
        return {
            "total_blocks": self.total_blocks,
            "block_size": self.block_size,
            "free_blocks": len(self._free),
            "allocated_blocks": allocated,
            "utilization": allocated / self.total_blocks,
            "request_block_tables": {key: list(value) for key, value in self._tables.items()},
        }
