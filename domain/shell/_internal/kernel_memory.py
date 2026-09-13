"""
AI-Native Memory Manager — tensor blocks, not pages.

Memory is allocated by shape + dtype, not by byte count. The allocator
manages a pool of memory blocks, tracks ownership per process, and
handles defragmentation.
"""

from __future__ import annotations

import threading
import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryBlock:
    """A contiguous allocation of tensor memory."""
    block_id: int
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    owner_pid: int | None = None
    data: np.ndarray | None = None
    allocated_at: float | None = None
    freed: bool = False

    @property
    def num_elements(self) -> int:
        s = 1
        for d in self.shape:
            s *= d
        return s


class TensorMemory:
    """
    AI-native memory manager.

    Memory is a pool of blocks. Each block holds a numpy array of a given
    shape and dtype. Allocation finds or creates a block of the right size.
    Freeing returns the block to the pool.

    Thread-safe via a lock.
    """

    def __init__(self, capacity_bytes: int = 256 * 1024 * 1024):  # 256 MB default
        self._capacity = capacity_bytes
        self._used = 0
        self._blocks: dict[int, MemoryBlock] = {}
        self._free_blocks: list[int] = []  # block_ids of freed blocks
        self._next_id = 1
        self._lock = threading.Lock()
        self._pid_blocks: dict[int, list[int]] = {}  # pid → block_ids

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def free(self) -> int:
        with self._lock:
            return self._capacity - self._used

    @property
    def utilization(self) -> float:
        with self._lock:
            return self._used / self._capacity if self._capacity > 0 else 0.0

    def allocate(self, shape: tuple[int, ...], dtype: str = "float32",
                 owner_pid: int | None = None) -> MemoryBlock:
        """
        Allocate a tensor block of given shape and dtype.

        Returns MemoryBlock with a live numpy array.
        Raises MemoryError if insufficient capacity.
        """
        dtype_obj = np.dtype(dtype)
        size_bytes = int(np.prod(shape)) * dtype_obj.itemsize

        with self._lock:
            if size_bytes > self._capacity - self._used:
                raise MemoryError(
                    f"Out of tensor memory: need {size_bytes} bytes, "
                    f"{self._capacity - self._used} available"
                )

            block_id = self._next_id
            self._next_id += 1

            data = np.zeros(shape, dtype=dtype_obj)
            block = MemoryBlock(
                block_id=block_id,
                shape=shape,
                dtype=dtype,
                size_bytes=size_bytes,
                owner_pid=owner_pid,
                data=data,
            )
            self._blocks[block_id] = block
            self._used += size_bytes

            if owner_pid is not None:
                self._pid_blocks.setdefault(owner_pid, []).append(block_id)

            return block

    def free_block(self, block_id: int) -> bool:
        """Free a tensor block. Returns True if freed."""
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None or block.freed:
                return False

            block.freed = True
            block.data = None
            self._used -= block.size_bytes
            self._free_blocks.append(block_id)

            if block.owner_pid is not None:
                pid_blocks = self._pid_blocks.get(block.owner_pid, [])
                if block_id in pid_blocks:
                    pid_blocks.remove(block_id)

            return True

    def get(self, block_id: int) -> MemoryBlock | None:
        """Get a block by ID. Returns None if freed or nonexistent."""
        with self._lock:
            block = self._blocks.get(block_id)
            if block is not None and not block.freed:
                return block
            return None

    def read(self, block_id: int) -> np.ndarray | None:
        """Read tensor data from a block."""
        block = self.get(block_id)
        if block is not None and block.data is not None:
            return block.data.copy()
        return None

    def write(self, block_id: int, data: np.ndarray) -> bool:
        """Write tensor data into a block. Shape must match."""
        with self._lock:
            block = self._blocks.get(block_id)
            if block is None or block.freed:
                return False
            if block.data is None:
                return False
            if data.shape != block.shape:
                return False
            if str(data.dtype) != block.dtype:
                data = data.astype(block.dtype)
            block.data = data.copy()
            return True

    def free_pid(self, pid: int) -> int:
        """Free all blocks owned by a process. Returns count freed."""
        with self._lock:
            block_ids = self._pid_blocks.get(pid, [])
            count = 0
            for bid in list(block_ids):
                block = self._blocks.get(bid)
                if block and not block.freed:
                    block.freed = True
                    block.data = None
                    self._used -= block.size_bytes
                    self._free_blocks.append(bid)
                    count += 1
            self._pid_blocks.pop(pid, None)
            return count

    def stats(self) -> dict[str, Any]:
        """Return memory statistics."""
        with self._lock:
            total_blocks = len(self._blocks)
            freed_blocks = sum(1 for b in self._blocks.values() if b.freed)
            active_blocks = total_blocks - freed_blocks
            used = self._used
            capacity = self._capacity
            return {
                "capacity_bytes": capacity,
                "used_bytes": used,
                "free_bytes": capacity - used,
                "utilization": used / capacity if capacity > 0 else 0.0,
                "total_blocks": total_blocks,
                "active_blocks": active_blocks,
                "freed_blocks": freed_blocks,
            }

    def defragment(self) -> int:
        """
        Defragment by removing freed blocks from the block table.
        Returns count of blocks reclaimed.
        """
        with self._lock:
            reclaimed = 0
            for block_id in list(self._free_blocks):
                if block_id in self._blocks:
                    del self._blocks[block_id]
                    reclaimed += 1
            self._free_blocks.clear()
            return reclaimed
