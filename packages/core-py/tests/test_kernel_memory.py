"""Coverage tests for domains/shell/kernel_memory.py."""

import numpy as np
import pytest

from domains.shell.kernel_memory import MemoryBlock, TensorMemory


def test_memory_block_num_elements():
    b = MemoryBlock(block_id=1, shape=(2, 3, 4), dtype="float32", size_bytes=96)
    assert b.num_elements == 24
    b2 = MemoryBlock(block_id=2, shape=(), dtype="float32", size_bytes=4)
    assert b2.num_elements == 1


def test_properties_init():
    m = TensorMemory(capacity_bytes=1000)
    assert m.capacity == 1000
    assert m.used == 0
    assert m.free == 1000
    assert m.utilization == 0.0


def test_allocate_basic():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((2, 3), owner_pid=7)
    assert b.block_id == 1
    assert b.data.shape == (2, 3)
    assert b.dtype == "float32"
    assert m.used == 24
    assert m.free == 976
    assert m.utilization == pytest.approx(0.024)
    assert m.get(1) is b


def test_allocate_reuses_next_id():
    m = TensorMemory(capacity_bytes=1000)
    a = m.allocate((1,), owner_pid=1)
    m.free_block(a.block_id)
    b = m.allocate((1,))
    assert b.block_id == 2


def test_allocate_out_of_memory():
    m = TensorMemory(capacity_bytes=10)
    with pytest.raises(MemoryError, match="Out of tensor memory"):
        m.allocate((100,), dtype="float64")


def test_allocate_dtype_variants():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((2,), dtype="int8")
    assert b.dtype == "int8"
    assert b.size_bytes == 2
    assert m.get(1).data.dtype == np.int8


def test_free_block():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((2, 3), owner_pid=7)
    assert m.free_block(b.block_id) is True
    assert m.free == 1000
    assert m.get(b.block_id) is None


def test_free_block_double_free_returns_false():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((2, 3), owner_pid=7)
    assert m.free_block(b.block_id) is True
    assert m.free_block(b.block_id) is False


def test_free_block_unknown_returns_false():
    m = TensorMemory(capacity_bytes=1000)
    assert m.free_block(999) is False


def test_free_block_removes_pid_bookkeeping():
    m = TensorMemory(capacity_bytes=1000)
    a = m.allocate((1,), owner_pid=7)
    b = m.allocate((1,), owner_pid=7)
    m.free_block(a.block_id)
    assert m._pid_blocks[7] == [b.block_id]
    m.free_block(b.block_id)
    assert m._pid_blocks[7] == []


def test_get_freed_block_returns_none():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((1,))
    m.free_block(b.block_id)
    assert m.get(b.block_id) is None
    assert m.get(999) is None


def test_read_returns_copy():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((2,))
    b.data[0] = 5
    out = m.read(b.block_id)
    assert out[0] == 5
    out[0] = 99
    assert m.get(b.block_id).data[0] == 5


def test_read_freed_or_unknown_returns_none():
    m = TensorMemory(capacity_bytes=1000)
    assert m.read(999) is None
    b = m.allocate((1,))
    m.free_block(b.block_id)
    assert m.read(b.block_id) is None


def test_write_success():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((2, 3), owner_pid=7)
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert m.write(b.block_id, data) is True
    assert (m.get(b.block_id).data == data).all()


def test_write_rejects_wrong_shape():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((2, 3))
    assert m.write(b.block_id, np.zeros((3, 2))) is False


def test_write_rejects_freed():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((2,))
    m.free_block(b.block_id)
    assert m.write(b.block_id, np.zeros(2)) is False
    assert m.write(999, np.zeros(2)) is False


def test_write_rejects_none_data():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((2,))
    b.data = None
    assert m.write(b.block_id, np.zeros(2)) is False


def test_write_casts_dtype():
    m = TensorMemory(capacity_bytes=1000)
    b = m.allocate((2,), dtype="float32")
    assert m.write(b.block_id, np.array([1, 2], dtype=np.int64)) is True
    assert m.get(b.block_id).data.dtype == np.float32


def test_free_pid():
    m = TensorMemory(capacity_bytes=1000)
    a = m.allocate((1,), owner_pid=7)
    b = m.allocate((1,), owner_pid=7)
    c = m.allocate((1,), owner_pid=8)
    assert m.free_pid(7) == 2
    assert m.used == 4
    assert m.get(a.block_id) is None
    assert m.get(b.block_id) is None
    assert m.get(c.block_id) is not None
    assert m.free_pid(7) == 0
    assert 7 not in m._pid_blocks


def test_stats():
    m = TensorMemory(capacity_bytes=1000)
    m.allocate((2,), owner_pid=7)
    b = m.allocate((2,), owner_pid=7)
    m.free_block(b.block_id)
    stats = m.stats()
    assert stats["capacity_bytes"] == 1000
    assert stats["used_bytes"] == 8
    assert stats["free_bytes"] == 992
    assert stats["total_blocks"] == 2
    assert stats["active_blocks"] == 1
    assert stats["freed_blocks"] == 1


def test_stats_zero_capacity():
    m = TensorMemory(capacity_bytes=0)
    assert m.utilization == 0.0
    assert m.stats()["utilization"] == 0.0
    assert m.stats()["free_bytes"] == 0


def test_defragment():
    m = TensorMemory(capacity_bytes=1000)
    a = m.allocate((1,), owner_pid=7)
    b = m.allocate((1,), owner_pid=7)
    m.free_block(a.block_id)
    m.free_block(b.block_id)
    assert m.defragment() == 2
    assert len(m._blocks) == 0
    assert m.stats()["total_blocks"] == 0
    assert m.defragment() == 0
