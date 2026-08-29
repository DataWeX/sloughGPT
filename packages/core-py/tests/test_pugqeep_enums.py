"""Tests for domains.infrastructure.pugqeep — EvictionPolicy, Tier, ProcessStatus, StemStatus, TreeStatus, TaskStatus, TaskPriority, Task, CacheEntry, Process."""

from domains.infrastructure.pugqeep.cache import EvictionPolicy, Tier
from domains.infrastructure.pugqeep.engine import ProcessStatus, StemStatus, TreeStatus, Process
from domains.infrastructure.pugqeep.task_queue import TaskStatus, TaskPriority, Task


class TestEvictionPolicy:
    def test_all_members(self):
        assert len(EvictionPolicy) == 2
    def test_values(self):
        assert EvictionPolicy.LRU.value == "lru"
        assert EvictionPolicy.LFU.value == "lfu"


class TestTier:
    def test_all_members(self):
        assert len(Tier) == 3
    def test_values(self):
        assert Tier.DISK.value == "disk"
        assert Tier.HOT.value == "hot"
        assert Tier.MEMORY.value == "memory"


class TestProcessStatus:
    def test_all_members(self):
        assert len(ProcessStatus) == 7
    def test_values(self):
        assert ProcessStatus.CREATED.value == "created"
        assert ProcessStatus.RUNNING.value == "running"
        assert ProcessStatus.CANCELLED.value == "cancelled"


class TestStemStatus:
    def test_all_members(self):
        assert len(StemStatus) == 4
    def test_values(self):
        assert StemStatus.CREATED.value == "created"
        assert StemStatus.COMPLETED.value == "completed"


class TestTreeStatus:
    def test_all_members(self):
        assert len(TreeStatus) == 3
    def test_values(self):
        assert TreeStatus.IDLE.value == "idle"
        assert TreeStatus.BRANCHING.value == "branching"


class TestTaskStatus:
    def test_all_members(self):
        assert len(TaskStatus) == 5
    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTaskPriority:
    def test_all_members(self):
        assert len(TaskPriority) == 4
    def test_values(self):
        assert TaskPriority.LOW.value == 0
        assert TaskPriority.NORMAL.value == 1
        assert TaskPriority.URGENT.value == 3


class TestTask:
    def test_defaults(self):
        t = Task()
        assert t.status == TaskStatus.PENDING
        assert t.priority == TaskPriority.NORMAL
        assert t.data is None
    def test_custom(self):
        t = Task(name="test", priority=TaskPriority.HIGH, data={"key": "val"})
        assert t.name == "test"
        assert t.priority == TaskPriority.HIGH


class TestProcess:
    def test_defaults(self):
        p = Process(fn=lambda: None)
        assert p.status == ProcessStatus.CREATED
    def test_custom(self):
        p = Process(fn=lambda: None, name="test")
        assert p.name == "test"
