"""Tests for domains.infrastructure.cancel_manager — OpType, OpStatus, Operation; domains.infrastructure.cpu_topology — CpuTopology."""

import enum
from domains.infrastructure.cancel_manager import OpType, OpStatus, Operation
from domains.infrastructure.cpu_topology import CpuTopology


class TestOpType:
    def test_all_members(self):
        assert len(OpType) == 6
    def test_values(self):
        assert OpType.TRAINING.value == "training"
        assert OpType.INFERENCE.value == "inference"
        assert OpType.DOWNLOAD.value == "download"
        assert OpType.IMPORT.value == "import"
        assert OpType.BATCH.value == "batch"
        assert OpType.OTHER.value == "other"


class TestOpStatus:
    def test_all_members(self):
        assert len(OpStatus) == 6
    def test_values(self):
        assert OpStatus.REGISTERED.value == "registered"
        assert OpStatus.RUNNING.value == "running"
        assert OpStatus.CANCELLED.value == "cancelled"
        assert OpStatus.COMPLETED.value == "completed"
        assert OpStatus.FAILED.value == "failed"


class TestOperation:
    def test_fields(self):
        op = Operation(
            id="op1", op_type=OpType.TRAINING, label="train job",
            status=OpStatus.REGISTERED, cancel_fn=lambda: None, created_at=1.0,
        )
        assert op.id == "op1"
        assert op.op_type == OpType.TRAINING
        assert op.error is None
        assert op.meta == {}

    def test_to_dict(self):
        op = Operation(
            id="op1", op_type=OpType.TRAINING, label="train",
            status=OpStatus.RUNNING, cancel_fn=lambda: None, created_at=1.0,
        )
        d = op.to_dict()
        assert d["id"] == "op1"
        assert d["type"] == "training"
        assert d["status"] == "running"


class TestCpuTopology:
    def test_defaults(self):
        ct = CpuTopology()
        assert ct.logical_cores == 1
        assert ct.physical_cores == 1
        assert ct.has_hyperthreading is False

    def test_threads_per_core(self):
        ct = CpuTopology(logical_cores=8, physical_cores=4)
        assert ct.threads_per_core == 2

    def test_threads_per_core_zero(self):
        ct = CpuTopology(physical_cores=0)
        assert ct.threads_per_core == 1

    def test_effective_cores(self):
        ct = CpuTopology(logical_cores=8, physical_cores=4)
        assert ct.effective_cores == 5  # 4 + int(4 * 0.25)

    def test_summary(self):
        ct = CpuTopology(logical_cores=8, physical_cores=4, has_hyperthreading=True)
        s = ct.summary()
        assert "8L" in s
        assert "4P" in s
        assert "HT" in s

    def test_frozen(self):
        ct = CpuTopology()
        try:
            ct.logical_cores = 16
            assert False, "Should be frozen"
        except AttributeError:
            pass
