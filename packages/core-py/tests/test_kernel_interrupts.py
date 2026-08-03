"""Coverage tests for domains/shell/kernel_interrupts.py."""

import pytest

from domains.shell.kernel_interrupts import (
    Interrupt,
    InterruptManager,
    InterruptType,
    InterruptVector,
)


def test_interrupt_type_values():
    assert InterruptType.TIMER == 0
    assert InterruptType.INFERENCE_DONE == 1
    assert InterruptType.TRAINING_STEP == 2
    assert InterruptType.DATA_READY == 3
    assert InterruptType.GRADIENT_UPDATE == 4
    assert InterruptType.DEVICE_ERROR == 5
    assert InterruptType.MEMORY_FULL == 6
    assert InterruptType.PROCESS_DONE == 7
    assert InterruptType.USER_INPUT == 8
    assert InterruptType.NETWORK_IO == 9
    assert InterruptType.CUSTOM == 10


def test_interrupt_dataclass():
    i = Interrupt(vector=InterruptType.TIMER, source_pid=1, data={"x": 1}, priority=3)
    assert i.vector == InterruptType.TIMER
    assert i.source_pid == 1
    assert i.data == {"x": 1}
    assert i.priority == 3


def test_register_and_fire():
    iv = InterruptVector()
    seen = []
    iv.register(InterruptType.TIMER, lambda i: seen.append(i))
    assert iv.fire(Interrupt(InterruptType.TIMER)) is True
    assert len(seen) == 1
    assert seen[0].vector == InterruptType.TIMER


def test_unregister():
    iv = InterruptVector()
    seen = []
    handler = lambda i: seen.append(i)
    iv.register(InterruptType.TIMER, handler)
    iv.unregister(InterruptType.TIMER)
    assert iv.fire(Interrupt(InterruptType.TIMER)) is False
    assert seen == []


def test_unregister_missing_does_not_raise():
    iv = InterruptVector()
    iv.unregister(InterruptType.CUSTOM)


def test_fire_no_handler_returns_false():
    iv = InterruptVector()
    assert iv.fire(Interrupt(InterruptType.TIMER)) is False


def test_mask_and_is_masked():
    iv = InterruptVector()
    iv.register(InterruptType.TIMER, lambda i: None)
    iv.mask(InterruptType.TIMER)
    assert iv.is_masked(InterruptType.TIMER) is True
    assert iv.fire(Interrupt(InterruptType.TIMER)) is False
    iv.unmask(InterruptType.TIMER)
    assert iv.is_masked(InterruptType.TIMER) is False
    assert iv.fire(Interrupt(InterruptType.TIMER)) is True


def test_mask_unrelated_vector_untouched():
    iv = InterruptVector()
    iv.mask(InterruptType.TIMER)
    assert iv.is_masked(InterruptType.CUSTOM) is False


def test_handler_raising_returns_false():
    iv = InterruptVector()

    def boom(i):
        raise RuntimeError("boom")

    iv.register(InterruptType.TIMER, boom)
    assert iv.fire(Interrupt(InterruptType.TIMER)) is False


def test_history_records_all_fires():
    iv = InterruptVector()
    iv.register(InterruptType.TIMER, lambda i: None)
    for _ in range(5):
        iv.fire(Interrupt(InterruptType.TIMER))
    assert len(iv.history) == 5
    assert all(h.vector == InterruptType.TIMER for h in iv.history)


def test_history_truncation():
    iv = InterruptVector()
    iv._max_history = 3
    for n in range(5):
        iv.fire(Interrupt(InterruptType.TIMER, source_pid=n))
    assert len(iv.history) == 3
    assert [h.source_pid for h in iv.history] == [2, 3, 4]


def test_enqueue_dequeue_by_priority():
    iv = InterruptVector()
    iv.enqueue(Interrupt(InterruptType.TIMER, priority=10))
    iv.enqueue(Interrupt(InterruptType.CUSTOM, priority=0))
    iv.enqueue(Interrupt(InterruptType.TIMER, priority=5))
    assert iv.pending_count == 3
    assert iv.dequeue().vector == InterruptType.CUSTOM
    assert iv.dequeue().priority == 5
    assert iv.dequeue().priority == 10
    assert iv.dequeue() is None


def test_process_pending():
    iv = InterruptVector()
    fired = []
    iv.register(InterruptType.TIMER, lambda i: fired.append(i))
    iv.register(InterruptType.CUSTOM, lambda i: fired.append(i))
    iv.enqueue(Interrupt(InterruptType.TIMER))
    iv.enqueue(Interrupt(InterruptType.CUSTOM))
    iv.enqueue(Interrupt(InterruptType.NETWORK_IO))
    assert iv.process_pending() == 2
    assert len(fired) == 2
    assert iv.pending_count == 0


def test_stats():
    iv = InterruptVector()
    iv.register(InterruptType.TIMER, lambda i: None)
    iv.mask(InterruptType.MEMORY_FULL)
    iv.enqueue(Interrupt(InterruptType.TIMER))
    stats = iv.stats()
    assert stats["registered_handlers"] == 1
    assert stats["masked_vectors"] == 1
    assert stats["pending_interrupts"] == 1
    assert stats["total_fired"] == 0
    assert stats["handlers"] == ["TIMER"]


def test_manager_convenience_registers():
    m = InterruptManager()
    seen = []
    m.on_inference_done(lambda i: seen.append(("inf", i)))
    m.on_training_step(lambda i: seen.append(("train", i)))
    m.on_process_done(lambda i: seen.append(("proc", i)))
    m.on_device_error(lambda i: seen.append(("dev", i)))
    m.on_memory_full(lambda i: seen.append(("mem", i)))

    m.signal_inference_done(pid=1, result="r1")
    m.signal_process_done(pid=2, result="r2")
    m.signal_device_error(pid=3, error="oops")

    assert seen[0][0] == "inf"
    assert seen[0][1].source_pid == 1
    assert seen[1][0] == "proc"
    assert seen[1][1].data == "r2"
    assert seen[2][0] == "dev"
    assert seen[2][1].data == "oops"

    assert m.stats()["registered_handlers"] == 5
