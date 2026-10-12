"""
Equivalence contract between the standalone fork and the canonical copy.

infra_lib.task_queue is a DELIBERATE fork of
domain/infrastructure/_internal/task_queue.py so the library can ship with
zero internal dependencies — see the fork's module docstring for the
documented differences (num_workers default resolution, EventBus injection,
SSE envelope shape, log extras).

This suite asserts the two stay equivalent on everything ELSE. A failure
here means either drift crept in (resync the copies) or a new divergence was
intended (document it in the fork's docstring AND below).
"""

import inspect

import pytest

canonical = pytest.importorskip(
    "domain.infrastructure._internal.task_queue",
    reason="canonical domain copy not importable (standalone install)",
)

from infra_lib import task_queue as fork  # noqa: E402


def _public(cls) -> set[str]:
    return {n for n in dir(cls) if not n.startswith("_")}


class TestTaskStatusParity:
    def test_same_members_in_same_order(self):
        assert [m.name for m in canonical.TaskStatus] == [m.name for m in fork.TaskStatus]

    def test_same_string_values(self):
        assert [m.value for m in canonical.TaskStatus] == [m.value for m in fork.TaskStatus]

    def test_str_enum_semantics_both_serialize_as_value(self):
        # The historical drift: (str, Enum) vs StrEnum differ in str()/f-string
        # output. Both sides must now render the bare value.
        for member in fork.TaskStatus:
            twin = canonical.TaskStatus[member.name]
            assert str(member) == str(twin) == member.value
            assert f"{member}" == member.value


class TestPublicSurfaceParity:
    def test_taskqueue_surface_fork_may_only_add_event_bus_injection(self):
        canon = _public(canonical.TaskQueue)
        forked = _public(fork.TaskQueue)
        assert forked - canon == {"set_event_bus"}, forked - canon
        assert canon - forked == set(), canon - forked

    def test_num_workers_signature_parity(self):
        for cls in (
            canonical.TaskQueue,
            canonical.InProcessTaskQueue,
            fork.TaskQueue,
            fork.InProcessTaskQueue,
        ):
            p = inspect.signature(cls.__init__).parameters["num_workers"]
            assert p.default is None, f"{cls.__name__}.num_workers default drifted"
            assert "int | None" in str(p.annotation), (
                f"{cls.__name__}.num_workers annotation drifted"
            )


class TestDocumentedDivergences:
    """The differences the fork docstring promises — pin them so they stay deliberate."""

    def test_fork_num_workers_none_resolves_to_two(self):
        # Canonical resolves None via resource_manager; standalone fork maps
        # None to the literal default 2.
        assert fork.WorkerPool(num_workers=None).num_workers == 2

    def test_fork_default_singleton_uses_two(self):
        q = fork.InProcessTaskQueue(num_workers=None)
        assert q._pool.num_workers == 2


class TestBehaviorParity:
    @pytest.mark.asyncio
    async def test_enqueue_start_complete_identical(self):
        results = []

        async def make_queue(impl):
            q = impl.InProcessTaskQueue(num_workers=2)

            async def echo_handler(task):
                results.append(task.payload.get("msg"))
                return task.payload.get("msg")

            q.register_handler("echo", echo_handler)
            return q

        for impl in (canonical, fork):
            q = await make_queue(impl)
            t = impl.Task(name="echo1", task_type="echo", payload={"msg": "hello"})
            await q.enqueue(t)
            await q.start()
            import asyncio

            await asyncio.sleep(0.3)
            await q.stop()
            assert t.status == impl.TaskStatus.COMPLETED
            assert t.result == "hello"

        # Same work, same results, on both implementations.
        assert results == ["hello", "hello"]
