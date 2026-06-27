"""
Tests for the task queue infrastructure (task_queue.py).
"""

import asyncio
import pytest
from domains.infrastructure.task_queue import (
    Task, TaskStatus, Priority,
    InProcessTaskQueue, get_task_queue, set_task_queue,
)


@pytest.fixture
def queue():
    q = InProcessTaskQueue(num_workers=2)
    return q


class TestTask:
    def test_default_fields(self):
        t = Task()
        assert t.id.startswith("task_")
        assert t.status == TaskStatus.PENDING
        assert t.priority == Priority.NORMAL
        assert t.dependencies == []
        assert t.progress == 0.0
        assert t.max_retries == 0
        assert t.pause_event.is_set()

    def test_elapsed_none_when_not_started(self):
        assert Task().elapsed is None

    def test_elapsed_when_running(self):
        import time
        t = Task(started_at=time.time() - 5)
        assert t.elapsed is not None
        assert t.elapsed >= 5.0

    def test_elapsed_when_completed(self):
        import time
        now = time.time()
        t = Task(started_at=now - 10, completed_at=now)
        assert t.elapsed == pytest.approx(10.0, rel=0.1)


@pytest.mark.asyncio
class TestInProcessTaskQueue:
    async def test_enqueue_returns_id(self, queue):
        t = Task(name="test", task_type="echo")
        task_id = await queue.enqueue(t)
        assert task_id == t.id
        assert queue.get_task(task_id) is t
        assert queue.count() == 1

    async def test_enqueue_front_puts_at_front(self, queue):
        t1 = Task(name="first", task_type="echo")
        t2 = Task(name="urgent", task_type="echo")
        await queue.enqueue(t1)
        await queue.enqueue_front(t2)
        assert t2.priority == Priority.CRITICAL

    async def test_cancel_pending(self, queue):
        t = Task(task_type="echo")
        await queue.enqueue(t)
        ok = await queue.cancel(t.id)
        assert ok is True
        assert t.status == TaskStatus.CANCELLED
        assert queue.count(TaskStatus.CANCELLED) == 1

    async def test_cancel_nonexistent(self, queue):
        ok = await queue.cancel("no_such_task")
        assert ok is False

    async def test_cancel_completed_returns_false(self, queue):
        t = Task(task_type="echo")
        await queue.enqueue(t)
        t.status = TaskStatus.COMPLETED
        ok = await queue.cancel(t.id)
        assert ok is False

    async def test_pause_running(self, queue):
        t = Task(task_type="echo")
        await queue.enqueue(t)
        t.status = TaskStatus.RUNNING
        queue._running[t.id] = t
        ok = await queue.pause(t.id)
        assert ok is True
        assert t.status == TaskStatus.PAUSED
        assert t.pause_event.is_set() is False

    async def test_pause_non_running(self, queue):
        t = Task(task_type="echo")
        ok = await queue.pause(t.id)
        assert ok is False

    async def test_resume_paused(self, queue):
        t = Task(task_type="echo")
        await queue.enqueue(t)
        t.status = TaskStatus.PAUSED
        queue._paused[t.id] = t
        ok = await queue.resume(t.id)
        assert ok is True
        assert t.status == TaskStatus.QUEUED
        assert t.pause_event.is_set()

    async def test_resume_non_paused(self, queue):
        t = Task(task_type="echo")
        await queue.enqueue(t)
        ok = await queue.resume(t.id)
        assert ok is False

    async def test_update_progress(self, queue):
        t = Task(task_type="echo")
        t.status = TaskStatus.RUNNING
        queue._tasks[t.id] = t
        await queue.update_progress(t.id, 0.5, "halfway")
        assert t.progress == 0.5
        assert t.progress_msg == "halfway"

    async def test_update_progress_non_running_ignored(self, queue):
        t = Task(task_type="echo")
        queue._tasks[t.id] = t
        await queue.update_progress(t.id, 0.5, "halfway")
        assert t.progress == 0.0

    async def test_list_tasks_all(self, queue):
        t1 = Task(task_type="a")
        t2 = Task(task_type="b")
        await queue.enqueue(t1)
        await queue.enqueue(t2)
        assert len(queue.list_tasks()) == 2

    async def test_list_tasks_filtered(self, queue):
        t = Task(task_type="a")
        await queue.enqueue(t)
        assert len(queue.list_tasks(TaskStatus.QUEUED)) == 1
        assert len(queue.list_tasks(TaskStatus.RUNNING)) == 0

    async def test_handler_execution(self, queue):
        results = []

        async def echo_handler(task: Task):
            results.append(task.payload.get("msg"))
            return task.payload.get("msg")

        queue.register_handler("echo", echo_handler)
        t = Task(name="echo1", task_type="echo", payload={"msg": "hello"})
        await queue.enqueue(t)
        await queue.start()
        await asyncio.sleep(0.3)
        await queue.stop()
        assert results == ["hello"]
        assert t.status == TaskStatus.COMPLETED

    async def test_no_handler_fails_task(self, queue):
        t = Task(task_type="no_handler")
        await queue.enqueue(t)
        await queue.start()
        await asyncio.sleep(0.3)
        await queue.stop()
        assert t.status == TaskStatus.FAILED
        assert "No handler" in (t.error or "")

    async def test_cancel_before_execution(self, queue):
        results = []

        async def slow_handler(task: Task):
            await asyncio.sleep(5)
            results.append("done")

        queue.register_handler("slow", slow_handler)
        t = Task(task_type="slow")
        await queue.enqueue(t)
        await queue.start()
        await asyncio.sleep(0.1)
        ok = await queue.cancel(t.id)
        assert ok is True
        assert t.status == TaskStatus.CANCELLED
        await queue.stop(timeout=1.0)
        assert results == []

    async def test_retry_on_failure(self, queue):
        attempt_count = 0

        async def flaky_handler(task: Task):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("transient error")

        queue.register_handler("flaky", flaky_handler)
        t = Task(task_type="flaky", max_retries=2)
        await queue.enqueue(t)
        await queue.start()
        await asyncio.sleep(2.0)
        await queue.stop()
        assert t.status == TaskStatus.COMPLETED
        assert attempt_count == 3

    async def test_max_retries_exceeded(self, queue):
        attempt_count = 0

        async def always_fails(task: Task):
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("always fails")

        queue.register_handler("fail", always_fails)
        t = Task(task_type="fail", max_retries=1)
        await queue.enqueue(t)
        await queue.start()
        await asyncio.sleep(1.0)
        await queue.stop()
        assert t.status == TaskStatus.FAILED
        assert t.retry_count == 2

    async def test_task_timeout(self, queue):
        async def slow_handler(task: Task):
            await asyncio.sleep(10)

        queue.register_handler("slow", slow_handler)
        t = Task(task_type="slow", timeout=0.2)
        await queue.enqueue(t)
        await queue.start()
        await asyncio.sleep(0.5)
        await queue.stop()
        assert t.status == TaskStatus.FAILED
        assert "Timeout" in (t.error or "")

    async def test_dependency_ordering(self, queue):
        order = []

        async def dep_handler(task: Task):
            order.append(task.name)

        queue.register_handler("dep", dep_handler)
        ta = Task(name="A", task_type="dep")
        tb = Task(name="B", task_type="dep", dependencies=[ta.id])
        await queue.enqueue(tb)
        await queue.enqueue(ta)
        await queue.start()
        await asyncio.sleep(0.5)
        await queue.stop()
        assert order == ["A", "B"]

    async def test_sse_callbacks(self, queue):
        events = []

        def on_event(event: str, task: Task):
            events.append((event, task.id))

        queue.subscribe(on_event)
        t = Task(task_type="echo")

        async def echo_handler(task: Task):
            pass

        queue.register_handler("echo", echo_handler)
        await queue.enqueue(t)
        assert ("enqueued", t.id) in events

    async def test_unsubscribe(self, queue):
        events = []

        def on_event(event: str, task: Task):
            events.append(event)

        queue.subscribe(on_event)
        queue.unsubscribe(on_event)
        t = Task(task_type="echo")
        await queue.enqueue(t)
        assert events == []

    async def test_priority_ordering(self, queue):
        order = []

        async def p_handler(task: Task):
            order.append(task.name)

        queue.register_handler("prio", p_handler)
        low = Task(name="low", task_type="prio", priority=Priority.LOW)
        high = Task(name="high", task_type="prio", priority=Priority.HIGH)
        await queue.enqueue(low)
        await queue.enqueue(high)
        await queue.start()
        await asyncio.sleep(0.5)
        await queue.stop()
        assert order == ["high", "low"]

    async def test_singleton(self):
        q1 = get_task_queue()
        q2 = get_task_queue()
        assert q1 is q2

    async def test_set_singleton(self):
        q = InProcessTaskQueue()
        set_task_queue(q)
        assert get_task_queue() is q


@pytest.mark.asyncio
class TestWorkerPool:
    async def test_start_stop(self):
        from domains.infrastructure.task_queue import WorkerPool
        pool = WorkerPool(num_workers=2)
        await pool.start()
        assert pool.active_workers == 2
        await pool.stop()
        assert pool.active_workers == 0

    async def test_handler_called(self):
        from domains.infrastructure.task_queue import WorkerPool, Task
        results = []

        async def handler(task: Task):
            results.append(task.name)

        pool = WorkerPool(num_workers=1)
        pool.set_handler(handler)
        await pool.start()
        t = Task(name="test", task_type="generic")
        await pool.queue.put(t)
        await asyncio.sleep(0.2)
        await pool.stop()
        assert results == ["test"]
