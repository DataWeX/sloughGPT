"""Tasks tests — steps, preconditions, optional failures, reporter."""

import asyncio
import json

from arken.core.reporter import Reporter
from arken.core.task import Task, TaskResult, TaskStatus, TaskStep


def run(coro):
    return asyncio.run(coro)


async def ok_action(ctx):
    ctx["ran"] = ctx.get("ran", 0) + 1


async def boom_action(ctx):
    raise RuntimeError("kaput")


class TestTask:
    def test_all_steps_pass(self):
        t = Task("t", steps=[TaskStep("a", ok_action), TaskStep("b", ok_action)])
        r = run(t.execute({}))
        assert r.status == TaskStatus.PASSED
        assert r.steps_passed == 2

    def test_failure_stops_and_reports(self):
        t = Task(
            "t",
            steps=[
                TaskStep("a", ok_action),
                TaskStep("b", boom_action),
                TaskStep("c", ok_action),
            ],
        )
        ctx: dict = {}
        r = run(t.execute(ctx))
        assert r.status == TaskStatus.FAILED
        assert "b" in r.error
        assert ctx["ran"] == 1  # step c never ran

    def test_optional_failure_continues(self):
        t = Task(
            "t",
            steps=[
                TaskStep("b", boom_action, optional=True),
                TaskStep("c", ok_action),
            ],
        )
        r = run(t.execute({}))
        assert r.status == TaskStatus.PASSED
        assert r.steps_failed == 1
        assert r.steps_passed == 1

    def test_precondition_false_skips(self):
        t = Task("t", steps=[TaskStep("a", ok_action, precondition=lambda ctx: False)])
        r = run(t.execute({}))
        assert r.status == TaskStatus.PASSED
        assert r.steps_skipped == 1
        assert r.steps_passed == 0

    def test_async_precondition(self):
        async def pre(ctx):
            return True

        t = Task("t", steps=[TaskStep("a", ok_action, precondition=pre)])
        assert run(t.execute({})).steps_passed == 1

    def test_result_to_dict(self):
        r = TaskResult(task_name="t", status=TaskStatus.FAILED, error="x")
        d = r.to_dict()
        assert d["task"] == "t" and d["status"] == "failed"


class TestReporter:
    def _reporter(self):
        rep = Reporter("R")
        rep.add(TaskResult(task_name="a", status=TaskStatus.PASSED))
        rep.add(TaskResult(task_name="b", status=TaskStatus.FAILED, error="bad"))
        return rep

    def test_terminal(self):
        out = self._reporter().report("terminal")
        assert "[PASS] a" in out and "[FAIL] b" in out

    def test_markdown(self):
        out = self._reporter().report("markdown")
        assert "| a | passed |" in out

    def test_json(self):
        data = json.loads(self._reporter().report("json"))
        assert data["passed"] == 1 and data["failed"] == 1

    def test_summary(self):
        assert self._reporter().summary() == {"total": 2, "passed": 1, "failed": 1}
