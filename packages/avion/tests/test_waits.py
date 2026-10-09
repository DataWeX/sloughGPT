"""Waits tests — settle detection, text waits, timeouts."""

import asyncio

from avion.waits import SmartWaiter, WaitCondition, WaitConfig


def run(coro):
    return asyncio.run(coro)


class ScriptedBackend:
    """evaluate() returns scripted values per call; counts polls."""

    def __init__(self, script, default=0):
        self.script = list(script)
        self.default = default
        self.polls = 0

    async def evaluate(self, expression):
        self.polls += 1
        if self.script:
            return self.script.pop(0)
        return self.script_default(expression)

    def script_default(self, expression):
        return self.default

    async def wait_for_timeout(self, ms):
        pass


class TestSettled:
    def test_immediate_settle(self):
        w = SmartWaiter(ScriptedBackend([0, 0]))
        r = run(w.wait())
        assert r.success is True

    def test_waits_for_spinner_gone(self):
        w = SmartWaiter(ScriptedBackend([2, 1, 0, 0]))
        r = run(w.wait())
        assert r.success is True
        assert w._backend.polls >= 3

    def test_timeout_when_never_settles(self):
        w = SmartWaiter(ScriptedBackend([5] * 100, default=5))
        r = run(w.wait(WaitConfig(timeout=0.05, poll_ms=5)))
        assert r.success is False
        assert "timeout" in r.error

    def test_text_visible(self):
        backend = ScriptedBackend(["nope", "hello world"])
        w = SmartWaiter(backend)
        r = run(w.wait_for_text("hello", timeout=1.0))
        assert r.success is True

    def test_text_gone(self):
        backend = ScriptedBackend(["loading...", "done"])
        w = SmartWaiter(backend)
        r = run(
            w.wait(WaitConfig(conditions=[WaitCondition.TEXT_GONE], text="loading...", timeout=1.0))
        )
        assert r.success is True

    def test_evaluate_error_retries_then_times_out(self):
        class Broken:
            async def evaluate(self, e):
                raise RuntimeError("dead")

            async def wait_for_timeout(self, ms):
                pass

        r = run(SmartWaiter(Broken()).wait(WaitConfig(timeout=0.05, poll_ms=5)))
        assert r.success is False
        assert "dead" in r.error
