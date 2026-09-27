"""
Tests for interactive VM console sessions.

Covers the shell REPL kernel (direct, headless driver), the HTTP session
endpoints (create/input/delete/errors via TestClient), and the SSE stream
(iterated directly — TestClient cannot consume an infinite SSE stream).
"""

import asyncio
import json
import os
import sys
import time

# Ensure the server directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_all_handlers
from routers.vm import (
    _SESSIONS,
    VMConsoleCreateRequest,
    VMConsoleInputRequest,
    console_session_input,
    console_session_stream,
    create_console_session,
)
from routers.vm import (
    router as vm_router,
)

test_app = FastAPI()
test_app.include_router(vm_router)
register_all_handlers(test_app)


def _reset_sessions() -> None:
    for session in list(_SESSIONS.values()):
        session.closed = True
        try:
            if session._task is not None:
                session._task.cancel()
        except Exception:
            pass
    _SESSIONS.clear()


def _wait_history(session_id: str, needle: str, timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        text = "".join(list(_SESSIONS[session_id].history))
        if needle in text:
            return text
        time.sleep(0.05)
    return text


class TestShellKernel:
    """The shell REPL builtin assembles and behaves correctly."""

    def test_shell_source_assembles(self):
        from vm_builtins import get_builtin

        from domain.shell._internal.vm import X86Assembler

        code = X86Assembler().assemble(get_builtin("shell"))
        assert len(code) > 0

    def test_shell_repl_direct_flow(self):
        from vm_builtins import get_builtin

        from domain.shell._internal.vm import X86VirtualSystem
        from domain.shell._internal.vm_permissions import Role

        vs = X86VirtualSystem(memory_size=0x100000)
        pid = vs.spawn("web_user", get_builtin("shell"))
        assert pid is not None
        vs._syscall._rbac.assign(pid, Role.USER)
        vs.scheduler.start(vs.cpu)
        vs.scheduler.current.restore_to_cpu(vs.cpu)

        out: list[str] = []
        original = vs._syscall._sys_write

        def capture(fd, addr, count):
            if fd in (1, 2):
                out.append(
                    bytes(vs.cpu._read8(addr + i) for i in range(count)).decode("ascii", "replace")
                )
                return count
            return original(fd, addr, count)

        vs._syscall._sys_write = capture

        def feed(text: str) -> None:
            for ch in text:
                if ch in ("\b", "\x7f"):
                    vs.cpu.push_scancode(0x0E)
                else:
                    vs.cpu.push_key(ch)

        def run_steps(n: int) -> bool:
            for _ in range(n):
                vs.cpu.transfer_key()
                if not vs.cpu.step():
                    return True
            return False

        feed("help\n")
        assert not run_steps(40000)
        text = "".join(out)
        assert "sloughvm>" in text
        assert (
            "commands: help, ls, cat <file>, uname, pid, echo <text>, train, train-status, train-result, about, clear, halt"
            in text
        )

        out.clear()
        feed("echo hello\n")
        run_steps(40000)
        feed("xyzzy\n")
        run_steps(40000)
        text = "".join(out)
        assert "hello\n" in text
        assert "unknown command" in text

        out.clear()
        feed("echx")  # typo
        run_steps(40000)
        feed("\x7f")  # backspace deletes the x
        run_steps(40000)
        feed("o ok\n")
        run_steps(40000)
        text = "".join(out)
        assert "unknown command" not in text
        assert "ok\n" in text

        out.clear()
        feed("halt\n")
        halted = run_steps(40000)
        assert halted, "halt command must terminate the process"

    def test_shell_fs_and_system_commands(self):
        from vm_builtins import get_builtin

        from domain.shell._internal.vm import X86VirtualSystem
        from domain.shell._internal.vm_permissions import Role

        vs = X86VirtualSystem(memory_size=0x100000)
        vs._fs.write("motd", b"hello from flatfs")
        pid = vs.spawn("web_user", get_builtin("shell"))
        assert pid is not None
        vs._syscall._rbac.assign(pid, Role.USER)
        vs.scheduler.start(vs.cpu)
        vs.scheduler.current.restore_to_cpu(vs.cpu)

        out: list[str] = []
        original = vs._syscall._sys_write

        def capture(fd, addr, count):
            if fd in (1, 2):
                out.append(
                    bytes(vs.cpu._read8(addr + i) for i in range(count)).decode("ascii", "replace")
                )
                return count
            return original(fd, addr, count)

        vs._syscall._sys_write = capture

        def feed(text: str) -> None:
            for ch in text:
                if ch in ("\b", "\x7f"):
                    vs.cpu.push_scancode(0x0E)
                else:
                    vs.cpu.push_key(ch)

        def run_steps(n: int) -> bool:
            for _ in range(n):
                vs.cpu.transfer_key()
                if not vs.cpu.step():
                    return True
            return False

        def check(cmd: str, expect: str) -> None:
            out.clear()
            feed(cmd)
            assert not run_steps(60000), f"unexpected halt running {cmd!r}"
            text = "".join(out)
            assert expect in text, f"{cmd!r} -> {text!r}"

        check("ls\n", "motd")
        check("uname\n", "SloughOS sloughvm 0.1.0 #1 SMP i686")
        check("pid\n", str(pid))
        check("cat motd\n", "hello from flatfs")
        check("cat nofile\n", "cat: no such file")
        check("cat\n", "usage: cat <file>")
        check("cat   motd\n", "hello from flatfs")
        check("catalog\n", "unknown command")

        assert vs._fs.delete("motd")
        check("ls\n", "(no files)")

    def test_shell_train_commands(self, monkeypatch):
        """REPL train/train-status/train-result with a fake training bridge."""
        from vm_builtins import get_builtin

        from domain.shell._internal import vm_training_bridge
        from domain.shell._internal.vm import X86VirtualSystem
        from domain.shell._internal.vm_permissions import Role

        class FakeBridge:
            def __init__(self):
                self.start_ret = 5
                self.status_val = "completed"
                self.result = '{"final_loss":0.5}'
                self.start_calls = 0
                self.last_cfg: str | None = None

            def start(self, cfg: str) -> int:
                self.start_calls += 1
                self.last_cfg = cfg
                return self.start_ret

            def status(self, job_id: int) -> dict:
                return {"status": self.status_val}

            def get_result_json(self, job_id: int) -> str | None:
                return self.result

        fake = FakeBridge()
        monkeypatch.setattr(vm_training_bridge, "get_bridge", lambda: fake)

        def make_shell(role):
            vs = X86VirtualSystem(memory_size=0x100000)
            pid = vs.spawn("web_user", get_builtin("shell"))
            assert pid is not None
            vs._syscall._rbac.assign(pid, role)
            vs.scheduler.start(vs.cpu)
            vs.scheduler.current.restore_to_cpu(vs.cpu)
            out: list[str] = []
            original = vs._syscall._sys_write

            def capture(fd, addr, count):
                if fd in (1, 2):
                    out.append(
                        bytes(vs.cpu._read8(addr + i) for i in range(count)).decode(
                            "ascii", "replace"
                        )
                    )
                    return count
                return original(fd, addr, count)

            vs._syscall._sys_write = capture

            def feed(text: str) -> None:
                for ch in text:
                    if ch in ("\b", "\x7f"):
                        vs.cpu.push_scancode(0x0E)
                    else:
                        vs.cpu.push_key(ch)

            def run_steps(n: int) -> bool:
                for _ in range(n):
                    vs.cpu.transfer_key()
                    if not vs.cpu.step():
                        return True
                return False

            def check(cmd: str, expect: str) -> None:
                out.clear()
                feed(cmd)
                assert not run_steps(60000), f"unexpected halt running {cmd!r}"
                text = "".join(out)
                assert expect in text, f"{cmd!r} -> {text!r}"

            return check

        check = make_shell(Role.ADMIN)
        check("train-status\n", "no training job yet")
        check("train\n", "started job 5")
        check("train-status\n", "status: completed")
        check("train-result\n", '{"final_loss":0.5}')
        check("help\n", "train, train-status, train-result")
        fake.start_ret = -1
        check("train\n", "train: could not start job")
        assert fake.last_cfg is not None and '"dataset":"shakespeare"' in fake.last_cfg

        calls_before = fake.start_calls
        check_user = make_shell(Role.USER)
        check_user("train\n", "permission denied (ADMIN role required)")
        check_user("train-result\n", "permission denied (ADMIN role required)")
        assert fake.start_calls == calls_before, "bridge must not run without TRAINING perm"

    def test_session_train_denied_over_http(self):
        with TestClient(test_app) as client:
            _reset_sessions()
            resp = client.post("/vm/session", json={"role": "user"})
            assert resp.status_code == 200
            session_id = resp.json()["data"]["session_id"]
            resp = client.post(f"/vm/session/{session_id}/input", json={"text": "train\n"})
            assert resp.status_code == 200
            text = _wait_history(session_id, "permission denied", timeout=8.0)
            assert "permission denied (ADMIN role required)" in text
            client.delete(f"/vm/session/{session_id}")


class TestVMConsoleHTTP:
    """Session lifecycle over HTTP: create, input, delete, errors."""

    def test_session_lifecycle(self):
        _reset_sessions()
        with TestClient(test_app) as client:
            resp = client.post("/vm/session", json={"role": "user"})
            assert resp.status_code == 200
            session_id = resp.json()["data"]["session_id"]
            assert session_id

            text = _wait_history(session_id, "sloughvm>")
            assert "sloughvm>" in text

            resp = client.post(f"/vm/session/{session_id}/input", json={"text": "help\n"})
            assert resp.status_code == 200
            assert resp.json()["data"]["accepted"] == 5

            text = _wait_history(session_id, "commands:")
            assert (
                "commands: help, ls, cat <file>, uname, pid, echo <text>, train, train-status, train-result, about, clear, halt"
                in text
            )

            resp = client.post(f"/vm/session/{session_id}/input", json={"text": "echo zebra9x\n"})
            assert resp.status_code == 200

            text = _wait_history(session_id, "zebra9x")
            assert "zebra9x\n" in text

            resp = client.delete(f"/vm/session/{session_id}")
            assert resp.status_code == 200
            assert resp.json()["data"]["closed"] is True

            resp = client.get(f"/vm/session/{session_id}/stream")
            assert resp.status_code == 404

    def test_session_backspace_via_input_endpoint(self):
        _reset_sessions()
        with TestClient(test_app) as client:
            session_id = client.post("/vm/session", json={"role": "user"}).json()["data"][
                "session_id"
            ]
            client.post(f"/vm/session/{session_id}/input", json={"text": "echx"})
            client.post(f"/vm/session/{session_id}/input", json={"text": "\x7f"})
            client.post(f"/vm/session/{session_id}/input", json={"text": "o ok\n"})

            text = _wait_history(session_id, "ok\n\nsloughvm>")
            assert "unknown command" not in text
            assert "ok\n\nsloughvm>" in text
            client.delete(f"/vm/session/{session_id}")

    def test_session_not_found(self):
        _reset_sessions()
        with TestClient(test_app) as client:
            resp = client.post("/vm/session/nope/input", json={"text": "x"})
            assert resp.status_code == 404
            resp = client.get("/vm/session/nope/stream")
            assert resp.status_code == 404
            resp = client.delete("/vm/session/nope")
            assert resp.status_code == 404

    def test_session_cap_returns_429(self):
        _reset_sessions()
        with TestClient(test_app) as client:
            ids = []
            try:
                for _ in range(8):
                    resp = client.post("/vm/session", json={"role": "user"})
                    assert resp.status_code == 200
                    ids.append(resp.json()["data"]["session_id"])
                resp = client.post("/vm/session", json={"role": "user"})
                assert resp.status_code == 429
            finally:
                for sid in ids:
                    client.delete(f"/vm/session/{sid}")
                _reset_sessions()


class TestVMConsoleStream:
    """SSE stream behavior — iterated directly on a pytest-managed loop."""

    async def _open(self, session_id: str):
        resp = await console_session_stream(session_id, auth_user={})
        return resp.body_iterator

    async def _read_output(self, it, predicate, limit: int = 60) -> str:
        chunks: list[str] = []
        for _ in range(limit):
            chunk = await asyncio.wait_for(anext(it), 5.0)
            payload = json.loads(chunk[len("data: ") :])
            if payload.get("stream") != "vm_console":
                continue
            if payload.get("phase") == "output":
                chunks.append(str(payload.get("data", {}).get("text", "")))
            if predicate("".join(chunks)):
                return "".join(chunks)
        return "".join(chunks)

    async def _read_status(self, it, limit: int = 60) -> dict:
        for _ in range(limit):
            chunk = await asyncio.wait_for(anext(it), 5.0)
            payload = json.loads(chunk[len("data: ") :])
            if payload.get("stream") != "vm_console":
                continue
            if payload.get("phase") == "status":
                return payload
        return {}

    async def test_stream_backlog_then_live_output_then_complete(self):
        _reset_sessions()
        created = await create_console_session(VMConsoleCreateRequest(role="user"), auth_user={})
        session_id = created["data"]["session_id"]

        # Wait for the prompt to reach history (pump's first chunk)
        for _ in range(100):
            if "sloughvm>" in "".join(list(_SESSIONS[session_id].history)):
                break
            await asyncio.sleep(0.05)

        it = await self._open(session_id)

        # Backlog must carry the prompt
        text = await self._read_output(it, lambda t: "sloughvm>" in t)
        assert "sloughvm>" in text

        # Live output: input help, expect the commands line
        await console_session_input(session_id, VMConsoleInputRequest(text="help\n"), auth_user={})
        text = await self._read_output(it, lambda t: "commands:" in t)
        assert (
            "commands: help, ls, cat <file>, uname, pid, echo <text>, train, train-status, train-result, about, clear, halt"
            in text
        )

        # halt → status complete ends the stream
        await console_session_input(session_id, VMConsoleInputRequest(text="halt\n"), auth_user={})
        payload = await self._read_status(it)
        assert payload.get("status") in ("complete", "closed")

        await _SESSIONS[session_id].close()
        _SESSIONS.pop(session_id, None)

    async def test_stream_rejects_unknown_session(self):
        from domain.infrastructure._internal.errors import AppError

        _reset_sessions()
        try:
            await console_session_stream("missing-id", auth_user={})
            raise AssertionError("expected AppError")
        except AppError:
            pass
