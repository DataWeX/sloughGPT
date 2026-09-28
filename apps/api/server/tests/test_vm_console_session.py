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
            "commands: help, ls, cat <file>, cp <src> <dst>, grep <pattern> <file>, wc <file>, head <file> [n], tail <file> [n], uname, pid, echo <text> [> <file>], write <file> <text>, train, train-status, train-result, about, clear, halt"
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

    def test_shell_cp_command(self):
        """REPL cp copies a file; missing source and usage paths report errors."""
        from vm_builtins import get_builtin

        from domain.shell._internal.vm import X86VirtualSystem
        from domain.shell._internal.vm_permissions import Role

        vs = X86VirtualSystem(memory_size=0x100000)
        vs._fs.write("orig", b"payload-data")
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

        check("cp orig copy\n", "copied orig -> copy")
        assert vs._fs.exists("copy")
        assert vs._fs.read("copy").rstrip(b"\x00") == b"payload-data"
        check("cat copy\n", "payload-data")
        check("cp orig truncated\n", "copied orig -> truncated")
        check("cp nofile x\n", "cp: no such file")
        assert not vs._fs.exists("x")
        check("cp\n", "usage: cp <src> <dst>")
        check("cp orig\n", "usage: cp <src> <dst>")
        check("cpfoo\n", "unknown command")

    def test_shell_grep_command(self):
        """REPL grep prints matching lines; usage/missing-file/silent paths report correctly."""
        from vm_builtins import get_builtin

        from domain.shell._internal.vm import X86VirtualSystem
        from domain.shell._internal.vm_permissions import Role

        vs = X86VirtualSystem(memory_size=0x100000)
        vs._fs.write("multi", b"alpha beta\ngamma delta\nbeta two")
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

        check("grep beta multi\n", "alpha beta")
        text = "".join(out)
        assert "beta two" in text, text
        assert "gamma delta" not in text, text
        out.clear()
        feed("grep zzz multi\n")
        assert not run_steps(60000)
        text = "".join(out)
        assert "alpha beta" not in text and "gamma delta" not in text, text
        check("grep beta\n", "usage: grep <pattern> <file>")
        check("grep beta nofile\n", "grep: no such file")
        check("grep\n", "usage: grep <pattern> <file>")
        check("grepx\n", "unknown command")

    def test_shell_echo_redirect(self):
        """REPL echo 'text > file' writes the file; shift symbols survive input."""
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

        # '>' and '%' have no unshifted set-1 scancode — they used to be
        # dropped before ever reaching the REPL.
        feed("echo hi > greet.txt\n")
        assert not run_steps(60000)
        assert vs._fs.exists("greet.txt"), "redirect target not created"
        # FlatFS stores sector-padded; compare up to the first NUL.
        content = vs._fs.read("greet.txt").split(b"\x00")[0]
        assert content == b"hi\n", content

        # No-space form exercises the same split path.
        feed("echo yo > y.txt\n")
        assert not run_steps(60000)
        content = vs._fs.read("y.txt").split(b"\x00")[0]
        assert content == b"yo\n", content

        # Plain echo with a shift symbol prints verbatim (no redirect):
        # appears once as typed echo-back, once as command output.
        out.clear()
        feed("echo 100% sure\n")
        assert not run_steps(60000)
        text = "".join(out)
        assert text.count("100% sure") >= 2, text

        # The redirected text is not printed to the console (sh semantics):
        # only the typed line echoes back, then the next prompt.
        out.clear()
        feed("echo secret > hidden.txt\n")
        assert not run_steps(60000)
        text = "".join(out)
        after_prompt = text.rsplit("sloughvm> ", 1)[-1]
        assert "secret" not in after_prompt, text

        # Redirect through cat round-trips.
        check("cat greet.txt\n", "hi")

        # Usage paths: missing target and extra args.
        check("echo hi >\n", "usage: echo <text> > <file>")
        check("echo hi > f.txt extra\n", "usage: echo <text> > <file>")

    def test_shell_wc_head_tail(self):
        """REPL wc/head/tail count and slice files; usage + missing-file paths."""
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

        data = b"alpha beta\ngamma delta epsilon\n\nomega\n"
        vs._fs.write("lines.txt", data)
        nlines = data.count(b"\n")
        nwords = len(data.split())
        nbytes = len(data)

        # wc counts logical bytes — FlatFS sector padding must not count.
        check("wc lines.txt\n", f"{nlines} lines, {nwords} words, {nbytes} bytes")

        # head: default 10 (file has 4 lines), explicit n, n=1.
        check("head lines.txt\n", data.decode())
        check("head lines.txt 2\n", "alpha beta\ngamma delta epsilon\n")
        check("head lines.txt 1\n", "alpha beta\n")

        # tail: last line, last two (crossing the blank line), default.
        check("tail lines.txt 1\n", "omega\n")
        check("tail lines.txt 2\n", "\nomega\n")
        check("tail lines.txt\n", data.decode())

        # wc over an echo-redirect file (12 bytes, one line, two words).
        out.clear()
        feed("echo hello world > hw.txt\n")
        assert not run_steps(60000)
        check("wc hw.txt\n", "1 lines, 2 words, 12 bytes")

        # Empty file → all zeros.
        vs._fs.write("empty.txt", b"")
        check("wc empty.txt\n", "0 lines, 0 words, 0 bytes")

        # head/tail 0 print nothing after the command.
        out.clear()
        feed("head lines.txt 0\n")
        assert not run_steps(60000)
        text = "".join(out)
        assert "alpha" not in text.rsplit("sloughvm> ", 1)[-1], text

        # Missing files, usage, and unknown-command paths.
        check("wc nope.txt\n", "wc: no such file")
        check("wc\n", "usage: wc <file>")
        check("wc lines.txt x\n", "usage: wc <file>")
        check("head nope.txt\n", "head: no such file")
        check("head lines.txt x\n", "usage: head <file> [n]")
        check("tail nope.txt\n", "tail: no such file")
        check("tail\n", "usage: tail <file> [n]")
        check("wcz lines.txt\n", "unknown command")
        check("headshot\n", "unknown command")

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
                "commands: help, ls, cat <file>, cp <src> <dst>, grep <pattern> <file>, wc <file>, head <file> [n], tail <file> [n], uname, pid, echo <text> [> <file>], write <file> <text>, train, train-status, train-result, about, clear, halt"
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
            "commands: help, ls, cat <file>, cp <src> <dst>, grep <pattern> <file>, wc <file>, head <file> [n], tail <file> [n], uname, pid, echo <text> [> <file>], write <file> <text>, train, train-status, train-result, about, clear, halt"
            in text
        )

        # halt → status complete ends the stream
        await console_session_input(session_id, VMConsoleInputRequest(text="halt\n"), auth_user={})
        payload = await self._read_status(it)
        assert payload.get("status") in ("complete", "closed")

        await _SESSIONS[session_id].close()
        _SESSIONS.pop(session_id, None)

    async def test_stream_reconnect_resends_full_backlog(self):
        """A second subscriber gets complete scrollback, then live output (reconnect path)."""
        _reset_sessions()
        created = await create_console_session(VMConsoleCreateRequest(role="user"), auth_user={})
        session_id = created["data"]["session_id"]
        try:
            for _ in range(100):
                if "sloughvm>" in "".join(_SESSIONS[session_id].history):
                    break
                await asyncio.sleep(0.05)

            # First subscriber takes the boot backlog and one live round-trip.
            it1 = await self._open(session_id)
            text1 = await self._read_output(it1, lambda t: "sloughvm>" in t)
            assert "sloughvm>" in text1
            await console_session_input(
                session_id, VMConsoleInputRequest(text="echo recon\n"), auth_user={}
            )
            text1 = await self._read_output(it1, lambda t: "recon" in t)
            assert "recon" in text1
            # Client drops the connection.
            await it1.aclose()

            # Reconnect: new subscriber must see the whole scrollback again…
            it2 = await self._open(session_id)
            text2 = await self._read_output(it2, lambda t: "recon" in t)
            assert "sloughvm>" in text2
            assert "recon" in text2
            # …and keep receiving live output after reconnect.
            await console_session_input(
                session_id, VMConsoleInputRequest(text="echo again\n"), auth_user={}
            )
            text2 = await self._read_output(it2, lambda t: "again" in t)
            assert "again" in text2
            await it2.aclose()
        finally:
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
