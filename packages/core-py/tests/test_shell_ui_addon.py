"""
Tests for the shell_ui addon — shell spawn, kernel shell command loop,
VM process syscall handling, and run_program device wiring.
"""

import types

from domains.shell.kernel import Kernel
from domains.shell.addons import shell_ui

SYS_SRC = "\n".join([
    "LOAD_CONST R7, 111", "LOAD_CONST R0, 42", "SYSCALL",
    "LOAD_CONST R7, 110", "SYSCALL",
    "LOAD_CONST R7, 20", "LOAD_CONST R0, 4", "SYSCALL",
    "LOAD_CONST R7, 21", "SYSCALL",
    "LOAD_CONST R7, 200", "SYSCALL",
    "LOAD_CONST R7, 201", "SYSCALL",
    "LOAD_CONST R7, 999", "SYSCALL",
    "LOAD_CONST R7, 2", "SYSCALL",
    "HALT",
])


def _shell_outputs(k, lines):
    """Run the kernel shell entry with a command feed, capturing stdout."""
    outputs = []
    feed = iter(lines)

    def stdin_fn():
        return next(feed, None)

    proc = k.spawn_kernel_shell(stdin_fn=stdin_fn, stdout_fn=outputs.append)
    proc.entry()
    return outputs


# ---------------------------------------------------------------------------
# spawn_shell
# ---------------------------------------------------------------------------

class TestSpawnShell:
    def test_spawn_shell_with_given_class(self):
        k = Kernel()
        k.boot()
        calls = []

        class FakeShell:
            def __init__(self, **kw):
                calls.append(kw)

            def run(self):
                calls.append("run")

        proc = k.spawn_shell(shell_class=FakeShell, greeting="hi")
        assert proc.name == "shell"
        proc.entry()
        assert calls == [{"greeting": "hi"}, "run"]

    def test_spawn_shell_default_class_lazy_import(self, monkeypatch):
        import domains.shell.repl as repl

        ran = []

        class FakeShell:
            def __init__(self, **kw):
                pass

            def run(self):
                ran.append("ran")

        monkeypatch.setattr(repl, "ShellREPL", FakeShell)
        k = Kernel()
        k.boot()
        proc = k.spawn_shell()
        proc.entry()
        assert ran == ["ran"]


# ---------------------------------------------------------------------------
# spawn_kernel_shell command loop
# ---------------------------------------------------------------------------

class TestKernelShell:
    def test_command_dispatch_without_filesystem(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, [
            "help",
            "meminfo",
            "procs",
            "run",
            "ls",
            "cat",
            "cat foo.txt",
            "write",
            "write a.txt x",
            "run missing",
            "ls",
            "boguscmd",
            "",
            "halt",
        ])
        joined = "\n".join(out)
        assert "commands: help, meminfo, procs, run, ls, cat, write, halt" in joined
        assert "blocks: 0" in joined
        assert "pid=" in joined
        assert "usage: run <program.asm>" in joined
        assert "no filesystem mounted" in joined
        assert "error:" in joined
        assert "(empty)" in joined
        assert "usage: cat <file>" in joined
        assert "usage: write <file> <content>" in joined
        assert "unknown: boguscmd" in joined
        assert "shutting down..." in joined

    def test_no_stdin_breaks_immediately(self):
        k = Kernel()
        k.boot()
        outputs = []
        proc = k.spawn_kernel_shell(stdout_fn=outputs.append)
        proc.entry()
        assert outputs == ["ai-compteur> "]

    def test_filesystem_commands(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        fs.write("hello.txt", b"hello world")
        fs.write("test.asm", b"HALT")
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, [
            "ls",
            "cat hello.txt",
            "cat missing.txt",
            "write out.txt data",
            "run test",
            "halt",
        ])
        joined = "\n".join(out)
        assert "hello.txt" in joined
        assert "test.asm" in joined
        assert "hello world" in joined
        assert "error:" in joined
        assert "wrote 4 bytes to out.txt" in joined
        assert "loading test..." in joined
        assert "done (1 steps)" in joined

    def test_write_error_path(self):
        k = Kernel()
        k.boot()

        class FailingFS:
            def write(self, name, data):
                raise OSError("disk full")

        k._fs = FailingFS()
        k._block_device = object()
        out = _shell_outputs(k, ["write a.txt x", "halt"])
        assert "error: disk full" in "\n".join(out)


# ---------------------------------------------------------------------------
# spawn_vm_process syscall handling
# ---------------------------------------------------------------------------

class TestSpawnVMProcess:
    def test_syscalls_with_stdout_and_stdin(self):
        k = Kernel()
        k.boot()
        outputs = []
        proc = k.spawn_vm_process(
            "vmtest", SYS_SRC, use_syscalls=True,
            stdin_fn=lambda: "typed-input",
            stdout_fn=outputs.append,
        )
        proc.entry()
        assert "42\n" in outputs

    def test_syscalls_without_stdout_or_stdin(self):
        k = Kernel()
        k.boot()
        proc = k.spawn_vm_process("vmtest2", SYS_SRC, use_syscalls=True)
        proc.entry()
        assert proc.metadata["output_log"] == ["42"]

    def test_vm_process_without_syscalls(self):
        k = Kernel()
        k.boot()
        outputs = []
        proc = k.spawn_vm_process(
            "vmplain", "LOAD_CONST R0, 5\nHALT", stdout_fn=outputs.append,
        )
        proc.entry()
        assert outputs == []


# ---------------------------------------------------------------------------
# run_program device wiring
# ---------------------------------------------------------------------------

class TestRunProgram:
    def test_no_devices_registered(self):
        k = Kernel()
        k.boot()
        result = k.run_program("HALT")
        assert result["steps"] == 1
        assert result["output"] == []
        assert result["regs"] == {}

    def test_wires_table_devices(self):
        k = Kernel()
        k.boot()
        k._devices = types.SimpleNamespace(
            _table=types.SimpleNamespace(_devices={"d": object()}),
        )
        result = k.run_program("HALT")
        assert result["steps"] == 1

    def test_wires_plain_devices_dict_and_trace(self):
        k = Kernel()
        k.boot()
        k._devices = types.SimpleNamespace(_devices={"d": object()})
        result = k.run_program("HALT", trace=True)
        assert result["steps"] == 1
        assert len(result["trace"]) == 1


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

class TestSetup:
    def test_setup_installs_functions(self):
        k = Kernel()
        shell_ui.setup(k)
        assert k.has_addon("shell_ui")
        for fn in ("spawn_shell", "spawn_kernel_shell", "spawn_vm_process", "run_program"):
            assert callable(getattr(k, fn))
