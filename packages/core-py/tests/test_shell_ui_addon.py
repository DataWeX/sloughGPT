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

    def test_spawn_shell_passes_kwargs(self):
        k = Kernel()
        k.boot()
        captured = {}

        class FakeShell:
            def __init__(self, **kw):
                captured.update(kw)

            def run(self):
                pass

        proc = k.spawn_shell(shell_class=FakeShell, theme="dark", prompt=">>> ")
        proc.entry()
        assert captured["theme"] == "dark"
        assert captured["prompt"] == ">>> "

    def test_spawn_shell_process_name(self):
        k = Kernel()
        k.boot()

        class FakeShell:
            def __init__(self, **kw):
                pass
            def run(self):
                pass

        proc = k.spawn_shell(shell_class=FakeShell)
        assert proc.name == "shell"

    def test_spawn_shell_entry_callable(self):
        k = Kernel()
        k.boot()

        class FakeShell:
            def __init__(self, **kw):
                pass
            def run(self):
                pass

        proc = k.spawn_shell(shell_class=FakeShell)
        assert callable(proc.entry)


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

    def test_help_command(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["help", "halt"])
        joined = "\n".join(out)
        assert "help" in joined
        assert "meminfo" in joined
        assert "procs" in joined
        assert "halt" in joined

    def test_meminfo_command(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["meminfo", "halt"])
        joined = "\n".join(out)
        assert "blocks:" in joined

    def test_procs_command(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["procs", "halt"])
        joined = "\n".join(out)
        assert "pid=" in joined

    def test_empty_line_shows_prompt(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["", "halt"])
        assert out[0] == "ai-compteur> "
        assert "ai-compteur> " in out[1]

    def test_ls_no_filesystem(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["ls", "halt"])
        assert "no filesystem mounted" in "\n".join(out)

    def test_cat_no_filesystem(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["cat foo.txt", "halt"])
        assert "no filesystem mounted" in "\n".join(out)

    def test_write_no_filesystem(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["write a.txt data", "halt"])
        assert "no filesystem mounted" in "\n".join(out)

    def test_cat_no_args(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, ["cat", "halt"])
        assert "usage: cat <file>" in "\n".join(out)

    def test_write_no_args(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, ["write", "halt"])
        assert "usage: write <file> <content>" in "\n".join(out)

    def test_ls_empty_filesystem(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, ["ls", "halt"])
        assert "(empty)" in "\n".join(out)

    def test_exit_command(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["exit"])
        assert "shutting down..." in "\n".join(out)

    def test_quit_command(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["quit"])
        assert "shutting down..." in "\n".join(out)

    def test_unknown_command(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["foobar", "halt"])
        assert "unknown: foobar" in "\n".join(out)

    def test_cat_file_content(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        fs.write("data.txt", b"some data here")
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, ["cat data.txt", "halt"])
        assert "some data here" in "\n".join(out)

    def test_write_and_cat_roundtrip(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, [
            "write test.txt hello world",
            "cat test.txt",
            "halt",
        ])
        joined = "\n".join(out)
        assert "wrote 11 bytes to test.txt" in joined
        assert "hello world" in joined


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

    def test_vm_process_metadata_source(self):
        k = Kernel()
        k.boot()
        src = "LOAD_CONST R0, 1\nHALT"
        proc = k.spawn_vm_process("test", src, use_syscalls=True)
        assert proc.metadata["source"] == src

    def test_vm_process_metadata_output_log(self):
        k = Kernel()
        k.boot()
        proc = k.spawn_vm_process("test", "HALT", use_syscalls=True)
        assert isinstance(proc.metadata["output_log"], list)

    def test_vm_process_name(self):
        k = Kernel()
        k.boot()
        proc = k.spawn_vm_process("myprog", "HALT")
        assert proc.name == "myprog"

    def test_vm_process_priority(self):
        from domains.shell.kernel_process import Priority
        k = Kernel()
        k.boot()
        proc = k.spawn_vm_process("test", "HALT", priority=Priority.HIGH)
        assert proc.priority == Priority.HIGH

    def test_vm_process_entry_callable(self):
        k = Kernel()
        k.boot()
        proc = k.spawn_vm_process("test", "HALT")
        assert callable(proc.entry)


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

    def test_run_program_returns_dict(self):
        k = Kernel()
        k.boot()
        result = k.run_program("HALT")
        assert isinstance(result, dict)
        assert "output" in result
        assert "steps" in result
        assert "trace" in result
        assert "regs" in result

    def test_run_program_with_nonzero_reg(self):
        k = Kernel()
        k.boot()
        result = k.run_program("LOAD_CONST R0, 99\nHALT")
        assert result["regs"]["R0"] == 99

    def test_run_program_steps_count(self):
        k = Kernel()
        k.boot()
        result = k.run_program("LOAD_CONST R0, 1\nLOAD_CONST R1, 2\nHALT")
        assert result["steps"] == 3

    def test_run_program_trace_empty_without_flag(self):
        k = Kernel()
        k.boot()
        result = k.run_program("HALT", trace=False)
        assert result["trace"] == []

    def test_run_program_trace_populated(self):
        k = Kernel()
        k.boot()
        result = k.run_program("HALT", trace=True)
        assert len(result["trace"]) >= 1

    def test_run_program_output_list(self):
        k = Kernel()
        k.boot()
        result = k.run_program("HALT")
        assert isinstance(result["output"], list)

    def test_run_program_regs_dict(self):
        k = Kernel()
        k.boot()
        result = k.run_program("HALT")
        assert isinstance(result["regs"], dict)


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

    def test_setup_adds_to_addons_dict(self):
        k = Kernel()
        shell_ui.setup(k)
        assert "shell_ui" in k._addons

    def test_spawn_shell_bound_to_kernel(self):
        k = Kernel()
        shell_ui.setup(k)

        class FakeShell:
            def __init__(self, **kw):
                pass
            def run(self):
                pass

        proc = k.spawn_shell(shell_class=FakeShell)
        assert proc.name == "shell"

    def test_spawn_kernel_shell_bound_to_kernel(self):
        k = Kernel()
        shell_ui.setup(k)
        proc = k.spawn_kernel_shell()
        assert proc.name == "kernel-shell"

    def test_run_program_bound_to_kernel(self):
        k = Kernel()
        shell_ui.setup(k)
        result = k.run_program("HALT")
        assert result["steps"] == 1

    def test_spawn_vm_process_bound_to_kernel(self):
        k = Kernel()
        shell_ui.setup(k)
        proc = k.spawn_vm_process("test", "HALT")
        assert proc.name == "test"

    def test_setup_idempotent(self):
        k = Kernel()
        shell_ui.setup(k)
        shell_ui.setup(k)
        assert k.has_addon("shell_ui")

    def test_setup_preserves_existing_addons(self):
        k = Kernel()
        k._addons["other"] = True
        shell_ui.setup(k)
        assert "other" in k._addons
        assert "shell_ui" in k._addons


# ---------------------------------------------------------------------------
# Kernel shell — extended command coverage
# ---------------------------------------------------------------------------

class TestKernelShellExtended:
    def test_multiple_files_ls(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        fs.write("a.txt", b"aaa")
        fs.write("b.txt", b"bbb")
        fs.write("c.txt", b"ccc")
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, ["ls", "halt"])
        joined = "\n".join(out)
        assert "a.txt" in joined
        assert "b.txt" in joined
        assert "c.txt" in joined

    def test_write_multiple_words(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, [
            "write test.txt hello beautiful world",
            "cat test.txt",
            "halt",
        ])
        joined = "\n".join(out)
        assert "wrote 21 bytes to test.txt" in joined
        assert "hello beautiful world" in joined

    def test_run_multiple_programs(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        fs.write("p1.asm", b"HALT")
        fs.write("p2.asm", b"HALT")
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, ["run p1", "run p2", "halt"])
        joined = "\n".join(out)
        assert "done (1 steps)" in joined

    def test_cat_nonexistent_file(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, ["cat nope.txt", "halt"])
        assert "error:" in "\n".join(out)

    def test_run_no_filesystem(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["run test", "halt"])
        joined = "\n".join(out)
        assert "loading test..." in joined
        assert "error:" in joined


# ---------------------------------------------------------------------------
# VM process — extended syscall coverage
# ---------------------------------------------------------------------------

class TestSpawnVMProcessExtended:
    def test_vm_console_write_with_stdout_fn(self):
        k = Kernel()
        k.boot()
        outputs = []
        src = "\n".join([
            "LOAD_CONST R7, 111", "LOAD_CONST R0, 42", "SYSCALL",
            "HALT",
        ])
        proc = k.spawn_vm_process(
            "vmtest", src, use_syscalls=True,
            stdout_fn=outputs.append,
        )
        proc.entry()
        assert "42\n" in outputs

    def test_vm_console_write_without_stdout_fn(self):
        k = Kernel()
        k.boot()
        src = "\n".join([
            "LOAD_CONST R7, 111", "LOAD_CONST R0, 42", "SYSCALL",
            "HALT",
        ])
        proc = k.spawn_vm_process("vmtest", src, use_syscalls=True)
        proc.entry()
        assert proc.metadata["output_log"] == ["42"]

    def test_vm_console_read_with_stdin(self):
        k = Kernel()
        k.boot()
        src = "\n".join([
            "LOAD_CONST R7, 111", "LOAD_CONST R7, 110", "SYSCALL",
            "HALT",
        ])
        proc = k.spawn_vm_process(
            "vmtest", src, use_syscalls=True,
            stdin_fn=lambda: "hello",
            stdout_fn=lambda v: None,
        )
        proc.entry()
        assert isinstance(proc.metadata["output_log"], list)

    def test_vm_malloc_and_free_no_crash(self):
        k = Kernel()
        k.boot()
        src = "\n".join([
            "LOAD_CONST R7, 20", "LOAD_CONST R0, 4", "SYSCALL",
            "LOAD_CONST R7, 21", "SYSCALL",
            "HALT",
        ])
        proc = k.spawn_vm_process("vmtest", src, use_syscalls=True)
        proc.entry()
        assert isinstance(proc.metadata["output_log"], list)

    def test_vm_uptime_syscall_no_crash(self):
        k = Kernel()
        k.boot()
        src = "\n".join([
            "LOAD_CONST R7, 200", "SYSCALL",
            "HALT",
        ])
        proc = k.spawn_vm_process("vmtest", src, use_syscalls=True)
        proc.entry()
        assert isinstance(proc.metadata["output_log"], list)

    def test_vm_stats_syscall_no_crash(self):
        k = Kernel()
        k.boot()
        src = "\n".join([
            "LOAD_CONST R7, 201", "SYSCALL",
            "HALT",
        ])
        proc = k.spawn_vm_process("vmtest", src, use_syscalls=True)
        proc.entry()
        assert isinstance(proc.metadata["output_log"], list)

    def test_vm_metadata_has_source(self):
        k = Kernel()
        k.boot()
        src = "LOAD_CONST R0, 1\nHALT"
        proc = k.spawn_vm_process("test", src, use_syscalls=True)
        assert proc.metadata["source"] == src

    def test_vm_with_default_priority(self):
        k = Kernel()
        k.boot()
        proc = k.spawn_vm_process("test", "HALT")
        from domains.shell.kernel_process import Priority
        assert proc.priority == Priority.NORMAL

    def test_vm_exit_syscall(self):
        k = Kernel()
        k.boot()
        src = "\n".join([
            "LOAD_CONST R7, 2", "SYSCALL",
            "LOAD_CONST R0, 99",
            "HALT",
        ])
        proc = k.spawn_vm_process("vmtest", src, use_syscalls=True)
        proc.entry()
        assert isinstance(proc.metadata["output_log"], list)

    def test_vm_multiple_console_writes(self):
        k = Kernel()
        k.boot()
        outputs = []
        src = "\n".join([
            "LOAD_CONST R7, 111", "LOAD_CONST R0, 10", "SYSCALL",
            "LOAD_CONST R7, 111", "LOAD_CONST R0, 20", "SYSCALL",
            "HALT",
        ])
        proc = k.spawn_vm_process(
            "vmtest", src, use_syscalls=True,
            stdout_fn=outputs.append,
        )
        proc.entry()
        assert "10\n" in outputs
        assert "20\n" in outputs

    def test_vm_no_syscalls_mode(self):
        k = Kernel()
        k.boot()
        outputs = []
        src = "LOAD_CONST R0, 42\nHALT"
        proc = k.spawn_vm_process(
            "vmtest", src, use_syscalls=False,
            stdout_fn=outputs.append,
        )
        proc.entry()
        assert outputs == []

    def test_vm_metadata_output_log_is_list(self):
        k = Kernel()
        k.boot()
        proc = k.spawn_vm_process("test", "HALT", use_syscalls=True)
        assert isinstance(proc.metadata["output_log"], list)
        assert len(proc.metadata["output_log"]) == 0


# ---------------------------------------------------------------------------
# Extended run_program tests
# ---------------------------------------------------------------------------

class TestRunProgramExtended:
    def test_run_program_load_const(self):
        k = Kernel()
        k.boot()
        result = k.run_program("LOAD_CONST R0, 77\nHALT")
        assert result["regs"]["R0"] == 77

    def test_run_program_multiple_regs(self):
        k = Kernel()
        k.boot()
        result = k.run_program("LOAD_CONST R0, 1\nLOAD_CONST R1, 2\nLOAD_CONST R2, 3\nHALT")
        assert result["regs"]["R0"] == 1
        assert result["regs"]["R1"] == 2
        assert result["regs"]["R2"] == 3

    def test_run_program_steps_multiple(self):
        k = Kernel()
        k.boot()
        src = "\n".join(["LOAD_CONST R0, 1"] * 10 + ["HALT"])
        result = k.run_program(src)
        assert result["steps"] == 11

    def test_run_program_trace_content(self):
        k = Kernel()
        k.boot()
        result = k.run_program("HALT", trace=True)
        assert len(result["trace"]) >= 1

    def test_run_program_output_is_list(self):
        k = Kernel()
        k.boot()
        result = k.run_program("HALT")
        assert isinstance(result["output"], list)

    def test_run_program_regs_excludes_zero(self):
        k = Kernel()
        k.boot()
        result = k.run_program("LOAD_CONST R0, 0\nHALT")
        assert "R0" not in result["regs"]


# ---------------------------------------------------------------------------
# Extended kernel shell — additional commands
# ---------------------------------------------------------------------------

class TestKernelShellAdditional:
    def test_help_lists_all_commands(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["help", "halt"])
        joined = "\n".join(out)
        for cmd in ["help", "meminfo", "procs", "run", "ls", "cat", "write", "halt"]:
            assert cmd in joined

    def test_meminfo_after_boot(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["meminfo", "halt"])
        joined = "\n".join(out)
        assert "blocks:" in joined

    def test_procs_after_boot(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["procs", "halt"])
        joined = "\n".join(out)
        assert "pid=" in joined

    def test_multiple_empty_lines(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["", "", "halt"])
        prompt_count = sum(1 for o in out if "ai-compteur> " in o)
        assert prompt_count >= 3

    def test_run_program_from_fs(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        fs.write("prog.asm", "LOAD_CONST R0, 55\nHALT")
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, ["run prog", "halt"])
        joined = "\n".join(out)
        assert "loading prog..." in joined
        assert "done" in joined

    def test_cat_multiple_files(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        fs.write("a.txt", b"aaa")
        fs.write("b.txt", b"bbb")
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, ["cat a.txt", "cat b.txt", "halt"])
        joined = "\n".join(out)
        assert "aaa" in joined
        assert "bbb" in joined

    def test_write_and_run_roundtrip(self):
        k = Kernel()
        k.boot()
        from domains.shell.vm import BlockDevice, FlatFS
        block = BlockDevice()
        fs = FlatFS(block)
        k._block_device = block
        k._fs = fs
        out = _shell_outputs(k, [
            "write prog.asm LOAD_CONST R0, 77\nHALT",
            "run prog",
            "halt",
        ])
        joined = "\n".join(out)
        assert "wrote" in joined
        assert "done" in joined

    def test_halt_command(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["halt"])
        assert "shutting down..." in "\n".join(out)

    def test_exit_command(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["exit"])
        assert "shutting down..." in "\n".join(out)

    def test_quit_command(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["quit"])
        assert "shutting down..." in "\n".join(out)

    def test_whitespace_only_line(self):
        k = Kernel()
        k.boot()
        out = _shell_outputs(k, ["   ", "halt"])
        assert len(out) >= 1


# ---------------------------------------------------------------------------
# Extended setup() tests
# ---------------------------------------------------------------------------

class TestSetupExtended:
    def test_setup_adds_shell_ui_addon(self):
        k = Kernel()
        shell_ui.setup(k)
        assert k._addons.get("shell_ui") is True

    def test_setup_does_not_remove_existing_addons(self):
        k = Kernel()
        k._addons["custom"] = "value"
        shell_ui.setup(k)
        assert k._addons["custom"] == "value"

    def test_spawn_shell_via_addon(self):
        k = Kernel()
        shell_ui.setup(k)

        class FakeShell:
            def __init__(self, **kw):
                pass
            def run(self):
                pass

        proc = k.spawn_shell(shell_class=FakeShell)
        assert proc.name == "shell"
        proc.entry()

    def test_spawn_kernel_shell_via_addon(self):
        k = Kernel()
        shell_ui.setup(k)
        proc = k.spawn_kernel_shell()
        assert proc.name == "kernel-shell"

    def test_spawn_vm_process_via_addon(self):
        k = Kernel()
        shell_ui.setup(k)
        proc = k.spawn_vm_process("test", "HALT")
        assert proc.name == "test"

    def test_run_program_via_addon(self):
        k = Kernel()
        shell_ui.setup(k)
        result = k.run_program("HALT")
        assert result["steps"] == 1

    def test_setup_idempotent(self):
        k = Kernel()
        shell_ui.setup(k)
        shell_ui.setup(k)
        assert k._addons.get("shell_ui") is True
