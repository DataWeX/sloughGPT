"""
Coverage-completing tests for the Kernel class (edge branches, hooks, lifecycle).

Run: PYTHONPATH=packages/core-py python -m pytest tests/test_kernel_coverage.py -x -q
"""

import time
from types import SimpleNamespace

import pytest

import types

from domains.shell import kernel as kernel_mod
from domains.shell.kernel import Kernel, NeuralKernel
from domains.shell.kernel_devices import NullDevice
from domains.shell.kernel_interrupts import Interrupt, InterruptType
from domains.shell.kernel_process import ProcessState
from domains.shell.kernel_syscall import SyscallNumber


def _make_fake_addon():
    m = types.ModuleType("pkg.fake")
    m.setup_calls = [0]

    def setup(kernel):
        m.setup_calls[0] += 1
        kernel._addons["fake"] = True

    m.setup = setup
    return m


_FakeAddon = _make_fake_addon()


class _NoNameAddon:
    """Addon instance without __name__ but with __spec__.name."""

    def __init__(self):
        self.__spec__ = SimpleNamespace(name="pkg.something")

    def setup(self, kernel):
        kernel._addons["something"] = True


class _AnonymousAddon:
    """Addon instance with neither __name__ nor __spec__."""

    def setup(self, kernel):
        kernel._addons["anonymous"] = True


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def _run_until_done(k, proc, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.state == ProcessState.ZOMBIE:
            return True
        k.tick()
        time.sleep(0.005)
    return False


class TestInstallAddon:
    def test_install_with_name(self):
        k = Kernel()
        k.install_addon(_FakeAddon)
        assert k._addons.get("fake") is True

    def test_install_without_name_uses_spec(self):
        k = Kernel()
        k.install_addon(_NoNameAddon())
        assert k._addons.get("something") is True

    def test_install_idempotent(self):
        _FakeAddon.setup_calls[0] = 0
        k = Kernel()
        k.install_addon(_FakeAddon)
        k.install_addon(_FakeAddon)
        assert _FakeAddon.setup_calls[0] == 1

    def test_install_addon_without_spec(self):
        k = Kernel()
        k.install_addon(_AnonymousAddon())
        assert k._addons.get("anonymous") is True

    def test_has_addon_true(self):
        k = Kernel()
        k.install_addon(_FakeAddon)
        assert k.has_addon("fake") is True

    def test_has_addon_false(self):
        k = Kernel()
        assert k.has_addon("missing") is False


class TestLifecycle:
    def test_boot_twice_returns_already_booted(self):
        k = Kernel()
        k.boot()
        assert k.boot() == "Already booted"

    def test_boot_neural_addon_failure(self, monkeypatch):
        import domains.shell.addons.neural as neural_mod
        monkeypatch.setattr(neural_mod, "setup", _boom)
        k = Kernel()
        msg = k.boot()
        assert "Kernel booted" in msg

    def test_boot_shell_ui_addon_failure(self, monkeypatch):
        import domains.shell.addons.shell_ui as shell_ui_mod
        monkeypatch.setattr(shell_ui_mod, "setup", _boom)
        k = Kernel()
        msg = k.boot()
        assert "Kernel booted" in msg

    def test_shutdown_when_not_running(self):
        k = Kernel()
        assert k.shutdown() == "Already shut down"

    def test_uptime_before_boot_is_zero(self):
        k = Kernel()
        assert k.uptime == 0.0

    def test_tick_count_property(self):
        k = Kernel()
        k.boot()
        k.tick()
        assert k.tick_count == 1

    def test_running_property(self):
        k = Kernel()
        assert k.running is False
        k.boot()
        assert k.running is True

    def test_neural_kernel_deprecation(self):
        with pytest.warns(DeprecationWarning):
            nk = NeuralKernel()
        assert nk._running is False


class TestProcessManagement:
    def test_kill_process(self):
        k = Kernel()
        k.boot()
        proc = k.spawn_process("test", entry=lambda: 42)
        assert k.kill_process(proc.pid) is True
        assert proc.state == ProcessState.STOPPED

    def test_kill_nonexistent_process(self):
        k = Kernel()
        assert k.kill_process(999) is False

    def test_get_process(self):
        k = Kernel()
        proc = k.spawn_process("x")
        assert k.get_process(proc.pid) is proc

    def test_create_process_returns_pid(self):
        k = Kernel()
        pid = k.create_process("y")
        assert k.get_process(pid) is not None

    def test_spawn_with_depends_on(self):
        k = Kernel()
        a = k.spawn_process("a", entry=lambda: 1)
        b = k.spawn_process("b", entry=lambda: 2, depends_on=[a.pid])
        assert b.metadata["depends_on"] == [a.pid]

    def test_list_processes(self):
        k = Kernel()
        k.spawn_process("a")
        k.spawn_process("b")
        assert len(k.list_processes()) == 2

    def test_require_addon_raises(self):
        k = Kernel()
        with pytest.raises(RuntimeError, match="not installed"):
            k._require_addon("missing")

    def test_process_done_callback_fires(self):
        k = Kernel()
        k.boot()
        done = []
        k.on_process_done(done.append)
        proc = k.spawn_process("work", entry=lambda: "result")
        assert _run_until_done(k, proc)
        assert proc in done

    def test_process_done_callback_raises(self):
        k = Kernel()
        k.boot()

        def _raise_cb(p):
            raise ValueError("cb boom")

        k.on_process_done(_raise_cb)
        proc = k.spawn_process("work", entry=lambda: 1)
        assert _run_until_done(k, proc)

    def test_process_entry_crash_sets_error(self):
        k = Kernel()
        k.boot()

        def _crash():
            raise RuntimeError("entry boom")

        proc = k.spawn_process("crash", entry=_crash)
        assert _run_until_done(k, proc)
        assert "entry boom" in proc.error

    def test_on_process_done_registration(self):
        k = Kernel()
        cb = lambda p: None  # noqa: E731
        k.on_process_done(cb)
        assert cb in k._on_process_done


class TestSyscalls:
    def test_syscall_without_caller_no_processes(self):
        k = Kernel()
        result = k.syscall(SyscallNumber.GETPID)
        assert result.success is True

    def test_syscall_base_dispatch_on_booted(self):
        k = Kernel()
        k.boot()
        proc = k.spawn_process("x")
        result = k.syscall(SyscallNumber.GETPID, caller=proc)
        assert result.success is True

    def test_syscall_tensor_alloc(self):
        k = Kernel()
        result = k.syscall(SyscallNumber.TENSOR_ALLOC, (2, 3), "float32")
        assert result.success is True
        assert result.value["shape"] == (2, 3)

    def test_syscall_unknown_number(self):
        k = Kernel()
        result = k.syscall(999999)
        assert result.success is False

    def test_syscall_infers_caller_from_first_process(self):
        k = Kernel()
        k.boot()
        result = k.syscall(SyscallNumber.GETPID)
        assert result.success is True

    def test_syscall_table_property(self):
        k = Kernel()
        assert k.syscall_table is k._syscall_table


class TestDevices:
    def test_devices_property(self):
        k = Kernel()
        assert k.devices is k._devices

    def test_register_and_unregister_device(self):
        k = Kernel()
        assert k.register_device(NullDevice()) is True
        assert k.unregister_device("null") is True

    def test_open_and_close_device_handle(self):
        k = Kernel()
        k.boot()
        handle = k.open_device("null")
        assert k.close_device(handle) is True

    def test_close_device_by_int_fd(self):
        k = Kernel()
        k.boot()
        handle = k.open_device("null")
        assert k.close_device(handle.fd) is True

    def test_interrupts_property(self):
        k = Kernel()
        assert k.interrupts is k._interrupts

    def test_scheduler_property(self):
        k = Kernel()
        assert k.scheduler is k._scheduler

    def test_register_devices_noop(self):
        k = Kernel()
        k.register_devices()

    def test_memory_property(self):
        k = Kernel()
        assert k.memory is k._memory

    def test_alloc_and_free_tensor(self):
        k = Kernel()
        info = k.alloc_tensor((2, 3), "float32")
        assert info["shape"] == (2, 3)
        assert k.free_tensor(info["block_id"]) is True

    def test_vfs_requires_filesystem_addon(self):
        from domains.shell.addons import filesystem
        k = Kernel()
        k.install_addon(filesystem)
        assert k.vfs is k._vfs


class TestInterruptHandlers:
    def _booted_with_proc(self):
        k = Kernel()
        k.boot()
        proc = k.spawn_process("x", entry=lambda: 1)
        return k, proc

    def test_handle_process_done(self):
        k, proc = self._booted_with_proc()
        done = []
        k.on_process_done(done.append)
        k._handle_process_done(Interrupt(
            vector=InterruptType.PROCESS_DONE,
            source_pid=proc.pid,
            data={"ok": True},
        ))
        assert proc.result == {"ok": True}
        assert proc in done

    def test_handle_process_done_callback_raises(self):
        k, proc = self._booted_with_proc()

        def _raise_cb(p):
            raise RuntimeError("cb fail")

        k.on_process_done(_raise_cb)
        k._handle_process_done(Interrupt(
            vector=InterruptType.PROCESS_DONE,
            source_pid=proc.pid,
            data={"ok": True},
        ))
        assert proc.result == {"ok": True}

    def test_handle_process_done_missing_pid(self):
        k = Kernel()
        k.boot()
        k._handle_process_done(Interrupt(
            vector=InterruptType.PROCESS_DONE,
            source_pid=9999,
            data={"ok": True},
        ))

    def test_handle_memory_full(self):
        k = Kernel()
        k.boot()
        k._handle_memory_full(Interrupt(vector=InterruptType.MEMORY_FULL))

    def test_handle_device_error(self):
        k = Kernel()
        k.boot()
        k._handle_device_error(Interrupt(vector=InterruptType.DEVICE_ERROR, data="err"))


class TestTickRun:
    def test_tick_when_not_running(self):
        k = Kernel()
        out = k.tick()
        assert out == {"current_pid": None, "tick_count": 0}

    def test_tick_fires_on_tick_callback(self):
        k = Kernel()
        k.boot()
        ticks = []
        k.on_tick(ticks.append)
        k.tick()
        assert ticks == [1]

    def test_on_tick_callback_raises(self):
        k = Kernel()
        k.boot()
        k.on_tick(_boom)
        out = k.tick()
        assert out["tick_count"] == 1

    def test_on_tick_registration(self):
        k = Kernel()
        cb = lambda t: None  # noqa: E731
        k.on_tick(cb)
        assert cb in k._on_tick

    def test_run_when_not_running(self):
        k = Kernel()
        assert k.run(3) == []

    def test_run_stops_when_no_processes(self):
        k = Kernel()
        k.boot()
        k._processes.clear()
        results = k.run(3)
        assert len(results) == 1


class TestInfo:
    def test_stats_keys(self):
        k = Kernel()
        k.boot()
        s = k.stats()
        assert "uptime" in s
        assert "tick_count" in s
        assert "scheduler" in s

    def test_info_aliases_uptime(self):
        k = Kernel()
        k.boot()
        info = k.info()
        assert "uptime_s" in info
        assert "uptime" not in info


class TestSingleton:
    def test_get_kernel_creates_singleton(self, monkeypatch):
        monkeypatch.setattr(kernel_mod, "_kernel", None)
        k = kernel_mod.get_kernel()
        assert k is kernel_mod.get_kernel()
        assert k.has_addon("neural")
        assert k.has_addon("filesystem")
        assert k.has_addon("shell_ui")

    def test_reset_kernel_shuts_down_running(self, monkeypatch):
        k = Kernel()
        k.boot()
        monkeypatch.setattr(kernel_mod, "_kernel", k)
        new_k = kernel_mod.reset_kernel()
        assert new_k is not k
        assert new_k is kernel_mod._kernel
        assert k._running is False
