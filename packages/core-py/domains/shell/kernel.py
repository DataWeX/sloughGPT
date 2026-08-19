"""
AI-Native Kernel — core process/memory/device management.

Core OS concerns: processes, memory, devices, scheduler, interrupts, syscalls.
Neural capabilities live in addons/neural.py (canonical) and are composed into
the Kernel class via the addon system (kernel.install_addon()).

All kernel imports go through this module:
    from domains.shell.kernel import Kernel, Process, ProcessState

Supporting modules (kernel_process, kernel_memory, etc.) are imported by this
file — they contain the subsystem implementations. kernel.py is the canonical
source of the Kernel class.
"""

from __future__ import annotations

import time
import logging
import threading
from typing import Any, Callable

from .kernel_process import Process, ProcessState, Priority
from .kernel_memory import TensorMemory
from .kernel_scheduler import Scheduler
from .kernel_syscall import SyscallTable, SyscallResult, SyscallNumber, build_default_syscall_table
from .kernel_devices import DeviceManager, DeviceDriver, DeviceHandle, NullDevice
from .kernel_interrupts import InterruptManager, InterruptType, Interrupt

logger = logging.getLogger("slo.kernel")


# ---------------------------------------------------------------------------
# Unified Kernel — core + neural
# ---------------------------------------------------------------------------

class Kernel:
    """
    AI-native unified kernel.

    Core process/memory/device management with composable addons.
    Addons register via kernel.install_addon(addon_module).

    Single entry point for all kernel operations.
    """

    def __init__(self):
        # Core subsystems
        self._scheduler = Scheduler()
        self._memory = TensorMemory()
        self._devices = DeviceManager()
        self._interrupts = InterruptManager()
        self._syscall_table = build_default_syscall_table()

        # Process tracking
        self._next_pid = 1
        self._processes: dict[int, Process] = {}
        self._lock = threading.Lock()

        # Lifecycle
        self._boot_time: float | None = None
        self._running = False
        self._tick_count = 0

        # Hooks
        self._on_tick: list = []
        self._on_process_done: list = []

        # Addon registry
        self._addons: dict[str, Any] = {}

        # Wire up default interrupt handlers
        self._interrupts.vector.register(
            InterruptType.PROCESS_DONE, self._handle_process_done
        )
        self._interrupts.vector.register(
            InterruptType.MEMORY_FULL, self._handle_memory_full
        )
        self._interrupts.vector.register(
            InterruptType.DEVICE_ERROR, self._handle_device_error
        )

    # --- Addon API ---

    def install_addon(self, addon: Any) -> None:
        """Install an addon module. Calls addon.setup(self).

        Idempotent — second call for the same addon is a no-op.
        """
        name = getattr(addon, "__name__", None)
        if name is None:
            spec = getattr(addon, "__spec__", None)
            name = getattr(spec, "name", None) if spec else None
        short = name.rsplit(".", 1)[-1] if name else str(addon)
        if short in self._addons:
            return
        addon.setup(self)

    def has_addon(self, name: str) -> bool:
        return name in self._addons

    def _require_addon(self, name: str) -> None:
        if name not in self._addons:
            raise RuntimeError(f"Addon '{name}' not installed. Call kernel.install_addon() first.")

    # --- Lifecycle ---

    def boot(self) -> str:
        """Boot the kernel: install addons, register devices, start init process.

        Returns:
            Boot message with pid and memory info.
        """
        if self._running:
            return "Already booted"
        self._boot_time = time.time()
        self._running = True

        # Auto-install neural addon if not yet installed
        if "neural" not in self._addons:
            try:
                from .addons import neural
                self.install_addon(neural)
            except Exception:
                pass

        # Auto-install shell_ui addon if not yet installed
        if "shell_ui" not in self._addons:
            try:
                from .addons import shell_ui
                self.install_addon(shell_ui)
            except Exception:
                pass

        # Register built-in devices
        self._devices.register(NullDevice())

        # Boot init process (completes immediately — it's just the boot marker)
        init_proc = self.spawn_process(
            "kernel-init",
            Priority.CRITICAL,
            entry=lambda: "booted",
        )

        msg = f"Kernel booted (pid={init_proc.pid}, memory={self._memory.capacity // (1024 * 1024)}MB)"
        logger.debug(msg)
        return msg

    def shutdown(self) -> str:
        """Shut down the kernel: stop processes, free memory.

        Returns:
            Shutdown message with uptime and tick count.
        """
        if not self._running:
            return "Already shut down"
        self._running = False

        # Stop all processes
        for proc in list(self._processes.values()):
            if proc.is_active:
                proc.transition(ProcessState.STOPPED)

        # Free all process memory
        for pid in list(self._processes.keys()):
            self._memory.free_pid(pid)

        msg = f"Kernel shut down (uptime={self.uptime:.1f}s, ticks={self._tick_count})"
        logger.debug(msg)
        return msg

    @property
    def uptime(self) -> float:
        if self._boot_time is None:
            return 0.0
        return time.time() - self._boot_time

    @property
    def running(self) -> bool:
        return self._running

    @property
    def tick_count(self) -> int:
        return self._tick_count

    # --- Process management ---

    def spawn_process(self, name: str, priority: Priority = Priority.NORMAL,
                      entry: Any = None, args: tuple = (), metadata: dict | None = None,
                      depends_on: list[int] | None = None) -> Process:
        """Create and register a new process.

        Args:
            name: Human-readable process name.
            priority: Scheduling priority (CRITICAL, HIGH, NORMAL, LOW, IDLE).
            entry: Callable to execute when process runs.
            args: Positional arguments for entry.
            metadata: Arbitrary metadata dict.
            depends_on: List of pids this process depends on.

        Returns:
            The created Process object.
        """
        with self._lock:
            pid = self._next_pid
            self._next_pid += 1

        proc = Process(
            pid=pid,
            name=name,
            priority=priority,
            entry=entry,
            args=args,
            metadata=metadata or {},
        )
        if depends_on:
            proc.metadata["depends_on"] = depends_on

        with self._lock:
            self._processes[pid] = proc

        self._scheduler.add(proc)
        logger.debug("Spawned pid=%d name=%s priority=%s", pid, name, priority.name)
        return proc

    def create_process(self, name: str, priority: Priority = Priority.NORMAL,
                       depends_on: list[int] | None = None) -> int:
        """Create a process and return its PID. Backward-compatible wrapper."""
        proc = self.spawn_process(name, priority, depends_on=depends_on)
        return proc.pid

    def kill_process(self, pid: int) -> bool:
        proc = self._processes.get(pid)
        if proc is None:
            return False
        proc.transition(ProcessState.STOPPED)
        self._scheduler.remove(pid)
        self._memory.free_pid(pid)
        self._interrupts.signal_process_done(pid)
        logger.debug("Killed pid=%d", pid)
        return True

    def get_process(self, pid: int) -> Process | None:
        return self._processes.get(pid)

    def list_processes(self) -> list[Process]:
        return list(self._processes.values())

    # --- Memory ---

    @property
    def memory(self) -> TensorMemory:
        return self._memory

    def alloc_tensor(self, shape: tuple, dtype: str = "float32") -> dict:
        """Allocate a tensor block in kernel memory. Returns block metadata."""
        block = self._memory.allocate(shape, dtype)
        return {
            "block_id": block.block_id,
            "shape": block.shape,
            "dtype": block.dtype,
            "size_bytes": block.size_bytes,
        }

    def free_tensor(self, block_id: int) -> bool:
        """Free a tensor block by ID."""
        return self._memory.free_block(block_id)

    # --- Devices ---

    @property
    def devices(self) -> DeviceManager:
        return self._devices

    def register_device(self, device: DeviceDriver) -> bool:
        return self._devices.register(device)

    def unregister_device(self, name: str) -> bool:
        return self._devices.unregister(name)

    def open_device(self, name: str) -> Any:
        """Open a device by name, returns a DeviceHandle."""
        return self._devices.open(name)

    def close_device(self, fd: Any) -> bool:
        """Close a device handle. Accepts DeviceHandle or int fd."""
        if isinstance(fd, DeviceHandle):
            return self._devices.close(fd.fd)
        return self._devices.close(fd)

    # --- Interrupts ---

    @property
    def interrupts(self) -> InterruptManager:
        return self._interrupts

    # --- Syscalls ---

    @property
    def syscall_table(self) -> SyscallTable:
        return self._syscall_table

    def syscall(self, number: Any, *args: Any, caller: Process | None = None, **kwargs: Any) -> Any:
        """Dispatch a syscall, handling both base and neural syscall numbers."""
        if caller is None:
            for proc in self._processes.values():
                caller = proc
                break
        if caller is None:
            caller = Process(pid=0, name="kernel", state=ProcessState.RUNNING)

        # Check if this is a known base syscall number
        try:
            sn = SyscallNumber(number)
        except ValueError:
            sn = None

        if sn is not None:
            # Handle TENSOR_ALLOC directly
            if sn == SyscallNumber.TENSOR_ALLOC:
                shape, dtype = args[0], args[1] if len(args) > 1 else "float32"
                info = self.alloc_tensor(shape, dtype)
                return SyscallResult(success=True, value=info)
            return self._syscall_table.dispatch(sn, caller, *args, **kwargs)

        # Custom neural syscall — dispatch directly through table
        return self._syscall_table.dispatch(number, caller, *args, **kwargs)

    # --- Scheduler ---

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    # --- Tick ---

    def tick(self) -> dict:
        """Advance the kernel by one tick.

        If the scheduled process has an entry function and hasn't been started
        yet, launches it in a background thread. When the entry completes,
        the process transitions to ZOMBIE.
        """
        if not self._running:
            return {"current_pid": None, "tick_count": self._tick_count}

        self._tick_count += 1
        proc = self._scheduler.tick()

        # Launch process entry function if present and not yet started
        if proc is not None and proc.entry is not None and proc._thread is None:
            proc.transition(ProcessState.RUNNING)
            proc.started_at = time.time()

            def _run_proc(p: Process):
                try:
                    result = p.entry(*p.args, **p.kwargs)
                    p.result = result
                except Exception as exc:
                    p.error = str(exc)
                    logger.error("Process %d (%s) crashed: %s", p.pid, p.name, exc)
                finally:
                    p.finished_at = time.time()
                    p.cpu_time_ms = (p.finished_at - (p.started_at or p.created_at)) * 1000
                    p.transition(ProcessState.ZOMBIE)
                    self._scheduler.complete(p.pid)
                    for cb in self._on_process_done:
                        try:
                            cb(p)
                        except Exception:
                            logger.debug("on_process_done callback failed", exc_info=True)

            t = threading.Thread(target=_run_proc, args=(proc,), daemon=True, name=f"proc-{proc.pid}")
            proc._thread = t
            t.start()

        # Fire timer interrupt
        self._interrupts.vector.fire(Interrupt(
            vector=InterruptType.TIMER,
            data={"tick": self._tick_count},
        ))

        # Process pending interrupts
        self._interrupts.vector.process_pending()

        # Fire tick callbacks
        for cb in self._on_tick:
            try:
                cb(self._tick_count)
            except Exception:
                logger.debug("on_tick callback failed", exc_info=True)

        return {
            "current_pid": proc.pid if proc else None,
            "tick_count": self._tick_count,
        }

    # --- Hooks ---

    def on_tick(self, callback: Callable[[int], None]) -> None:
        """Register a callback to fire on each scheduler tick.

        Args:
            callback: Function called with tick_count on each tick.
        """
        self._on_tick.append(callback)

    def on_process_done(self, callback: Callable[[Process], None]) -> None:
        """Register a callback to fire when a process completes.

        Args:
            callback: Function called with the completed Process.
        """
        self._on_process_done.append(callback)

    # --- Interrupt handlers ---

    def _handle_process_done(self, interrupt: Interrupt) -> None:
        pid = interrupt.source_pid
        if pid is not None:
            proc = self._processes.get(pid)
            if proc is not None:
                proc.result = interrupt.data
                for cb in self._on_process_done:
                    try:
                        cb(proc)
                    except Exception:
                        logger.exception("on_process_done callback failed")

    def _handle_memory_full(self, interrupt: Interrupt) -> None:
        logger.warning("Memory full interrupt fired")

    def _handle_device_error(self, interrupt: Interrupt) -> None:
        logger.error("Device error interrupt: %s", interrupt.data)

    # --- Run loop ---

    def run(self, max_ticks: int = 100) -> list[dict]:
        """Run the kernel for up to max_ticks, returning tick results."""
        from .kernel_process import ProcessState
        results = []
        for _ in range(max_ticks):
            if not self._running:
                break
            result = self.tick()
            results.append(result)
            if not any(p.state not in (ProcessState.ZOMBIE, ProcessState.STOPPED)
                       for p in self._processes.values()):
                break
        return results

    # --- Info ---

    def info(self) -> dict:
        """Return a snapshot of kernel state (backward-compatible keys)."""
        s = self.stats()
        s["uptime_s"] = s.pop("uptime")
        return s

    def stats(self) -> dict:
        return {
            "uptime": self.uptime,
            "running": self._running,
            "tick_count": self._tick_count,
            "process_count": len(self._processes),
            "scheduler": self._scheduler.stats(),
            "memory": self._memory.stats(),
            "devices": self._devices.stats(),
            "interrupts": self._interrupts.stats(),
            "syscalls": self._syscall_table.stats(),
        }

    # ===================================================================
    # Filesystem (via addon)
    # ===================================================================

    def register_devices(self) -> None:
        """Register built-in and addon-provided devices.

        Built-in devices (NullDevice) are registered in boot().
        Addon devices are registered during install_addon() in each
        addon's setup(). This method is maintained as a convenience
        for explicit one-shot device registration, but addon setup()
        is the canonical path.
        """
        pass

    @property
    def vfs(self):
        self._require_addon("filesystem")
        return self._vfs


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_kernel: Kernel | None = None


def get_kernel() -> Kernel:
    global _kernel
    if _kernel is None:
        _kernel = Kernel()
        from .addons import neural, filesystem, shell_ui
        _kernel.install_addon(neural)
        _kernel.install_addon(filesystem)
        _kernel.install_addon(shell_ui)
    return _kernel


def reset_kernel() -> Kernel:
    global _kernel
    if _kernel is not None and _kernel.running:
        _kernel.shutdown()
    _kernel = Kernel()
    from .addons import neural, filesystem, shell_ui
    _kernel.install_addon(neural)
    _kernel.install_addon(filesystem)
    _kernel.install_addon(shell_ui)
    return _kernel


class NeuralKernel(Kernel):
    """Deprecated: use Kernel() instead. Kernel auto-installs the neural addon."""

    def __init__(self):
        import warnings
        warnings.warn(
            "NeuralKernel is deprecated, use Kernel() instead. "
            "Kernel auto-installs the neural addon via boot().",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__()
