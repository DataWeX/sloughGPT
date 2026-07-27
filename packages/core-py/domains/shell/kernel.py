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
from typing import Any

import numpy as np

from .kernel_process import Process, ProcessState, Priority, TensorRef
from .kernel_memory import TensorMemory, MemoryBlock
from .kernel_scheduler import Scheduler
from .kernel_syscall import SyscallTable, SyscallResult, SyscallNumber, build_default_syscall_table
from .kernel_devices import DeviceManager, DeviceDriver, DeviceType, DeviceState, DeviceHandle
from .kernel_interrupts import InterruptManager, InterruptType, Interrupt

logger = logging.getLogger("slo.kernel")


# ---------------------------------------------------------------------------
# Null Device
# ---------------------------------------------------------------------------

class NullDevice(DeviceDriver):
    """A null /dev/null device that discards writes and returns empty on read."""
    def __init__(self):
        super().__init__("null", DeviceType.CUSTOM)

    def read(self, **kwargs) -> bytes:
        return b""

    def write(self, data: Any) -> bool:
        return True


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

        # Auto-install neural addon if available
        try:
            from .addons import neural
            self.install_addon(neural)
        except Exception:
            pass

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
        """Install an addon module. Calls addon.setup(self)."""
        name = getattr(addon, "__name__", None) or addon.__spec__.name.rsplit(".", 1)[-1]
        short = name.rsplit(".", 1)[-1]
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
        if self._running:
            return "Already booted"
        self._boot_time = time.time()
        self._running = True

        # Register built-in devices
        self._devices.register(NullDevice())

        # Boot init process (completes immediately — it's just the boot marker)
        init_proc = self.spawn_process(
            "kernel-init",
            Priority.CRITICAL,
            entry=lambda: "booted",
        )

        msg = f"Kernel booted (pid={init_proc.pid}, memory={self._memory.capacity // (1024 * 1024)}MB)"
        logger.info(msg)
        return msg

    def spawn_shell(self, shell_class: Any = None, **kwargs: Any) -> Process:
        """Spawn the interactive shell as a kernel process.

        If shell_class is None, lazily imports ShellREPL. The shell runs
        as a NORMAL priority process — it gets scheduled by the kernel's
        tick loop.
        """
        if shell_class is None:
            from .repl import ShellREPL
            shell_class = ShellREPL

        def _shell_entry():
            shell = shell_class(**kwargs)
            shell.run()

        proc = self.spawn_process(
            "shell",
            priority=Priority.NORMAL,
            entry=_shell_entry,
        )
        return proc

    def spawn_kernel_shell(self, stdin_fn=None, stdout_fn=None) -> Process:
        """Spawn a simple kernel shell process.

        A minimal command loop that processes commands via the kernel's
        built-in dispatch. Commands: help, meminfo, procs, halt.
        """
        kernel = self

        def _kernel_shell_entry():
            prompt = "ai-compteur> "
            if stdout_fn:
                stdout_fn(prompt)

            while True:
                if stdin_fn:
                    line = stdin_fn()
                else:
                    break

                line = line.strip()
                if not line:
                    if stdout_fn:
                        stdout_fn(prompt)
                    continue

                parts = line.split()
                cmd = parts[0].lower()
                args = parts[1:]

                if cmd == "help":
                    cmds = "help, meminfo, procs, run, ls, cat, write, halt"
                    if stdout_fn:
                        stdout_fn(f"commands: {cmds}\n")
                elif cmd == "meminfo":
                    info = kernel._memory.stats()
                    if stdout_fn:
                        stdout_fn(f"blocks: {info.get('block_count', 0)}\n")
                elif cmd == "procs":
                    for p in kernel.list_processes():
                        if stdout_fn:
                            stdout_fn(f"  pid={p.pid} {p.name} {p.state.name}\n")
                elif cmd == "run":
                    if not args:
                        if stdout_fn:
                            stdout_fn("usage: run <program.asm>\n")
                    else:
                        prog_name = args[0]
                        if stdout_fn:
                            stdout_fn(f"loading {prog_name}...\n")
                        try:
                            from .vm import DiskProgramLoader, FlatFS, BlockDevice
                            if not hasattr(kernel, '_block_device'):
                                kernel._block_device = BlockDevice()
                                kernel._fs = FlatFS(kernel._block_device)
                            loader = DiskProgramLoader(kernel._fs)
                            result = loader.run(prog_name, stdout_fn=stdout_fn)
                            if stdout_fn:
                                stdout_fn(f"done ({result['steps']} steps)\n")
                        except Exception as e:
                            if stdout_fn:
                                stdout_fn(f"error: {e}\n")
                elif cmd == "ls":
                    if not hasattr(kernel, '_fs'):
                        if stdout_fn:
                            stdout_fn("no filesystem mounted\n")
                    else:
                        files = kernel._fs.list_files()
                        if not files:
                            if stdout_fn:
                                stdout_fn("(empty)\n")
                        else:
                            for f in files:
                                if stdout_fn:
                                    stdout_fn(f"  {f}\n")
                elif cmd == "cat":
                    if not args:
                        if stdout_fn:
                            stdout_fn("usage: cat <file>\n")
                    elif not hasattr(kernel, '_fs'):
                        if stdout_fn:
                            stdout_fn("no filesystem mounted\n")
                    else:
                        try:
                            data = kernel._fs.read(args[0])
                            if stdout_fn:
                                stdout_fn(data.decode('utf-8', errors='replace').rstrip('\x00') + "\n")
                        except Exception as e:
                            if stdout_fn:
                                stdout_fn(f"error: {e}\n")
                elif cmd == "write":
                    if len(args) < 2:
                        if stdout_fn:
                            stdout_fn("usage: write <file> <content>\n")
                    elif not hasattr(kernel, '_fs'):
                        if stdout_fn:
                            stdout_fn("no filesystem mounted\n")
                    else:
                        fname = args[0]
                        content = " ".join(args[1:])
                        try:
                            kernel._fs.write(fname, content.encode('utf-8'))
                            if stdout_fn:
                                stdout_fn(f"wrote {len(content)} bytes to {fname}\n")
                        except Exception as e:
                            if stdout_fn:
                                stdout_fn(f"error: {e}\n")
                elif cmd in ("halt", "exit", "quit"):
                    if stdout_fn:
                        stdout_fn("shutting down...\n")
                    break
                else:
                    if stdout_fn:
                        stdout_fn(f"unknown: {cmd}\n")

                if stdout_fn:
                    stdout_fn(prompt)

        proc = self.spawn_process(
            "kernel-shell",
            priority=Priority.NORMAL,
            entry=_kernel_shell_entry,
        )
        return proc

    def spawn_vm_process(self, name: str, source: str,
                         stdin_fn=None, stdout_fn=None,
                         priority: Priority = Priority.NORMAL,
                         use_syscalls: bool = False) -> Process:
        """Spawn a process that runs VM assembly code.

        Creates a VirtualSystem, loads the assembled program, and executes
        it in a background thread. I/O goes through the console device.
        If use_syscalls=True, wires SYSCALL instruction to kernel's syscall table.
        """
        from .vm import VirtualSystem, set_syscall_handler

        output_log: list[str] = []
        kernel = self

        def _handle_syscall(num, args):
            from .kernel_syscall import SyscallNumber
            if num == SyscallNumber.CONSOLE_WRITE:
                val = args[0]
                if stdout_fn:
                    stdout_fn(str(val) + "\n")
                else:
                    output_log.append(str(val))
                return 0
            elif num == SyscallNumber.CONSOLE_READ:
                if stdin_fn:
                    return stdin_fn()
                return ""
            elif num == SyscallNumber.EXIT:
                return -1
            elif num == SyscallNumber.MALLOC:
                block = kernel._memory.allocate(
                    shape=(args[0],) if args[0] else (1,),
                    dtype="float32",
                )
                return block.block_id
            elif num == SyscallNumber.FREE:
                kernel._memory.free_block(args[0])
                return 0
            elif num == SyscallNumber.UPTIME:
                return int(kernel.uptime * 1000)
            elif num == SyscallNumber.STATS:
                return kernel.info()
            return 0

        def _vm_entry():
            handler = _handle_syscall if use_syscalls else None
            vs = VirtualSystem(
                stdin_fn=stdin_fn,
                stdout_fn=stdout_fn or (lambda v: output_log.append(str(v))),
                syscall_handler=handler,
            )
            vs.load_program(source)
            vs.run()

        proc = self.spawn_process(
            name,
            priority=priority,
            entry=_vm_entry,
            metadata={"source": source, "output_log": output_log},
        )
        return proc

    def shutdown(self) -> str:
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
        logger.info(msg)
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
                            pass

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

        return {
            "current_pid": proc.pid if proc else None,
            "tick_count": self._tick_count,
        }

    # --- Hooks ---

    def on_tick(self, callback) -> None:
        self._on_tick.append(callback)

    def on_process_done(self, callback) -> None:
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
        results = []
        for _ in range(max_ticks):
            if not self._running:
                break
            result = self.tick()
            results.append(result)
            if not self._processes:
                break
        return results

    def run_program(self, source: str, trace: bool = False) -> dict:
        """Run a VM assembly program through the kernel's device bus.

        Creates a VM CPU, wires it to the kernel's devices, and executes
        the assembled program. Returns output, trace, and step count.
        """
        from .vm import CPU, Assembler, DeviceBus as VMBus

        vm_bus = VMBus()
        if hasattr(self._devices, '_table'):
            for name, dev in self._devices._table._devices.items():
                vm_bus.register(name, dev)
        elif hasattr(self._devices, '_devices'):
            for name, dev in self._devices._devices.items():
                vm_bus.register(name, dev)

        cpu = CPU(devices=vm_bus)
        cpu._tracing = trace
        assembler = Assembler()
        instructions = assembler.assemble(source)
        cpu.load_program(instructions)
        output = cpu.run()

        return {
            "output": output,
            "steps": cpu._step_count,
            "trace": cpu.get_trace() if trace else [],
            "regs": {f"R{i}": v for i, v in enumerate(cpu.regs) if v != 0},
        }

    # --- Info ---

    def info(self) -> dict:
        """Return a snapshot of kernel state."""
        return {
            "uptime_s": self.uptime,
            "running": self._running,
            "tick_count": self._tick_count,
            "process_count": len(self._processes),
            "memory": self._memory.stats(),
            "devices": self._devices.stats(),
            "interrupts": self._interrupts.stats(),
            "syscalls": self._syscall_table.stats(),
        }

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
    # Neural capabilities (via addon)
    # ===================================================================

    @property
    def engine(self):
        self._require_addon("neural")
        return self._engine

    @property
    def tokenizer_device(self):
        self._require_addon("neural")
        return self._tokenizer_device

    @property
    def embedding_device(self):
        self._require_addon("neural")
        return self._embedding_device

    @property
    def embedding_store(self):
        from .addons.neural import NeuralEmbeddingStore
        self._require_addon("neural")
        stores = list(self._embedding_stores.values())
        return stores[0] if stores else NeuralEmbeddingStore()

    @property
    def kv_caches(self) -> dict:
        self._require_addon("neural")
        return self._kv_caches

    @property
    def gradient_accumulator(self):
        self._require_addon("neural")
        return self._gradient_accumulator

    @property
    def batch_processor(self):
        self._require_addon("neural")
        return self._batch_processor

    def create_neural_process(self, name: str, neural_type: Any = None,
                              model_name: str = "", priority: Priority = Priority.NORMAL,
                              **kwargs) -> Any:
        from .addons.neural import NeuralProcess, NeuralProcessType
        self._require_addon("neural")
        proc = self.spawn_process(name, priority)
        neural = NeuralProcess(process=proc, model_name=model_name)
        neural.neural_type = neural_type or NeuralProcessType.INFERENCE
        with self._lock:
            self._neural_processes[proc.pid] = neural
        return neural

    def get_neural_process(self, pid: int):
        self._require_addon("neural")
        return self._neural_processes.get(pid)

    def list_neural_processes(self) -> list:
        self._require_addon("neural")
        return list(self._neural_processes.values())

    def tokenize(self, text: str) -> list[int]:
        self._require_addon("neural")
        result = self._tokenizer_device.ioctl("encode", text)
        if result and hasattr(result, 'value') and result.value:
            return result.value.get("tokens", [])
        return list(text.encode("utf-8"))

    def detokenize(self, tokens: list[int]) -> str:
        self._require_addon("neural")
        result = self._tokenizer_device.ioctl("decode", tokens)
        if result and hasattr(result, 'value') and result.value:
            return result.value.get("text", "")
        return bytes(tokens).decode("utf-8", errors="replace")

    def embed(self, ids: np.ndarray, store_name: str = "default") -> np.ndarray | None:
        self._require_addon("neural")
        store = self._embedding_stores.get(store_name)
        if store is None:
            return None
        return store.lookup(ids)

    def embed_text(self, text: str) -> np.ndarray:
        from .addons.neural import NeuralEmbeddingStore, NeuralSyscall
        self._require_addon("neural")
        store = list(self._embedding_stores.values())[0] if self._embedding_stores else NeuralEmbeddingStore()
        return NeuralSyscall.embed(store, text)

    def create_embedding_store(self, name: str, vocab_size: int = 1000,
                               embed_dim: int = 64):
        from .addons.neural import NeuralEmbeddingStore
        self._require_addon("neural")
        store = NeuralEmbeddingStore(vocab_size=vocab_size, embed_dim=embed_dim)
        self._embedding_stores[name] = store
        return store

    def create_kv_cache(self, name: str, num_layers: int = 6,
                        head_dim: int = 32, **kwargs):
        from .addons.neural import NeuralKVCache
        self._require_addon("neural")
        cache = NeuralKVCache(num_layers=num_layers, head_dim=head_dim,
                              max_positions=kwargs.get('max_positions', 512))
        self._kv_caches[name] = cache
        return cache

    def get_kv_cache(self, name: str):
        self._require_addon("neural")
        return self._kv_caches.get(name)

    def remove_kv_cache(self, name: str) -> None:
        self._require_addon("neural")
        with self._lock:
            self._kv_caches.pop(name, None)

    def generate(self, model_name: str, prompt: str, max_tokens: int = 10, **kwargs: Any) -> dict[str, Any] | None:
        self._require_addon("neural")
        result = self._engine.ioctl("generate", model_name, prompt, max_tokens=max_tokens, **kwargs)
        if result and hasattr(result, 'value') and result.value:
            return result.value
        return None

    def forward(self, neural_proc: Any, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        from .addons.neural import NeuralSyscall
        return NeuralSyscall.forward(neural_proc, inputs)

    def backward(self, neural_proc: Any, grad_output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        from .addons.neural import NeuralSyscall
        return NeuralSyscall.backward(neural_proc, grad_output)

    def attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                  mask: np.ndarray | None = None) -> np.ndarray:
        from .addons.neural import NeuralSyscall
        return NeuralSyscall.attention(self._attention_device, q, k, v, mask)

    def neural_syscall(self, proc: Any, op: str, *args: Any, **kwargs: Any) -> Any:
        from .addons.neural import NeuralSyscall, NeuralEmbeddingStore
        if op == "forward":
            return NeuralSyscall.forward(proc, *args, **kwargs)
        elif op == "backward":
            return NeuralSyscall.backward(proc, *args, **kwargs)
        elif op == "embed":
            return NeuralSyscall.embed(self._embedding_stores.get("default", NeuralEmbeddingStore()), *args, **kwargs)
        return None

    def register_devices(self) -> None:
        if "neural" in self._addons:
            self.register_device(self._engine)
            self.register_device(self._tokenizer_device)
            self.register_device(self._embedding_device)
            self.register_device(self._attention_device)

    def cleanup_pid(self, pid: int) -> None:
        if "neural" in self._addons:
            with self._lock:
                self._neural_processes.pop(pid, None)
        self.memory.free_pid(pid)

    def neural_stats(self) -> dict:
        self._require_addon("neural")
        return {
            "neural_processes": len(self._neural_processes),
            "kv_caches": len(self._kv_caches),
            "embedding_stores": len(self._embedding_stores),
            "gradient_accumulator": self._gradient_accumulator.stats(),
            "batch_processor": self._batch_processor.stats(),
            "attention_device": self._attention_device.info(),
            "engine": self._engine.info(),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_kernel: Kernel | None = None


def get_kernel() -> Kernel:
    global _kernel
    if _kernel is None:
        _kernel = Kernel()
        from .addons import neural
        _kernel.install_addon(neural)
        _kernel.register_devices()
    return _kernel


def reset_kernel() -> Kernel:
    global _kernel
    if _kernel is not None and _kernel.running:
        _kernel.shutdown()
    _kernel = Kernel()
    from .addons import neural
    _kernel.install_addon(neural)
    _kernel.register_devices()
    return _kernel
