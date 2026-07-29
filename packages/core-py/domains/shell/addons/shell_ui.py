"""
Shell/VM spawn addon — interactive shell and VM process creation.

Extracts spawn_shell, spawn_kernel_shell, spawn_vm_process, and run_program
from kernel.py. These are shell-level concerns (UI, command dispatch, VM
assembly) that don't belong in kernel core.

Install via:
    from domains.shell.addons import shell_ui
    kernel.install_addon(shell_ui)
"""
from __future__ import annotations

import logging
from typing import Any

from ..kernel_process import Priority, Process

logger = logging.getLogger("slo.kernel.shell_ui")


def spawn_shell(kernel: Any, shell_class: Any = None, **kwargs: Any) -> Process:
    """Spawn the interactive shell as a kernel process.

    If shell_class is None, lazily imports ShellREPL. The shell runs
    as a NORMAL priority process — it gets scheduled by the kernel's
    tick loop.
    """
    if shell_class is None:
        from ..repl import ShellREPL
        shell_class = ShellREPL

    def _shell_entry():
        shell = shell_class(**kwargs)
        shell.run()

    proc = kernel.spawn_process(
        "shell",
        priority=Priority.NORMAL,
        entry=_shell_entry,
    )
    return proc


def spawn_kernel_shell(kernel: Any, stdin_fn=None, stdout_fn=None) -> Process:
    """Spawn a simple kernel shell process.

    A minimal command loop that processes commands via the kernel's
    built-in dispatch. Commands: help, meminfo, procs, run, ls, cat, write, halt.
    """

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
                        from ..vm import DiskProgramLoader, FlatFS, BlockDevice
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

    proc = kernel.spawn_process(
        "kernel-shell",
        priority=Priority.NORMAL,
        entry=_kernel_shell_entry,
    )
    return proc


def spawn_vm_process(kernel: Any, name: str, source: str,
                     stdin_fn=None, stdout_fn=None,
                     priority: Priority = Priority.NORMAL,
                     use_syscalls: bool = False) -> Process:
    """Spawn a process that runs VM assembly code.

    Creates a VirtualSystem, loads the assembled program, and executes
    it in a background thread. I/O goes through the console device.
    If use_syscalls=True, wires SYSCALL instruction to kernel's syscall table.
    """
    from ..vm import VirtualSystem

    output_log: list[str] = []

    def _handle_syscall(num, args):
        from ..kernel_syscall import SyscallNumber
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

    proc = kernel.spawn_process(
        name,
        priority=priority,
        entry=_vm_entry,
        metadata={"source": source, "output_log": output_log},
    )
    return proc


def run_program(kernel: Any, source: str, trace: bool = False) -> dict:
    """Run a VM assembly program through the kernel's device bus.

    Creates a VM CPU, wires it to the kernel's devices, and executes
    the assembled program. Returns output, trace, and step count.
    """
    from ..vm import CPU, Assembler, DeviceBus as VMBus

    vm_bus = VMBus()
    if hasattr(kernel._devices, '_table'):
        for name, dev in kernel._devices._table._devices.items():
            vm_bus.register(name, dev)
    elif hasattr(kernel._devices, '_devices'):
        for name, dev in kernel._devices._devices.items():
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


# --- Addon setup ---

def setup(kernel: Any) -> None:
    """Install shell/VM spawn capabilities on the kernel."""
    _fns = {
        "spawn_shell": lambda *a, **kw: spawn_shell(kernel, *a, **kw),
        "spawn_kernel_shell": lambda *a, **kw: spawn_kernel_shell(kernel, *a, **kw),
        "spawn_vm_process": lambda *a, **kw: spawn_vm_process(kernel, *a, **kw),
        "run_program": lambda *a, **kw: run_program(kernel, *a, **kw),
    }
    for attr_name, fn in _fns.items():
        setattr(kernel, attr_name, fn)
    kernel._addons["shell_ui"] = True
    logger.info("Shell UI addon installed")
