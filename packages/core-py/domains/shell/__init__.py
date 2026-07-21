"""
Dait — interactive shell for SloughGPT.

Provides an interactive shell that manages models,
processes (training jobs), memory (context/knowledge), souls, datasets,
system resources, and init services through a unified interface.

Includes:
  - ShellREPL: interactive shell with 80+ commands, pipelines, backgrounding
  - DaitRuntime: top-level runtime layer (kernel + init system)
  - Kernel: process/resource manager
  - WindowManager: curses-based multi-pane TUI
  - InitSystem: multi-runlevel service manager with boot/shutdown/respawn

Usage:
    from domains.shell import DaitRuntime, get_dait_runtime
    rt = get_dait_runtime()
    await rt.repl.run()

Or via CLI:
    sloughgpt shell
"""

from typing import Optional

from .kernel import DaitRuntime, Kernel, Process, ProcessState, Resource
from .repl import ShellREPL
from .commands import ShellCommands
from .state import ShellState
from .io import ShellIO, ConsoleIO, MemoryIO, capture_output
from .audit import ShellAuditLogger, get_shell_audit_logger
from .permissions import ShellPermissions, Risk
from .window_manager import WindowManager, get_window_manager, reset_window_manager
from .init import InitSystem, ServiceDef, ServiceInstance, ServiceManager, get_init_system, reset_init_system
from .devices import DeviceManager, AIDevice, create_default_devices
from .vm import VirtualCPU, VMRunner, ProgramLoader, Instruction, VMFault, Halt, MemFault, InsFault, SysFault, HELLO_ASM, COUNTER_ASM, FIB_ASM, COLLATZ_ASM, self_test
from .vfs import VFS, VFSEntry, VFSGeneratedFile, VFSDirectory, get_vfs, reset_vfs


_dait_instance: Optional[DaitRuntime] = None


def get_dait_runtime() -> DaitRuntime:
    global _dait_instance
    if _dait_instance is None:
        _dait_instance = DaitRuntime()
    return _dait_instance


__all__ = [
    "DaitRuntime",
    "Kernel",
    "Process",
    "ProcessState",
    "Resource",
    "ShellREPL",
    "ShellCommands",
    "ShellState",
    "WindowManager",
    "get_window_manager",
    "reset_window_manager",
    "get_dait_runtime",
    "InitSystem",
    "ServiceDef",
    "ServiceInstance",
    "ServiceManager",
    "get_init_system",
    "reset_init_system",
    "DeviceManager",
    "AIDevice",
    "create_default_devices",
    "VirtualCPU",
    "VMRunner",
    "ProgramLoader",
    "Instruction",
    "VMFault",
    "Halt",
    "MemFault",
    "InsFault",
    "SysFault",
    "HELLO_ASM",
    "COUNTER_ASM",
    "FIB_ASM",
    "COLLATZ_ASM",
    "self_test",
    "ShellPermissions",
    "Risk",
]
