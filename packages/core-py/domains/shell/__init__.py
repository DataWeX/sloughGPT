"""Backward-compatibility shim — imports from the new ``domain.shell`` package."""

import domain.shell as _domain_shell
from domain.shell import (
    Kernel,
    NeuralKernel,
    Process,
    ProcessState,
    ShellState,
    ShellIO,
    ConsoleIO,
    MemoryIO,
    capture_output,
    ShellCommands,
    DaitRuntime,
    Resource,
)

_dait_instance = None

def get_dait_runtime():
    global _dait_instance
    if _dait_instance is None:
        _dait_instance = DaitRuntime()
    return _dait_instance

__all__ = [
    "Kernel",
    "NeuralKernel",
    "Process",
    "ProcessState",
    "ShellState",
    "ShellIO",
    "ConsoleIO",
    "MemoryIO",
    "capture_output",
    "ShellCommands",
    "DaitRuntime",
    "Resource",
    "get_dait_runtime",
]
