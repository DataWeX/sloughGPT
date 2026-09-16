"""shell — Interactive Dait shell (kernel, processes, VM, TUI).

Public API:
    Kernel, NeuralKernel, Process, ProcessState
    ShellState, ShellIO, ConsoleIO, MemoryIO, capture_output
    ShellCommands, DaitRuntime, Resource, get_dait_runtime
"""

from domain.shell._internal.commands import (
    ShellCommands,
)
from domain.shell._internal.io import (
    ConsoleIO,
    MemoryIO,
    ShellIO,
    capture_output,
)
from domain.shell._internal.kernel import (
    Kernel,
    NeuralKernel,
)
from domain.shell._internal.kernel_process import (
    Process,
    ProcessState,
)
from domain.shell._internal.runtime import (
    DaitRuntime,
    Resource,
)
from domain.shell._internal.state import (
    ShellState,
)

_dait_instance = None


def get_dait_runtime() -> DaitRuntime:
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
