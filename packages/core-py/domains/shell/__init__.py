"""Backward-compatibility shim — imports from the new ``domain.shell`` package."""

from domain.shell import (
    ConsoleIO,
    DaitRuntime,
    Kernel,
    MemoryIO,
    NeuralKernel,
    Process,
    ProcessState,
    Resource,
    ShellCommands,
    ShellIO,
    ShellState,
    capture_output,
)

_dait_instance = None


def get_dait_runtime():
    global _dait_instance
    if _dait_instance is None:
        _dait_instance = DaitRuntime()
    return _dait_instance


from domain.shell._internal import (  # noqa: F401
    commands,
    init,
    kernel,
    kernel_scheduler,
    vm_training_bridge,
)

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
