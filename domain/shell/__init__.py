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
from domain.shell._internal.vm import (
    InsFault,
    MemFault,
    X86VirtualSystem,
)
from domain.shell._internal.vm_permissions import Role


def __getattr__(name):
    _lazy = {
        "ShellREPL": "domain.shell._internal.repl",
        "get_bridge": "domain.shell._internal.vm_training_bridge",
        "RenderBridge": "domain.shell._internal.world_render",
        "RenderConfig": "domain.shell._internal.world_render",
        "NeuralRenderBridge": "domain.shell._internal.world_render",
        "WorldGrid": "domain.shell._internal.simulation",
        "SimScene": "domain.shell._internal.simulation",
        "Simulation": "domain.shell._internal.simulation",
        "WorldParams": "domain.shell._internal.simulation",
    }
    if name in _lazy:
        import importlib

        return getattr(importlib.import_module(_lazy[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "InsFault",
    "MemFault",
    "Role",
    "ShellREPL",
    "get_bridge",
    "RenderBridge",
    "RenderConfig",
    "NeuralRenderBridge",
    "WorldGrid",
    "SimScene",
    "Simulation",
    "WorldParams",
    "X86VirtualSystem",
]
