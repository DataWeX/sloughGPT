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
    X86VirtualSystem,
)

_dait_instance = None

_LAZY_IMPORTS = {
    "ShellREPL": ("._internal.repl", "ShellREPL"),
    "InsFault": ("._internal.vm", "InsFault"),
    "MemFault": ("._internal.vm", "MemFault"),
    "Role": ("._internal.vm_permissions", "Role"),
    "get_bridge": ("._internal.vm_training_bridge", "get_bridge"),
    "WorldGrid": ("._internal.simulation", "WorldGrid"),
    "SimScene": ("._internal.simulation", "SimScene"),
    "Simulation": ("._internal.simulation", "Simulation"),
    "WorldParams": ("._internal.simulation", "WorldParams"),
    "RenderBridge": ("._internal.world_render", "RenderBridge"),
    "RenderConfig": ("._internal.world_render", "RenderConfig"),
    "NeuralRenderBridge": ("._internal.world_render", "NeuralRenderBridge"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return mod if attr is None else getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "X86VirtualSystem",
    "ShellREPL",
    "InsFault",
    "MemFault",
    "Role",
    "get_bridge",
    "WorldGrid",
    "SimScene",
    "Simulation",
    "WorldParams",
    "RenderBridge",
    "RenderConfig",
    "NeuralRenderBridge",
]
