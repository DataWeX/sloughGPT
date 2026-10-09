"""slolib — Unified tensor library (autograd, GPU acceleration, inference).

Public API:
    get_accelerator, reset_accelerator, set_accelerator_precision
    to_gpu, from_gpu, gelu, silu, softmax, benchmark_accelerators
"""

from domain.slolib._internal.gpu import (
    benchmark_accelerators,
    from_gpu,
    gelu,
    reset_accelerator,
    silu,
    softmax,
    to_gpu,
)

# API-layer names resolve lazily so patches on _internal.gpu stay visible
# at call time (AGENT_SYNC facade-patch gotcha).
_LAZY_IMPORTS = {
    "get_accelerator": ("._internal.gpu", "get_accelerator"),
    "set_accelerator_precision": ("._internal.gpu", "set_accelerator_precision"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_accelerator",
    "reset_accelerator",
    "set_accelerator_precision",
    "to_gpu",
    "from_gpu",
    "gelu",
    "silu",
    "softmax",
    "benchmark_accelerators",
]
