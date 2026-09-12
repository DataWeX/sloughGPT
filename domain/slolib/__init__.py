"""slolib — Unified tensor library (autograd, GPU acceleration, inference).

Public API:
    get_accelerator, reset_accelerator, set_accelerator_precision
    to_gpu, from_gpu, gelu, silu, softmax, benchmark_accelerators
"""

from domain.slolib._internal.gpu import (
    get_accelerator,
    reset_accelerator,
    set_accelerator_precision,
    to_gpu,
    from_gpu,
    gelu,
    silu,
    softmax,
    benchmark_accelerators,
)

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
