"""Backward-compatibility shim — imports from the new ``domain.slolib`` package."""

from domain.slolib import (
    benchmark_accelerators,
    from_gpu,
    gelu,
    get_accelerator,
    reset_accelerator,
    set_accelerator_precision,
    silu,
    softmax,
    to_gpu,
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
