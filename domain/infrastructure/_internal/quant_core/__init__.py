"""Quantization core — INT8/INT4 matrix multiply kernels + scalar numerics.

Kernel symbols are resolved lazily (PEP 562): importing this package — or its
pure-numpy :mod:`quant_core.numeric` submodule — must never trigger the native
build in :mod:`quant_core.wrapper` (``_build_all()`` compiles C at import).
pugqeep's block codecs and the .slnc compiler consume ``numeric`` from
contexts where no compiler/toolchain should be required; kernel consumers
(``from ...quant_core import matmul_int8_c`` or ``...quant_core.wrapper``)
still get the same objects, just loaded on first attribute access.
"""

from __future__ import annotations

_KERNELS = ("matmul_int4_c", "matmul_int8_c", "matmul_int8_f32_c", "swiglu_f32_c")

__all__ = list(_KERNELS)


def __getattr__(name: str):
    if name in _KERNELS:
        from domain.infrastructure._internal.quant_core import wrapper as _wrapper

        return getattr(_wrapper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_KERNELS))
