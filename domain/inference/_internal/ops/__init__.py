"""
ops/__init__.py — Modular inference operations.

C is the base layer. numpy is the fallback for ops without C implementations.

Both SloTransformer and NativeEngine use these ops. The dispatch is per-op:
each op tries C first, falls back to numpy if C is unavailable.

Available ops:
  - matmul(a, b)           → C (Accelerate cblas_sgemm) or numpy
  - layernorm(x, w, b)     → numpy (C future)
  - rmsnorm(x, w)          → numpy (C future)
"""

from __future__ import annotations

from .matmul import matmul
from .layernorm import layernorm
from .rmsnorm import rmsnorm

__all__ = ["matmul", "layernorm", "rmsnorm"]
