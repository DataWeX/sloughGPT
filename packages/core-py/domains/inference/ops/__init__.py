"""Backward-compatibility shim — imports from the new ``domain.inference._internal.ops`` package."""

from domain.inference._internal.ops import layernorm, matmul, rmsnorm

__all__ = ["matmul", "layernorm", "rmsnorm"]
