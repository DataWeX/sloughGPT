"""GPU accelerator — Metal/CUDA/CPU dispatch for training ops."""

from __future__ import annotations

from domain.training._internal.gpu.accelerator import get_accelerator

__all__ = ["get_accelerator"]
