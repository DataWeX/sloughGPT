"""GPU accelerator — Metal/CUDA/CPU dispatch for training ops."""

from __future__ import annotations

from domains.training.gpu.accelerator import get_accelerator

__all__ = ["get_accelerator"]
