"""Backward-compatibility shim — canonical code lives in domain.training._internal.gpu.accelerator."""

from domain.training._internal.gpu.accelerator import get_accelerator  # noqa: F401

__all__ = ["get_accelerator"]
