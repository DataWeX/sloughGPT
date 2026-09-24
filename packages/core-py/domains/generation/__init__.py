"""Backward-compatibility shim — canonical code lives in domain.generation."""

from domain.generation import *  # noqa: F401,F403
from domain.generation import __all__ as _all  # noqa: F401

__all__ = list(_all)
