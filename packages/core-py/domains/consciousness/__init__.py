"""Backward-compatibility shim — canonical code lives in domain.cognition."""

from domain.cognition import *  # noqa: F401,F403
from domain.cognition import __all__ as _all  # noqa: F401

__all__ = list(_all)
