"""Backward-compatibility shim — canonical code lives in domain."""

from domain import *  # noqa: F401,F403
from domain import __all__ as _all  # noqa: F401

__all__ = list(_all)
