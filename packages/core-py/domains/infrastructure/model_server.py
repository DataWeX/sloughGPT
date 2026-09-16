"""Backward-compatibility shim."""

from domain.infrastructure._internal.model_server import *  # noqa: F401,F403

try:
    from domain.infrastructure._internal.model_server import __all__  # noqa: F401
except ImportError:
    pass
import sys as _sys

_mod = _sys.modules[__name__]
_real = _sys.modules.get("domain.infrastructure._internal.model_server")
if _real is not None:
    for _k in dir(_real):
        if not _k.startswith("__"):
            setattr(_mod, _k, getattr(_real, _k))
