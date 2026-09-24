"""Re-export of the canonical infra module (was core-py)."""

from domain.infrastructure._internal.weight_loader import *  # noqa: F401,F403

try:  # noqa
    from domain.infrastructure._internal.weight_loader import __all__  # noqa: F401
except ImportError:
    pass
