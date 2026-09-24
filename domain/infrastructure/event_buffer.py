"""Re-export of the canonical infra module (was core-py)."""

from domain.infrastructure._internal.event_buffer import *  # noqa: F401,F403

try:  # noqa
    from domain.infrastructure._internal.event_buffer import __all__  # noqa: F401
except ImportError:
    pass
