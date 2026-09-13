"""Backward-compatibility — allows ``from domains.shell.runtime import ...``."""
from domain.shell._internal.runtime import (
    DaitRuntime,
    Resource,
    APIServerProcess,
    _probe_api,
    _default_api_url,
)
