"""
Shell domain configuration.

Reads from environment variables with sensible defaults.

Transport mode (card ad4f5ec2 — "shell binary runs core in-process"):

* **local (default)** — the shell talks to the domain core in-process.
  No HTTP, no server required.
* **remote** — the shell talks to an API server over HTTP. Enabled
  explicitly by setting ``MAN_API_URL`` or by ``shell --remote``.
"""

from __future__ import annotations

import os

DEFAULT_API_BASE = os.environ.get("MAN_API_URL", "http://localhost:8000")

# Explicit in-process→HTTP override set by ``shell --remote`` (takes
# effect process-wide; MAN_API_URL also implies remote).
_FORCE_REMOTE = False


def get_api_base() -> str:
    """Return the API base URL, respecting MAN_API_URL env var."""
    return os.environ.get("MAN_API_URL", DEFAULT_API_BASE)


def is_remote_mode() -> bool:
    """True when shell commands should go over HTTP to an API server.

    Local in-process mode is the default; HTTP is demoted to an explicit
    opt-in (``MAN_API_URL`` set or ``--remote`` passed).
    """
    return _FORCE_REMOTE or bool(os.environ.get("MAN_API_URL"))


def set_remote_mode(on: bool) -> None:
    """Force (or unforce) remote HTTP mode for this process (``--remote``)."""
    global _FORCE_REMOTE
    _FORCE_REMOTE = bool(on)
