"""
Path guard — canonicalize-then-check for model-supplied file paths.

Shared by agents.py (ToolRunner) and tools.py (ToolRegistry): both accept
paths a model chose, so both must enforce the SAME blocklist before any
filesystem access — two copies of this logic is exactly how the drift
happened (agents.py had the guard, tools.py had none).

Order matters: realpath() FIRST, then match — checking the raw string
before resolving is the classic ``..``/symlink bypass.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

# Subdirectories of $HOME the model may never read via a file tool.
SENSITIVE_HOME_DIRS = (".ssh", ".gnupg", ".aws", ".kube", ".docker")

# Basenames blocked anywhere on disk (passwd-style files, private keys).
SENSITIVE_BASENAMES = frozenset(
    {
        "shadow",
        "gshadow",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "credentials",
    }
)


def path_violation(path: str, allowed_directories: Sequence[str] = ()) -> str | None:
    """Deny model-supplied paths: sensitive locations, then allowlist.

    Returns a human-readable denial reason, or ``None`` when the path is
    permitted. When ``allowed_directories`` is non-empty it is enforced as
    an allowlist (resolved paths, prefix-bound on separators); otherwise
    only the sensitive blocklist applies.
    """
    resolved = os.path.realpath(path)
    home = os.path.realpath(os.path.expanduser("~"))
    for name in SENSITIVE_HOME_DIRS:
        blocked = os.path.join(home, name)
        if resolved == blocked or resolved.startswith(blocked + os.sep):
            return f"sensitive directory denied: {path}"
    base = os.path.basename(resolved)
    if base.startswith(".env") or base in SENSITIVE_BASENAMES:
        return f"sensitive file denied: {path}"
    if allowed_directories:
        roots = [os.path.realpath(a) for a in allowed_directories]
        if not any(resolved == root or resolved.startswith(root + os.sep) for root in roots):
            return f"path outside allowed_directories: {path}"
    return None
