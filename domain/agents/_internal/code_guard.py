"""
Code guard — the ONE blocked-pattern list for model-supplied code.

Shared by agents.py's ToolRunner and tools.py's ToolRegistry before
running code in a subprocess: both surfaces execute code a model wrote,
so both must consult the SAME list — two lists is exactly the drift this
module exists to prevent (agents.py guarded, tools.py did not).

Config note: ``SecurityConfig.blocked_patterns`` is a separate, unused
field (it never fed ``is_allowed``); this list is the enforced one.
"""

from __future__ import annotations

import re

#: Patterns denied in model code before it may execute.
BLOCKED_PATTERNS: list[str] = [
    r"import\s+os\s*.*remove",
    r"import\s+shutil\s*.*rmtree",
    r"subprocess\s*\..*shell\s*=\s*True",
    r"__import__\s*\(\s*['\"]os['\"]",
    r"eval\s*\(",
    r"exec\s*\(",
    r"open\s*\([^)]*\br\/w",
    r"pickle",
    r"marshal",
    r"compile\s*\(",
]

_BLOCKED_RE = [re.compile(p) for p in BLOCKED_PATTERNS]


def is_allowed(code: str) -> tuple[bool, str]:
    """Return ``(True, "")`` when code may run, else ``(False, reason)``.

    A single loop over the single list: both tool surfaces call this, so
    a pattern added here is enforced everywhere at once.
    """
    for pattern in _BLOCKED_RE:
        if pattern.search(code):
            return False, f"Blocked pattern: {pattern.pattern}"
    return True, ""
