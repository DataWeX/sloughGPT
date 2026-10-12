"""Make ``infra_lib`` importable when it is not installed editable.

Flat layout: the package directory sits directly under ``packages/infra-lib/`,
so the directory to add is the package root rather than a ``src/`` child.
Without this, 33 tests in this directory died at collection with
``ModuleNotFoundError: infra_lib`` and never ran at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]

try:  # pragma: no cover - trivial import guard
    import infra_lib  # noqa: F401
except ImportError:  # pragma: no cover
    if str(PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(PKG_ROOT))
