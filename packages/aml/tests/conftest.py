"""Make ``aml`` importable when it is not installed editable.

Flat layout: the package directory sits directly under ``packages/aml/``, so
the directory to add is the package root rather than a ``src/`` child. Without
this the whole suite dies at collection with ``ModuleNotFoundError: aml`` and
not one test ever runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]

try:  # pragma: no cover - trivial import guard
    import aml  # noqa: F401
except ImportError:  # pragma: no cover
    if str(PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(PKG_ROOT))
