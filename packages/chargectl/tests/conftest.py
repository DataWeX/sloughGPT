"""Make ``src/`` importable when chargectl is not installed editable."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

try:  # pragma: no cover - trivial import guard
    import chargectl  # noqa: F401
except ImportError:  # pragma: no cover
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
