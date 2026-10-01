"""Guard: the arken -> avion rename is complete inside ``packages/avion``.

Card 122109a9: the rename was half-done — the directory and ``src/avion``
moved, but imports, ``pyproject`` name and identity strings still said
``arken``, so the whole suite was uncollectable
(``ModuleNotFoundError: No module named 'arken'``).

These two tests keep the retired name from creeping back into the package.
(The ``feat/avion-journeys`` branch adds a repo-wide AST guard in
``test_naming.py``; this is the cheap package-local version that lands
with the fix itself. The ``packages/voyager`` shim and
``scripts/run_journey_tests.py`` are deliberately out of scope — they
re-export the voyager -> arken -> avion chain by design.)
"""

from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

_IMPORT_LINE = re.compile(r"^\s*(?:from|import)\s+arken\b")
_NAME_LINE = re.compile(r'^name\s*=\s*"([^"]+)"', re.MULTILINE)


def _stale_imports() -> list[str]:
    offenders: list[str] = []
    for py in sorted(PKG.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        for lineno, line in enumerate(
            py.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _IMPORT_LINE.match(line):
                offenders.append(f"{py.relative_to(PKG)}:{lineno}: {line.strip()}")
    return offenders


def test_no_stale_arken_imports() -> None:
    offenders = _stale_imports()
    assert not offenders, "stale arken imports (rename incomplete):\n" + "\n".join(
        offenders
    )


def test_pyproject_declares_avion() -> None:
    text = (PKG / "pyproject.toml").read_text(encoding="utf-8")
    m = _NAME_LINE.search(text)
    assert m is not None, "pyproject.toml has no name = ..."
    assert m.group(1) == "avion", f"pyproject name = {m.group(1)!r}, expected 'avion'"
