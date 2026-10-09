"""Import guard — avion's event logger must stay stdlib-only.

Two layers:

* static: AST-scan the logger modules for anything outside the stdlib,
  ``__future__``, and ``avion`` itself;
* runtime: import the modules in a clean interpreter and assert no
  ``domain`` (or other third-party) module got pulled into the process.

This is what makes avion's logger usable independently of any other
logger in the repo — the dependency edge simply cannot exist.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import avion

AVION_SRC = Path(avion.__file__).resolve().parents[1]

# Modules that make up avion's self-contained logging/event layer.
LOGGER_MODULES = (
    "events/journal.py",
    "events/logger.py",
    "events/sinks.py",
    "events/recorder.py",
    "events/models.py",
    "logging/structured.py",
)

ALLOWED = set(sys.stdlib_module_names) | {"avion", "__future__"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: same package, fine
                continue
            if node.module:
                found.add(node.module.split(".")[0])
    return found


def test_logger_modules_import_only_stdlib_and_avion():
    violations: dict[str, set[str]] = {}
    for rel in LOGGER_MODULES:
        path = AVION_SRC / "avion" / rel
        assert path.exists(), f"missing logger module: {rel}"
        foreign = _imports(path) - ALLOWED
        if foreign:
            violations[rel] = foreign
    assert not violations, f"logger modules import non-stdlib code: {violations}"


def test_importing_the_logger_pulls_in_no_domain_code():
    """Clean-interpreter proof: loading the event layer never drags in
    ``domain/`` or any other repo subsystem.

    Diff-based: only modules that appear *because of* the avion import
    count, so interpreter bootstrap noise (sitecustomize, editable
    finders) cannot mask or fake a violation.
    """
    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import avion.events.journal, avion.events.logger, "
        "avion.events.sinks, avion.events.recorder, avion.logging.structured\n"
        "allowed = set(sys.stdlib_module_names) | {'avion'}\n"
        "new = {m for m in set(sys.modules) - before "
        "if m.split('.')[0] not in allowed}\n"
        "print(','.join(sorted(new)))"
    )
    env = {**os.environ, "PYTHONPATH": str(AVION_SRC)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=True,
    )
    assert result.stdout.strip() == "", (
        f"event-logger import pulled in non-stdlib modules: {result.stdout.strip()}"
    )


def test_event_package_public_api_covers_the_new_surface():
    """The self-contained API must be importable from the package root."""
    from avion.events import (  # noqa: F401
        BusSink,
        CallbackSink,
        EventJournal,
        EventLogger,
        EventSink,
        StdlibSink,
    )
