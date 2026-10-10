#!/usr/bin/env python3
"""pytest plugin: the BOOM half of ``test-doctor --dead-seam``.

Loaded with ``-p dead_seam_plugin`` on a run whose ``PYTHONPATH`` includes
``scripts/``, and armed by ``SLO_DEAD_SEAM_TARGET=<dotted.path.to.symbol>``.

It replaces exactly one symbol with a callable that raises, so a parent run can
tell a *live* seam from a *dead* one by comparing two executions of the same
test::

    baseline (this plugin unarmed)   boom (armed)
        pass   +   pass   ->  DEAD SEAM: the symbol never ran, so patching it
                             is a no-op and the test's assertion is vacuous
        pass   +   fail   ->  live: production really reaches the symbol
        fail   +   anything ->  uninformative: the baseline was already red

Two guards keep the verdict honest rather than merely green:

* a symbol that is **not callable** (a path constant, a flag) cannot be
  booby-trapped this way -- a function substitute would simply be *returned* on
  every read, the run would pass, and we would report "dead" while the value is
  in fact read constantly.  Such a symbol is written as ``not-probeable`` and
  the parent reports it as inconclusive, never as dead.
* anything that fails to resolve (import error, missing attribute) is written
  as ``unresolved`` for the same reason.

The marker goes to a FILE (``SLO_DEAD_SEAM_MARKER``) rather than stdout, so
pytest's own capture cannot swallow it and the parent cannot confuse test
output with probe status.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

#: Prefix every marker line starts with; the parent re's on this.
MARKER_PREFIX = "SLO_DEAD_SEAM_PROBE "


class DeadSeamHit(RuntimeError):
    """Raised when a probed symbol is actually executed during the test run."""


def _write_marker(fields: dict[str, str]) -> None:
    path = os.environ.get("SLO_DEAD_SEAM_MARKER")
    if not path:
        return
    line = MARKER_PREFIX + " ".join(f"{k}={v}" for k, v in fields.items())
    try:
        Path(path).write_text(line + "\n")
    except OSError:  # pragma: no cover - marker is best-effort diagnostics
        pass


def pytest_configure(config):  # noqa: ARG001 - pytest hook, signature fixed by pytest
    target = os.environ.get("SLO_DEAD_SEAM_TARGET")
    if not target:
        return

    mod_name, _, attr = target.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
        original = getattr(mod, attr)
    except Exception as exc:  # noqa: BLE001 - report the class, not the message
        _write_marker({"target": target, "state": "unresolved", "err": type(exc).__name__})
        return

    if not callable(original):
        _write_marker({"target": target, "state": "not-probeable", "reason": "not-callable"})
        return

    def _boom(*_args, **_kwargs):
        raise DeadSeamHit(f"dead-seam probe: {target} was executed")

    _boom.__name__ = attr
    _boom.__qualname__ = attr
    _boom.__doc__ = f"BOOM (dead-seam probe armed over {target})"
    setattr(mod, attr, _boom)
    _write_marker({"target": target, "state": "armed"})
