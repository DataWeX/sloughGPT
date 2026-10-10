"""
Collection-health guards (card d2803a3c).

Two properties this suite depends on that a normal test run cannot see: a
test that executes never proves that its SIBLING still imports.

1. Duplicate basenames resolve to distinct modules. 73 test basenames are
   shared across tests/, tests/core-py/, tests/server/ and tests/api-server/
   with no __init__.py in any of them, and the files are genuinely different
   (test_training_pipeline.py is 5585 bytes in tests/ and 17253 in
   tests/core-py/). --import-mode=importlib derives each module name from
   its full path, so every copy collects. Under the older prepend mode the
   second copy collided with the first and WHICH ONE SURVIVED DEPENDED ON
   ARG ORDER — the same file ran green alone and red in a full suite. That is
   the failure this pins; it produced 53 collection errors and is why card
   d2803a3c existed.

2. Zero collection errors repo-wide. --continue-on-collection-errors in
   pytest.ini keeps one bad import from aborting the other ~48k tests, which
   is correct — before that flag a single uncollectable file turned every
   bare `python -m pytest` into a guaranteed no-op. The cost is that a broken
   file no longer stops anything, so nothing catches a NEW one. This does.

Both are marked slow: a full collection takes ~3 minutes, so they are
excluded from the default `-m "not slow"` run. Run them when touching test
layout, pytest config, or imports that affect collection:

    python -m pytest tests/test_collection_health.py -m slow -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _collect(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run `pytest --collect-only` from the repo root under the root ini."""
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        timeout=1200,
    )


def _error_lines(proc: subprocess.CompletedProcess[str]) -> list[str]:
    out = f"{proc.stdout}\n{proc.stderr}"
    return [line for line in out.splitlines() if line.startswith("ERROR")]


def _duplicate_groups() -> dict[str, list[Path]]:
    """Basenames that appear in more than one directory under tests/."""
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(_REPO_ROOT.glob("tests/**/test_*.py")):
        by_name[path.name].append(path)
    return {name: paths for name, paths in by_name.items() if len(paths) > 1}


@pytest.mark.slow
class TestCollectionHealth:
    def test_full_collection_has_no_errors(self):
        """The whole suite must collect, not merely run.

        This is the guard for --continue-on-collection-errors: without it a
        new file that fails to import prints as an ERROR and is skipped while
        every other test still passes, so the suite stays green around a
        silently missing module.
        """
        proc = _collect([])

        errors = _error_lines(proc)
        assert not errors, (
            f"{len(errors)} file(s) failed to import during full collection:\n"
            + "\n".join(errors[:20])
        )
        assert proc.returncode == 0, f"collection exited {proc.returncode}:\n{proc.stdout[-2000:]}"

    def test_import_mode_resolves_duplicate_basenames(self):
        """Shared basenames must all collect in one run, in any arg order.

        Sampling three groups keeps this to ~15s; the full-collection test
        above is the exhaustive version. The two directions both matter —
        arg order is what used to decide which copy won.
        """
        groups = _duplicate_groups()
        assert groups, (
            "expected the shared test basenames to still exist; if the layout "
            "was deliberately de-duplicated, delete this guard rather than "
            "letting it pass vacuously"
        )

        # Deterministic sample: three groups spanning the widest directory sets.
        sample = sorted(groups)[:3]
        for direction in (sample, list(reversed(sample))):
            files = [
                str(path.relative_to(_REPO_ROOT)) for name in direction for path in groups[name]
            ]
            proc = _collect(files)

            errors = _error_lines(proc)
            assert not errors, (
                "duplicate basenames collided (arg order changed the result) "
                f"for {direction}:\n" + "\n".join(errors)
            )
