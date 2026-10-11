"""Regression lock for the doc-vs-code parity gate.

``scripts/check_docs_api_parity.py`` is a hard CI step (``ci.yml`` runs it and
fails the build on exit 1). It had been silently reporting 22 tested routers
as *untested* for a long stretch, and nothing caught it, because the gate had
no test of its own.

Root cause: ``TEST_DIRS`` named a directory that no longer existed
(``apps/api/server/tests``), and ``collect_tests`` filtered entries with
``if d.exists()`` — so a missing directory was skipped without a word and the
gate searched roughly half the suite. Coverage looked thin; the report was a
lie.

The gate can therefore fail in two different ways, and both are locked here:

1. **Config rot** — ``TEST_DIRS`` drifting away from the real test roots.
   A missing entry is now a loud ``setup error`` (exit 2), not a quiet
   shrink. Exit 2 vs exit 1 matters: 1 means "found gaps", 2 means "I could
   not do my job" — conflating them hides the failure behind an unrelated
   red.
2. **Silent under-search** — the matcher consulting fewer directories than it
   was configured with. Both configured roots must contribute files, or the
   "untested" list is meaningless.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_docs_api_parity.py"


def _load_gate():
    """Import the gate script from ``scripts/`` (it is not a package)."""
    spec = importlib.util.spec_from_file_location("check_docs_api_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load_gate()


def test_all_test_dirs_exist(gate):
    """Every configured test root must exist.

    This is the config-level lock for the original defect: if someone renames
    or moves ``tests/server`` / ``tests/api-server``, this fails in the suite
    as well as in the gate itself, rather than silently narrowing coverage.
    """
    missing = [str(d) for d in gate.TEST_DIRS if not d.exists()]
    assert missing == [], f"TEST_DIRS names a directory that does not exist: {missing}"


def test_both_real_test_dirs_contribute_files(gate):
    """Both configured roots must actually supply test files.

    ``TEST_DIRS`` existing is not enough — if a root stops matching the
    ``test_*.py`` pattern (or a glob breaks) it contributes nothing and the
    untested list inflates with false positives. One file per root is the
    floor that proves each is really being searched.
    """
    for d in gate.TEST_DIRS:
        found = sorted(p.name for p in d.glob("test_*.py"))
        assert found, f"{d} contributed zero test files — the gate is not searching it"


def test_collect_tests_consults_every_configured_dir(gate, tmp_path, monkeypatch):
    """The matcher must see every directory it was configured with.

    Regression for the silent-skip: give the matcher two directories, each
    holding the test for a *different* router. If either directory were
    dropped, its router would come back untested and this test fails.
    """
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "test_alpha_router.py").write_text("# alpha\n", encoding="utf-8")
    (root_b / "test_beta_router.py").write_text("# beta\n", encoding="utf-8")

    monkeypatch.setattr(gate, "TEST_DIRS", (root_a, root_b))

    findings = gate.Findings()
    findings.routers_on_disk = ["alpha_router", "beta_router"]
    gate.collect_tests(findings)

    assert findings.untested == [], (
        "routers reported untested even though their test file exists in a "
        f"configured dir: {findings.untested}"
    )


def test_missing_test_dir_fails_loudly(gate, monkeypatch, capsys, tmp_path):
    """A vanished test root is a setup error (exit 2), never a quiet shrink.

    Guards the guard: the ``d.exists()`` filter that caused the original
    silent skip is still in ``collect_tests``, so the loud failure has to
    stay upstream of it in ``main``.
    """
    monkeypatch.setattr(
        gate,
        "TEST_DIRS",
        gate.TEST_DIRS + (tmp_path / "definitely-not-here",),
    )
    # main() parses sys.argv; under pytest that is pytest's own argv.
    monkeypatch.setattr(sys, "argv", ["check_docs_api_parity.py"])

    rc = gate.main()

    captured = capsys.readouterr()
    assert rc == 2, f"expected setup-error exit 2, got {rc}"
    assert "setup error" in captured.err
    assert "definitely-not-here" in captured.err
