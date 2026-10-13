"""Tests for `scripts/test-doctor.py --patch-targets` (static mock-drift tier).

The static tier is the cheap half of mock drift: an AST scan for `@patch`
targets that an EAGER package re-export makes blind. Both directions matter.
A detector that misses the known case is decoration; one that fires on correct
patches is noise nobody will read, and gets disabled. So every true-positive
test is paired with a near-miss negative that differs only in the detail that
decides the verdict.

Folded in from scripts/lint_patch_targets.py (c0ebd5f31) — the standalone
duplicate of this check — which false-positived on PEP 562 lazy re-exports.
That guard is pinned here as `test_lazy_reexport_is_not_flagged`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "test-doctor.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("test_doctor_patch_targets", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_script()


GPU = "def get_accelerator():\n    return 'the-real-one'\n"

LAZY_INIT = (
    "_LAZY = {'get_accelerator': 'pkg._internal.gpu'}\n"
    "def __getattr__(name):\n"
    "    import importlib\n"
    "    m, _, a = _LAZY[name].rpartition('.')\n"
    "    return importlib.import_module(m).__dict__[a]\n"
)


def _synthetic(tmp_path: Path, *, init_line: str, consumer_import: str) -> Path:
    """Build the smallest repo exhibiting (or not exhibiting) the hazard."""
    (tmp_path / "pkg" / "_internal").mkdir(parents=True)
    (tmp_path / "apps").mkdir()
    (tmp_path / "tests").mkdir()

    (tmp_path / "pkg" / "_internal" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "_internal" / "gpu.py").write_text(GPU, encoding="utf-8")
    (tmp_path / "pkg" / "__init__.py").write_text(init_line, encoding="utf-8")
    (tmp_path / "apps" / "consumer.py").write_text(consumer_import + "\n", encoding="utf-8")
    (tmp_path / "tests" / "test_thing.py").write_text(
        "from unittest.mock import patch\n\n"
        '@patch("pkg._internal.gpu.get_accelerator")\n'
        "def test_thing(mock_acc):\n    pass\n",
        encoding="utf-8",
    )
    return tmp_path


EAGER_INIT = "from ._internal.gpu import get_accelerator\n"
HARZARD_CONSUMER = "from pkg import get_accelerator"


def test_detects_patch_blind_to_a_package_reexport(tool, tmp_path):
    """The hazard: prod reads the package copy, test patches the source."""
    repo = _synthetic(tmp_path, init_line=EAGER_INIT, consumer_import=HARZARD_CONSUMER)

    findings = tool._pt_find_stale(repo, [str(repo / "tests")])

    assert len(findings) == 1, findings
    (hit,) = findings
    assert hit["patch"] == "pkg._internal.gpu.get_accelerator"
    assert hit["immune_read"] == "pkg.get_accelerator"
    assert hit["read_by"] == ["apps/consumer.py"]
    assert hit["suggest"] == 'patch("pkg.get_accelerator")'


def test_does_not_flag_a_patch_at_the_binding_production_reads(tool, tmp_path):
    """The near miss: identical tree, but prod imports the deep module.

    Then the patch and the read target the same binding, so it lands.
    """
    repo = _synthetic(
        tmp_path,
        init_line=EAGER_INIT,
        consumer_import="from pkg._internal.gpu import get_accelerator",
    )

    assert tool._pt_find_stale(repo, [str(repo / "tests")]) == []


def test_does_not_flag_when_the_name_is_not_reexported(tool, tmp_path):
    """The other near miss: prod reads the package, but nothing copied the name.

    Without the re-export there is no second binding to go stale.
    """
    repo = _synthetic(tmp_path, init_line="", consumer_import=HARZARD_CONSUMER)

    # the import would fail at runtime, but statically there is no re-export
    assert tool._pt_find_stale(repo, [str(repo / "tests")]) == []


def test_aliased_reexport_is_not_treated_as_a_copy(tool, tmp_path):
    """`from ._internal.gpu import get_accelerator as ga` binds a new name.

    Nothing reads `pkg.get_accelerator`, so flagging it would be a guess.
    """
    repo = _synthetic(
        tmp_path,
        init_line="from ._internal.gpu import get_accelerator as ga\n",
        consumer_import=HARZARD_CONSUMER,
    )

    assert tool._pt_find_stale(repo, [str(repo / "tests")]) == []


def test_lazy_reexport_is_not_flagged(tool, tmp_path):
    """PEP 562 lazy re-export: a deep patch DOES land, so it is not a copy.

    This is the guard that the standalone linter lacked, and the sole reason
    the fold was non-trivial: `domain.inference...simple_embed` is lazy, and
    the linter false-positived on it. Access defers to the submodule, so
    `mock.patch` on the deep module reaches the reader.
    """
    repo = _synthetic(tmp_path, init_line=LAZY_INIT, consumer_import=HARZARD_CONSUMER)

    assert tool._pt_find_stale(repo, [str(repo / "tests")]) == []


def test_real_repo_still_reports_the_accelerator_case(tool):
    """Pin the live instance that motivated this: it must stay findable.

    If this starts failing, either the accelerator patch target was corrected
    (good — update the expectation) or the detector regressed (bad).
    """
    patches = {f["patch"] for f in tool._pt_find_stale(_REPO)}

    assert "domain.slolib._internal.gpu.get_accelerator" in patches, (
        "the stale accelerator patch target is no longer detected; the two "
        "precision tests in tests/server/ depend on it"
    )


def test_advisory_cli_stays_green_despite_findings(tool, tmp_path, capsys, monkeypatch):
    """Advisory stays green so historical findings do not break the build.

    Unlike the linter's `--strict` (exit 1), this tier always returns 0: it
    cannot judge relevance, so a hard gate would just get bypassed.
    """
    repo = _synthetic(tmp_path, init_line=EAGER_INIT, consumer_import=HARZARD_CONSUMER)
    monkeypatch.setattr(tool, "REPO_ROOT", repo)

    assert tool.cmd_patch_targets() == 0

    out = capsys.readouterr().out
    assert "pkg._internal.gpu.get_accelerator" in out
    assert "pkg.get_accelerator" in out
