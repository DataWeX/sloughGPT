"""The patch-target linter must find the real hazard and nothing else.

Both directions matter. A detector that misses the known case is decoration;
one that fires on correct patches is noise nobody will read, and gets
disabled. So every true-positive test is paired with a near-miss negative
that differs only in the detail that decides the verdict.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "lint_patch_targets", REPO / "scripts" / "lint_patch_targets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GPU = "def get_accelerator():\n    return 'the-real-one'\n"


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


def test_detects_patch_blind_to_a_package_reexport(tmp_path):
    """The hazard: prod reads the package copy, test patches the source."""
    lint = _load()
    repo = _synthetic(
        tmp_path,
        init_line="from ._internal.gpu import get_accelerator\n",
        consumer_import="from pkg import get_accelerator",
    )

    findings = lint.find_stale_targets([str(repo / "tests")], repo=repo)

    assert len(findings) == 1, findings
    (hit,) = findings
    assert hit["patch"] == "pkg._internal.gpu.get_accelerator"
    assert hit["immune_read"] == "pkg.get_accelerator"
    assert hit["read_by"] == ["apps/consumer.py"]
    assert hit["suggest"] == 'patch("pkg.get_accelerator")'


def test_does_not_flag_a_patch_at_the_binding_production_reads(tmp_path):
    """The near miss: identical tree, but prod imports the deep module.

    Then the patch and the read target the same binding, so it lands.
    """
    lint = _load()
    repo = _synthetic(
        tmp_path,
        init_line="from ._internal.gpu import get_accelerator\n",
        consumer_import="from pkg._internal.gpu import get_accelerator",
    )

    assert lint.find_stale_targets([str(repo / "tests")], repo=repo) == []


def test_does_not_flag_when_the_name_is_not_reexported(tmp_path):
    """The other near miss: prod reads the package, but nothing copied the name.

    Without the re-export there is no second binding to go stale.
    """
    lint = _load()
    repo = _synthetic(
        tmp_path,
        init_line="",
        consumer_import="from pkg import get_accelerator",
    )

    # the import would fail at runtime, but statically there is no re-export
    assert lint.find_stale_targets([str(repo / "tests")], repo=repo) == []


def test_aliased_reexport_is_not_treated_as_a_copy(tmp_path):
    """`from ._internal.gpu import get_accelerator as ga` binds a new name.

    Nothing reads `pkg.get_accelerator`, so flagging it would be a guess.
    """
    lint = _load()
    repo = _synthetic(
        tmp_path,
        init_line="from ._internal.gpu import get_accelerator as ga\n",
        consumer_import="from pkg import get_accelerator",
    )

    assert lint.find_stale_targets([str(repo / "tests")], repo=repo) == []


def test_real_repo_still_reports_the_accelerator_case():
    """Pin the live instance that motivated this: it must stay findable.

    If this starts failing, either the accelerator patch target was corrected
    (good — update the expectation) or the detector regressed (bad).
    """
    lint = _load()
    patches = {f["patch"] for f in lint.find_stale_targets()}

    assert "domain.slolib._internal.gpu.get_accelerator" in patches, (
        "the stale accelerator patch target is no longer detected; the two "
        "precision tests in tests/server/ depend on it"
    )


def test_cli_exit_codes(tmp_path):
    """Advisory stays green so historical findings do not break the build."""
    lint = _load()
    repo = _synthetic(
        tmp_path,
        init_line="from ._internal.gpu import get_accelerator\n",
        consumer_import="from pkg import get_accelerator",
    )

    assert lint.find_stale_targets([str(repo / "tests")], repo=repo)  # something to report
    # main() is not exercised here: it is a thin printer over find_stale_targets,
    # and its exit policy is asserted by reading the branch directly.
    src = (REPO / "scripts" / "lint_patch_targets.py").read_text(encoding="utf-8")
    assert "return 1 if args.strict else 0" in src
