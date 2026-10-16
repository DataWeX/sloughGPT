"""scripts/python must make THIS checkout's copy win for every packages/ project.

AGENTS one-shared-copy rule: ``packages/*`` exists exactly once, in the working
tree you are editing. Two different things compete with it, and only a
``PYTHONPATH`` entry outranks either:

1. **A pip-installed site-packages snapshot.** Card ``dd670ece``: ``downcraft``
   was installed into the conda env as a Sep-18 copy whose "already downloaded"
   branch was unconditional, while the tree copy (``6c626f043``) re-downloads
   when a "complete" state entry has no file on disk. Because the launcher only
   put the repo *root* on ``PYTHONPATH`` — never ``packages/`` — every consumer
   of ``scripts/python`` (e2e, benchmarks) silently ran stale code, reported
   success for a download that wrote no file, and the merged reget fix was dead
   in practice.

2. **The ``sloughgpt`` editable install.** Its ``_EditableFinder`` carries a
   ``MAPPING`` of importable names to absolute paths, all hardcoded to the MAIN
   TREE checkout. From any worktree, ``import mogdb`` therefore resolved to the
   main tree's copy rather than the one being edited — testing a worktree's
   changes against the wrong code (card ``aecf2bb1``).

Both are silent, so the launcher's package list is a ratchet rather than an
optimisation, and this module verifies the *behaviour* rather than parsing the
shell: every importable ``packages/`` project must actually resolve from this
tree. That holds regardless of how the launcher expresses the list.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "python"
PACKAGES_DIR = REPO_ROOT / "packages"


def _repo_packages() -> list[str]:
    """Importable projects under ``packages/`` (flat or src layout)."""
    names = []
    for entry in sorted(PACKAGES_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / entry.name / "__init__.py").is_file():  # flat: packages/<n>/<n>
            names.append(entry.name)
        elif (entry / "src" / entry.name / "__init__.py").is_file():  # src layout
            names.append(entry.name)
    return names


def _import_origin(name: str) -> Path | None:
    """Where the launcher resolves ``name``, or None when not importable."""
    try:
        res = subprocess.run(
            [str(LAUNCHER), "-c", f"import {name}; print({name}.__file__)"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None  # missing third-party deps: out of scope for this ratchet
    out = res.stdout.strip().splitlines()
    return Path(out[-1]) if out else None


def test_every_repo_package_resolves_from_this_tree():
    """No packages/ project may resolve to site-packages or another checkout.

    This is the whole point of the launcher list: a worktree must run the code
    it contains. A stale installed snapshot and the editable install's
    main-tree MAPPING both defeat that, and neither leaves a visible error.
    """
    stray: list[tuple[str, Path]] = []
    checked = 0

    for name in _repo_packages():
        origin = _import_origin(name)
        if origin is None:
            continue
        checked += 1
        try:
            origin.resolve().relative_to(PACKAGES_DIR.resolve())
        except ValueError:
            stray.append((name, origin))

    assert checked, "no packages/ project imported — the ratchet is not exercising anything"
    assert not stray, (
        "scripts/python resolves these to something other than this checkout's "
        f"packages/ tree: {stray}. An installed snapshot or the sloughgpt "
        "editable install's main-tree MAPPING is winning; add the project root "
        "to the launcher's list (see the module docstring, cards dd670ece "
        "and aecf2bb1)."
    )


def test_repo_packages_are_present():
    """The scan itself must keep finding projects, or it passes vacuously."""
    assert _repo_packages(), "no importable project found under packages/"


def test_launcher_is_idempotent():
    """A pre-set PYTHONPATH must not gain duplicate entries."""
    extra = str(PACKAGES_DIR / "downcraft")
    res = subprocess.run(
        [str(LAUNCHER), "-c", "import os, sys; print(os.environ['PYTHONPATH'])"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": f"/nonexistent:{extra}", "PATH": "/usr/bin:/bin"},
        timeout=60,
    )
    if res.returncode != 0:
        pytest.skip("launcher could not run in this stripped environment")
    entries = res.stdout.strip().split(":")
    assert len(entries) == len(set(entries)), f"duplicate PYTHONPATH entries: {entries}"


@pytest.mark.skipif(
    not (PACKAGES_DIR / "downcraft" / "downcraft" / "__init__.py").is_file(),
    reason="tree copy of downcraft absent in this checkout",
)
def test_downcraft_reget_guard_is_live():
    """The staleness the snapshot hid must be gone from the tree copy too."""
    source = (PACKAGES_DIR / "downcraft" / "downcraft" / "__init__.py").read_text()
    assert "re-downloading" in source, (
        "the tree copy lost its re-download guard; the launcher is now pointing "
        "at code with the same defect the snapshot had."
    )
