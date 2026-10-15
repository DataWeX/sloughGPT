"""scripts/python must make the repo's shared copy beat any installed snapshot.

AGENTS one-shared-copy rule: ``packages/*`` exists exactly once, in the tree.
A pip-installed copy in site-packages silently shadows it, and a ``PYTHONPATH``
entry is the only thing that beats site-packages — so the launcher has to list
the repo path explicitly.

This is not theoretical. Card ``dd670ece``: ``downcraft`` was pip-installed
into the conda env as a Sep-18 snapshot whose "already downloaded" branch was
unconditional, while the tree copy (``6c626f043``, an ancestor of main) re-
downloads when a "complete" state entry has no file on disk. Because the
launcher only put the repo *root* on ``PYTHONPATH`` — never ``packages/`` —
``python -m downcraft`` resolved to site-packages, so every consumer (e2e,
benchmarks, anything using ``scripts/python``) ran stale code, silently
reported success for a download that wrote no file, and the merged reget fix
was dead in practice.

The launcher's list is therefore a ratchet, not an optimisation: adding a
``packages/`` entry when a new installed copy appears is the fix, and this
test fails until it does.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "scripts" / "python"
PACKAGES_DIR = Path(__file__).resolve().parents[2] / "packages"


def _launcher_repo_paths() -> set[str]:
    """Repo package paths the launcher explicitly puts on PYTHONPATH."""
    text = LAUNCHER.read_text()
    # `$ROOT/packages/<name>` (flat) or `$ROOT/packages/<name>/src` (src layout).
    # The class stops at shell metachars ($ { } : " ') so the capture is the
    # path itself, not the `${PYTHONPATH:+…}` expansion that follows it.
    return set(re.findall(r"\$ROOT/(packages/[A-Za-z0-9_./-]+)", text))


def _python_packages() -> list[str]:
    """``packages/`` entries that actually contain an importable package."""
    names = []
    for entry in sorted(PACKAGES_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / entry.name / "__init__.py").is_file():  # flat layout
            names.append(entry.name)
        elif (entry / "src" / entry.name / "__init__.py").is_file():  # src layout
            names.append(entry.name)
    return names


def _installed_site_packages() -> Path:
    """site-packages of the interpreter the launcher resolves to."""
    resolved = subprocess.run(
        [str(LAUNCHER), "--resolve"], capture_output=True, text=True, check=True
    ).stdout.strip()
    probe = subprocess.run(
        [resolved, "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(probe)


def test_every_installed_repo_package_is_pinned_by_the_launcher():
    """An installed copy must be shadowed by its tree path in scripts/python."""
    listed = _launcher_repo_paths()
    site_pkgs = _installed_site_packages()

    missing = [
        name
        for name in _python_packages()
        if (site_pkgs / name).exists() and f"packages/{name}" not in listed
    ]

    assert not missing, (
        "scripts/python lets site-packages shadow the shared tree copy for: "
        f"{missing}. The one-shared-copy rule makes the tree copy canonical, "
        "and only a PYTHONPATH entry beats site-packages — add the repo path "
        "to the launcher's list (see the module docstring, card dd670ece)."
    )


def test_listed_paths_exist_in_the_tree():
    """The launcher's ratchet must not reference paths that were moved."""
    for rel in _launcher_repo_paths():
        assert (PACKAGES_DIR.parent / rel).exists(), (
            f"scripts/python pins {rel}, which no longer exists — the package "
            "moved or was renamed, so the pin is dead and its shadow is back."
        )


def test_repo_copy_actually_wins_at_runtime():
    """The end-to-end invariant: the imported module comes from the tree."""
    result = subprocess.run(
        [str(LAUNCHER), "-c", "import downcraft; print(downcraft.__file__)"],
        capture_output=True,
        text=True,
        check=True,
    )
    imported = Path(result.stdout.strip())
    assert "site-packages" not in imported.parts, (
        f"import downcraft resolved to {imported} — the installed snapshot "
        "won over the tree copy (card dd670ece)."
    )
    assert imported.is_relative_to(PACKAGES_DIR), (
        f"import downcraft resolved to {imported}, which is outside packages/."
    )


@pytest.mark.skipif(
    not (PACKAGES_DIR / "downcraft" / "downcraft" / "__init__.py").is_file(),
    reason="tree copy of downcraft absent in this checkout",
)
def test_tree_copy_carries_the_reget_guard():
    """The staleness the shadow hid must be gone from the tree copy too."""
    source = (PACKAGES_DIR / "downcraft" / "downcraft" / "__init__.py").read_text()
    assert "re-downloading" in source, (
        "the tree copy lost its re-download guard; the launcher pin is now "
        "pointing at code with the same defect the snapshot had."
    )
