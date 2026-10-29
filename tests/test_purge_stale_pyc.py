"""Tests for scripts/purge_stale_pyc.py — orphan/stale bytecode classification.

Fixtures are built in tmp_path only; nothing here touches the repo's real
__pycache__ trees (the sessionstart guard in tests/conftest.py covers those).
"""

from __future__ import annotations

import py_compile
from pathlib import Path

from scripts.purge_stale_pyc import classify, purge_stale_pyc


def _compile_fake_module(tmp_path: Path, name: str = "sample") -> Path:
    """A real source + real pyc keyed on it; returns the source path."""
    src = tmp_path / f"{name}.py"
    src.write_text(f"x = {name!r}\n")
    py_compile.compile(str(src), doraise=True)
    return src


def _find_pyc(directory: Path) -> Path:
    pyCs = list(directory.rglob("*.pyc"))
    assert len(pyCs) == 1, f"expected exactly one pyc, got {pyCs}"
    return pyCs[0]


def test_orphan_pyc_is_removed_on_apply(tmp_path: Path) -> None:
    """A pyc with no .py sibling anywhere near it is orphaned and deleted."""
    src = _compile_fake_module(tmp_path)
    pyc = _find_pyc(tmp_path)
    src.unlink()  # source gone; cache left behind

    report = purge_stale_pyc([tmp_path], apply=True)

    assert (pyc, f"no source: {src}") in report.orphaned
    assert not pyc.exists()
    assert report.deleted == 1


def test_valid_pyc_is_kept(tmp_path: Path) -> None:
    """A pyc whose co_filename points at an existing source survives."""
    _compile_fake_module(tmp_path)
    pyc = _find_pyc(tmp_path)

    report = purge_stale_pyc([tmp_path], apply=True)

    assert report.kept == 1
    assert report.orphaned == [] and report.stale == []
    assert pyc.exists()


def test_dry_run_deletes_nothing(tmp_path: Path) -> None:
    """Default (apply=False) classifies but never unlinks."""
    src = _compile_fake_module(tmp_path)
    orphan_dir = tmp_path / "sub"
    orphan_dir.mkdir()
    (orphan_dir / "ghost.cpython-312.pyc").write_bytes(b"not really bytecode")

    report = purge_stale_pyc([tmp_path], apply=False)

    assert len(report.orphaned) == 1  # ghost, no ghost.py beside it
    assert report.deleted == 0
    assert src.exists()
    assert (orphan_dir / "ghost.cpython-312.pyc").exists()


def test_stale_pyc_after_move_is_classified_and_removed(tmp_path: Path) -> None:
    """mv preserves mtime+size: sibling exists but co_filename cites the old
    path -> stale (the card's traceback-cites-dead-paths failure mode)."""
    old = tmp_path / "old"
    old.mkdir()
    src = _compile_fake_module(old)
    pyc = _find_pyc(old)

    new = tmp_path / "new"
    (new / "__pycache__").mkdir(parents=True)
    new_src = new / src.name
    src.rename(new_src)  # mv both halves; mtime+size preserved
    new_pyc = new / "__pycache__" / pyc.name
    pyc.rename(new_pyc)

    report = classify([tmp_path])

    assert report.kept == 0
    assert report.orphaned == []
    assert len(report.stale) == 1
    stale_pyc, old_path = report.stale[0]
    assert stale_pyc == new_pyc
    assert old_path == str(src)  # the dead co_filename
    assert new_src.exists()  # the trap: sibling present, cache still wrong

    applied = purge_stale_pyc([tmp_path], apply=True)
    assert applied.deleted == 1
    assert not new_pyc.exists()
    assert new_src.exists()  # source untouched


def test_guard_never_raises_on_garbage(tmp_path: Path) -> None:
    """purge_stale_pyc degrades instead of raising — the sessionstart hook
    must not be able to break a test session."""
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.cpython-312.pyc").write_bytes(b"\x00" * 4)

    report = purge_stale_pyc([tmp_path], apply=True)  # unreadable, no sibling

    assert len(report.orphaned) == 1


def test_relative_coefilename_with_existing_source_is_kept(tmp_path: Path) -> None:
    """co_filename recorded relative to the compile-time cwd (repo root
    convention, e.g. 'tests/x.py') must NOT be resolved to source_dir/x.py
    — that false positive churn-deleted valid caches (measured 2026-10-12)."""
    src_dir = tmp_path / "tests"
    src_dir.mkdir()
    # Compile with a cwd-relative path so co_filename is stored relative.
    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        src = tmp_path / "tests" / "relmod.py"
        src.write_text("y = 1\n")
        py_compile.compile("tests/relmod.py", doraise=True)
    finally:
        os.chdir(old_cwd)
    pyc = _find_pyc(tmp_path)

    report = classify([tmp_path])

    assert report.kept == 1
    assert report.stale == [] and report.orphaned == []
    assert pyc.exists()
