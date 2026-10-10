"""Tests for ``domain.shared.data_root`` and live data-tree isolation.

The live ``<repo>/data`` tree is production state, not scratch space. Before
``data_root()`` existed, ``domain/agents/_internal/system.py`` built
``AGENTS_DIR`` from ``__file__`` arithmetic and pinned it to
``<repo>/data/agents`` unconditionally, so every pytest run persisted its
``auto-*``/``create-*``/``upd-*`` fixtures into the tree the UI reads — 916 of
the 1033 files under ``data/agents/`` matched test id patterns.

``SLO_DATA_DIR`` is the redirect, and it has to be established at conftest
import time: ``AGENTS_DIR`` is evaluated when a test module is *imported*,
which is collection time, before any fixture can monkeypatch an env var. These
tests pin that chain — that ``data_root`` honours the redirect, and that the
agent store therefore never resolves into the live tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

from domain.shared import data_root, find_repo_root


def test_data_root_defaults_to_the_repo_data_dir(monkeypatch) -> None:
    monkeypatch.delenv("SLO_DATA_DIR", raising=False)
    assert data_root() == find_repo_root() / "data"


def test_data_root_honours_the_env_redirect(monkeypatch, tmp_path) -> None:
    redirect = tmp_path / "scratch"
    monkeypatch.setenv("SLO_DATA_DIR", str(redirect))
    assert data_root() == redirect


def test_data_root_trims_the_redirect(monkeypatch, tmp_path) -> None:
    """Same trimming ``db_pool._data_root`` applies.

    A stray space in a shell export must not silently create a directory
    literally named ``' /tmp/...'``.
    """
    monkeypatch.setenv("SLO_DATA_DIR", f"  {tmp_path}  ")
    assert data_root() == tmp_path


def test_data_root_reads_the_environment_on_every_call(monkeypatch, tmp_path) -> None:
    """A value captured once would miss a later redirect.

    The conftest fixture redirects per test, so a module-level constant here
    would pin every test in a session to the first one's directory.
    """
    monkeypatch.delenv("SLO_DATA_DIR", raising=False)
    first = data_root()
    monkeypatch.setenv("SLO_DATA_DIR", str(tmp_path / "second"))
    second = data_root()
    assert second == tmp_path / "second"
    assert second != first


def test_agent_dir_never_resolves_into_the_live_tree() -> None:
    """An import-time data path must land outside ``<repo>/data`` during tests.

    ``AGENTS_DIR`` is computed when ``domain.agents._internal.system`` is first
    imported, which for ``test_agent_system.py`` happens at *collection* time —
    before the autouse fixture can redirect anything. It holds because the root
    conftest sets ``SLO_DATA_DIR`` before any test module is imported.
    """
    from domain.agents._internal.system import AGENTS_DIR

    live = find_repo_root() / "data"
    resolved = Path(AGENTS_DIR).resolve()
    assert not resolved.is_relative_to(live), (
        f"AGENTS_DIR={resolved} resolves inside the live data tree, so test "
        f"fixtures would persist into production state"
    )


def test_agent_dir_is_not_built_from_file_arithmetic() -> None:
    """Static guard: the data path derives from ``data_root()``, never ``__file__``.

    An import-based assertion cannot see *how* the value was derived, and a
    ``__file__`` join reads no environment at all — it defeats any redirect no
    matter what the fixture sets, which is exactly how 916 test-pattern files
    accumulated in ``data/agents/``. Same reasoning as the training-history
    guard in ``test_adaptive_infra.py``: assert on the source, not the value.
    """
    src_path = find_repo_root() / "domain" / "agents" / "_internal" / "system.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "AGENTS_DIR"
    ]
    assert values, "AGENTS_DIR is no longer assigned in system.py — update this guard"

    derived = ast.unparse(values[0])
    assert "data_root" in derived, (
        f"AGENTS_DIR must derive from data_root() so SLO_DATA_DIR applies; got: {derived}"
    )
    assert "__file__" not in derived, (
        f"a __file__-based join ignores SLO_DATA_DIR entirely; got: {derived}"
    )
