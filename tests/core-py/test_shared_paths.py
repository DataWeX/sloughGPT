"""Tests for domain.shared path resolution — find_repo_root / find_shared_root."""

from __future__ import annotations

import subprocess
from pathlib import Path

from domain.shared import find_repo_root, find_shared_root


def _init_repo(root: Path) -> None:
    """A minimal committed repo that ``git worktree add`` accepts."""
    (root / "apps").mkdir(parents=True)
    (root / "apps" / ".gitkeep").write_text("")
    (root / "packages").mkdir()
    (root / "packages" / ".gitkeep").write_text("")
    (root / "pyproject.toml").write_text("")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        check=True,
    )


class TestFindRepoRoot:
    def test_finds_root_by_marker_dirs(self, tmp_path):
        (tmp_path / "apps").mkdir()
        (tmp_path / "packages").mkdir()
        (tmp_path / "apps" / "web").mkdir()
        assert find_repo_root(tmp_path / "apps" / "web") == tmp_path


class TestFindSharedRoot:
    """The main checkout, not the worktree that asked.

    Regression: ``apps/api/server/main.py`` pointed ``HF_HOME`` at
    ``<root>/models/hf-cache`` using the *worktree* root. Every linked
    worktree therefore got an empty cache dir and the loader reported
    "No .slnc or .soul file found" while 3.8 GB of weights sat in the main
    checkout — the app could not run a model from a worktree at all.
    """

    def test_main_checkout_returns_itself(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo(repo)
        # find_repo_root walks *parents* of its start point, so start inside.
        assert find_shared_root(repo / "apps") == repo.resolve()

    def test_linked_worktree_returns_main_root(self, tmp_path):
        repo = tmp_path / "repo"
        _init_repo(repo)
        wt = tmp_path / "wt"
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "-q", str(wt)],
            check=True,
        )
        # The worktree has its own root ...
        assert find_repo_root(wt / "apps") == wt.resolve()
        # ... but shares the main checkout's machine-local trees.
        assert find_shared_root(wt / "apps") == repo.resolve()

    def test_non_repo_falls_back_to_repo_root(self, tmp_path):
        plain = tmp_path / "plain"
        (plain / "apps").mkdir(parents=True)
        (plain / "packages").mkdir()
        assert find_shared_root(plain / "apps") == find_repo_root(plain / "apps")
        assert find_repo_root(plain / "apps") == plain.resolve()

    def test_missing_git_falls_back(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        (repo / "apps").mkdir(parents=True)
        (repo / "packages").mkdir()

        def _no_git(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", _no_git)
        assert find_shared_root(repo / "apps") == repo.resolve()
