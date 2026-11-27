"""Tests for core/tree.py — which checkout runs, and whether it is stale.

Every git fact here is built in ``tmp_path``: the resolver must be correct for
a checkout nobody is using, not for the one the test suite happens to sit in.
The two regressions this file exists for were both silent:

* ``find_repo_root``/``parents[N]`` off by one — the "tree" resolved to
  ``apps/`` and every later check failed against the wrong directory;
* ``git rev-parse --short --abbrev-ref HEAD`` printing *only* the branch, so
  the reported commit hash was really a branch name.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def make_checkout(root: Path, commits: int = 1, *, behind: int = 0) -> Path:
    """A minimal runnable checkout with ``commits`` commits and ``origin/main``.

    ``behind=N`` moves ``origin/main`` N commits ahead of ``HEAD`` (by resetting
    HEAD back), which is the stale-tree shape: local refs only, no fetch.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "cli.py").write_text("# launcher target\n", encoding="utf-8")
    (root / "sloughgpt").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "apps").mkdir(exist_ok=True)
    (root / "packages").mkdir(exist_ok=True)
    (root / "file.txt").write_text("0\n", encoding="utf-8")

    _git(root, "init", "-q")
    _git(root, "checkout", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "Test")

    for i in range(commits):
        (root / "file.txt").write_text(f"{i}\n", encoding="utf-8")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", f"commit {i}")

    head = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/main", head)
    if behind:
        for i in range(behind):
            (root / "file.txt").write_text(f"origin {i}\n", encoding="utf-8")
            _git(root, "add", "-A")
            _git(root, "commit", "-q", "-m", f"origin {i}")
        origin = _git(root, "rev-parse", "HEAD")
        _git(root, "update-ref", "refs/remotes/origin/main", origin)
        _git(root, "reset", "-q", "--hard", head)
    return root


@pytest.fixture(autouse=True)
def _clear_notice_cache():
    """``stale_notice`` caches per process; a stale cache would mask a regression."""
    from core import tree

    tree._notice_cache.clear()
    yield
    tree._notice_cache.clear()


class TestRepositoryRoot:
    def test_package_tree_is_the_repo_root_not_apps(self):
        """The off-by-one guard: ``parents[3]`` is ``apps/``, the root is above."""
        from core import tree

        root = tree.package_tree()
        assert (root / "cli.py").is_file()
        assert (root / "apps").is_dir()
        assert (root / "packages").is_dir()
        assert not (root / "src").exists() or (root / "packages").is_dir()

    def test_looks_like_tree_requires_all_four_markers(self, tmp_path):
        from core import tree

        assert tree.looks_like_tree(make_checkout(tmp_path / "ok"))
        for marker in ("cli.py", "sloughgpt"):
            (tmp_path / "ok" / marker).unlink()
            assert not tree.looks_like_tree(tmp_path / "ok")
            (tmp_path / "ok" / marker).write_text("x", encoding="utf-8")
        assert not tree.looks_like_tree(tmp_path / "missing")


class TestResolveTree:
    def test_uses_slo_tree_when_it_names_a_checkout(self, tmp_path):
        from core import tree

        checkout = make_checkout(tmp_path / "chosen")
        assert tree.resolve_tree({"SLO_TREE": str(checkout)}) == checkout.resolve()

    def test_ignores_slo_tree_when_it_is_not_a_checkout(self, tmp_path):
        """A typo must not silently run some other directory."""
        from core import tree

        (tmp_path / "empty").mkdir()
        assert tree.resolve_tree({"SLO_TREE": str(tmp_path / "empty")}) == tree.package_tree()

    def test_falls_back_to_the_tree_running_this_code(self):
        from core import tree

        assert tree.resolve_tree({}) == tree.package_tree()


class TestTreeInfo:
    def test_head_is_a_hash_and_branch_is_a_branch(self, tmp_path):
        """`rev-parse --short --abbrev-ref HEAD` prints only the branch."""
        from core import tree

        root = make_checkout(tmp_path / "tree")
        info = tree.tree_info(root)
        assert info.head == _git(root, "rev-parse", "--short", "HEAD")
        assert info.branch == "main"
        assert info.head != info.branch

    def test_behind_and_ahead_come_from_origin_main(self, tmp_path):
        from core import tree

        root = make_checkout(tmp_path / "tree", commits=1, behind=3)
        info = tree.tree_info(root)
        assert (info.behind, info.ahead) == (3, 0)
        assert not info.current
        assert info.stale_reason() and "3 commit(s) behind" in info.stale_reason()

    def test_current_tree_has_no_stale_reason(self, tmp_path):
        from core import tree

        info = tree.tree_info(make_checkout(tmp_path / "tree"))
        assert (info.behind, info.ahead) == (0, 0)
        assert info.current
        assert info.stale_reason() is None

    def test_dirty_is_only_computed_when_asked(self, tmp_path):
        """The launch path skips `git status`: 96 ms on the real repo."""
        from core import tree

        root = make_checkout(tmp_path / "tree")
        assert tree.tree_info(root).dirty is None
        assert tree.tree_info(root, dirty=True).dirty is False
        (root / "file.txt").write_text("changed\n", encoding="utf-8")
        assert tree.tree_info(root, dirty=True).dirty is True


class TestSummaryAndNotice:
    def test_summary_line_says_where_what_and_how_far(self, tmp_path):
        from core import tree

        info = tree.tree_info(make_checkout(tmp_path / "tree", behind=2), dirty=True)
        line = tree.summary_line(info)
        assert "2 behind" in line
        assert "main @" in line

    def test_notice_names_the_fix(self, tmp_path):
        from core import tree

        info = tree.tree_info(make_checkout(tmp_path / "tree", behind=1))
        notice = tree.stale_notice(info, {})
        assert notice and "sloughgpt tree --set latest" in notice

    def test_current_tree_is_silent(self, tmp_path):
        from core import tree

        info = tree.tree_info(make_checkout(tmp_path / "tree"))
        assert tree.stale_notice(info, {}) is None

    def test_env_switch_silences_the_notice(self, tmp_path):
        from core import tree

        info = tree.tree_info(make_checkout(tmp_path / "tree", behind=1))
        assert tree.stale_notice(info, {"SLO_NO_TREE_WARN": "1"}) is None


class TestLatestTree:
    def test_never_returns_a_behind_tree(self, tmp_path):
        from core import tree

        stale = tree.tree_info(make_checkout(tmp_path / "stale", behind=4), dirty=True)
        assert tree.latest_tree([stale]) is None

    def test_prefers_clean_main_then_newest(self, tmp_path):
        from core import tree

        lane = make_checkout(tmp_path / "lane")
        _git(lane, "checkout", "-q", "-b", "feat/x")
        lane_info = tree.tree_info(lane, dirty=True)
        main = make_checkout(tmp_path / "main")
        main_info = tree.tree_info(main, dirty=True)
        assert tree.latest_tree([lane_info, main_info]).path == main.resolve()

    def test_dirty_tree_is_not_adoptable(self, tmp_path):
        from core import tree

        root = make_checkout(tmp_path / "dirty")
        (root / "file.txt").write_text("uncommitted\n", encoding="utf-8")
        assert tree.latest_tree([tree.tree_info(root, dirty=True)]) is None

    def test_never_adopts_a_lane_worktree_over_the_main_checkout(self, tmp_path):
        """Two sessions on one checkout is the incident lane claims prevent.

        Dozens of clean lane worktrees live on this box; "latest worktree"
        must not mean "somebody else's".
        """
        from core import tree

        main = make_checkout(tmp_path / "main")
        lane = make_checkout(tmp_path / "lane")
        _git(lane, "checkout", "-q", "-b", "feat/lane")
        candidates = [
            tree.tree_info(lane, dirty=True),
            tree.tree_info(main, dirty=True),
        ]
        assert tree.latest_tree(candidates, primary=main).path == main.resolve()

    def test_returns_none_when_the_main_checkout_is_stale(self, tmp_path):
        from core import tree

        main = make_checkout(tmp_path / "main", behind=2)
        lane = make_checkout(tmp_path / "lane")
        assert (
            tree.latest_tree(
                [tree.tree_info(lane, dirty=True), tree.tree_info(main, dirty=True)],
                primary=main,
            )
            is None
        )


class TestAdoptedTree:
    def test_accepts_a_clean_current_checkout(self, tmp_path):
        from core import tree

        root = make_checkout(tmp_path / "good", behind=0)
        assert tree.adopted_tree(root, session="ses_me") == root.resolve()

    def test_rejects_a_tree_that_is_not_a_checkout(self, tmp_path):
        from core import tree

        (tmp_path / "nope").mkdir()
        with pytest.raises(ValueError, match="not a sloughgpt checkout"):
            tree.adopted_tree(tmp_path / "nope", session="ses_me")

    def test_rejects_a_behind_tree(self, tmp_path):
        from core import tree

        root = make_checkout(tmp_path / "stale", behind=2)
        with pytest.raises(ValueError, match="behind origin/main"):
            tree.adopted_tree(root, session="ses_me")

    def test_rejects_a_dirty_tree(self, tmp_path):
        from core import tree

        root = make_checkout(tmp_path / "dirty")
        (root / "file.txt").write_text("wip\n", encoding="utf-8")
        with pytest.raises(ValueError, match="uncommitted changes"):
            tree.adopted_tree(root, session="ses_me")

    def test_refuses_another_sessions_worktree(self, tmp_path, monkeypatch):
        """Two sessions on one checkout is the incident lane claims exist for."""
        from core import tree

        root = make_checkout(tmp_path / "theirs")
        monkeypatch.setattr(tree, "active_claims", lambda: {root.resolve(): "ses_other"})
        with pytest.raises(ValueError, match="claimed by another session"):
            tree.adopted_tree(root, session="ses_me")

    def test_allows_own_claimed_worktree(self, tmp_path, monkeypatch):
        from core import tree

        root = make_checkout(tmp_path / "mine")
        monkeypatch.setattr(tree, "active_claims", lambda: {root.resolve(): "ses_me"})
        assert tree.adopted_tree(root, session="ses_me") == root.resolve()


class TestRepointLauncher:
    def test_swaps_the_symlink_and_reports_old_and_new(self, tmp_path):
        from core import tree

        old = make_checkout(tmp_path / "old")
        new = make_checkout(tmp_path / "new")
        launcher = tmp_path / "bin" / "sloughgpt"
        launcher.parent.mkdir()
        os.symlink(old / "sloughgpt", launcher)

        was, now = tree.repoint_launcher(new, launcher)

        assert was == old.resolve()
        assert now == new
        assert launcher.is_symlink()
        assert Path(os.readlink(launcher)) == new / "sloughgpt"
        assert launcher.resolve() == (new / "sloughgpt").resolve()

    def test_refuses_a_regular_file_launcher(self, tmp_path):
        """A wrapper script is not ours to rewrite — say so instead of guessing."""
        from core import tree

        launcher = tmp_path / "sloughgpt"
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a symlink"):
            tree.repoint_launcher(make_checkout(tmp_path / "new"), launcher)

    def test_missing_launcher_is_an_error_not_a_guess(self, tmp_path, monkeypatch):
        from core import tree

        monkeypatch.setattr(tree, "launcher_path", lambda: None)
        with pytest.raises(ValueError, match="not on PATH"):
            tree.repoint_launcher(make_checkout(tmp_path / "new"))


class TestDescribeAll:
    def test_one_info_per_path(self, tmp_path):
        from core import tree

        paths = [make_checkout(tmp_path / f"t{i}") for i in range(3)]
        infos = tree.describe_all(paths, dirty=False)
        assert [info.path for info in infos] == [p.resolve() for p in paths]
        assert all(info.dirty is None for info in infos)


class TestCliWiring:
    """The resolver is only useful if the command that reports it exists."""

    @pytest.fixture
    def cli_mod(self):
        from apps.cli.src import cli

        return cli

    def test_tree_command_is_registered(self, cli_mod):
        assert "tree" in cli_mod.cli.commands

    def test_tree_command_declares_its_switches(self, cli_mod):
        """The command's vocabulary is declared once, in the decorator."""
        cmd = cli_mod.cli.commands["tree"]
        names = {name for opt in cmd.options for name in opt.names}
        assert {"--resolve", "--list", "--set"} <= names

    def test_resolve_prints_the_tree_for_scripts(self, cli_mod, capsys):
        from core import tree
        from core.framework import parse_args

        cmd = cli_mod.cli.commands["tree"]
        kwargs, _ = parse_args(["--resolve"], cmd.options, [])
        cmd.func(**kwargs)

        assert capsys.readouterr().out.strip() == str(tree.resolve_tree())

    def test_stale_warning_goes_to_stderr_and_never_stdout(self, cli_mod, monkeypatch, capsys):
        """A warning must not corrupt piped stdout (`--json` consumers)."""
        from core import tree

        monkeypatch.setattr(tree, "stale_notice", lambda *a, **k: "! stale tree")
        cli_mod._warn_if_stale_tree([])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "stale tree" in captured.err

    @pytest.mark.parametrize(
        "argv",
        [
            ["--json"],
            ["--quiet"],
            ["--help"],
            ["--version"],
            ["tree"],
            ["tree", "--list"],
        ],
    )
    def test_stays_quiet_where_it_would_be_noise(self, cli_mod, monkeypatch, capsys, argv):
        from core import tree

        monkeypatch.setattr(tree, "stale_notice", lambda *a, **k: "! stale tree")
        cli_mod._warn_if_stale_tree(argv)
        assert capsys.readouterr().err == ""

    def test_a_broken_resolver_never_fails_a_command(self, cli_mod, monkeypatch, capsys):
        from core import tree

        def boom(*_a, **_k):
            raise RuntimeError("git exploded")

        monkeypatch.setattr(tree, "stale_notice", boom)
        cli_mod._warn_if_stale_tree([])  # must not raise
        assert capsys.readouterr().err == ""


class TestSetCommand:
    """`--set` is the only action with side effects, so it is pinned hardest."""

    def _args(self, **kwargs):
        class Args:
            pass

        args = Args()
        args.resolve = False
        args.list = False
        args.set = None
        for key, value in kwargs.items():
            setattr(args, key, value)
        return args

    def test_repoints_the_launcher_and_says_old_then_new(self, tmp_path, monkeypatch, capsys):
        from commands.tree import cmd_tree
        from core import tree

        old = make_checkout(tmp_path / "old")
        new = make_checkout(tmp_path / "new")
        launcher = tmp_path / "bin" / "sloughgpt"
        launcher.parent.mkdir()
        os.symlink(old / "sloughgpt", launcher)
        monkeypatch.setattr(tree, "launcher_path", lambda: launcher)

        with pytest.raises(SystemExit) as exit_info:
            cmd_tree(self._args(set=str(new)))

        assert exit_info.value.code == 0
        assert launcher.resolve() == (new / "sloughgpt").resolve()

    def test_refusal_is_a_nonzero_exit_not_a_traceback(self, tmp_path, capsys):
        from commands.tree import cmd_tree

        with pytest.raises(SystemExit) as exit_info:
            cmd_tree(self._args(set=str(tmp_path / "not-a-checkout")))

        assert exit_info.value.code == 1

    def test_latest_adopts_the_clean_current_main_checkout(self, tmp_path, monkeypatch):
        from commands.tree import cmd_tree
        from core import tree

        good = make_checkout(tmp_path / "good")
        stale = make_checkout(tmp_path / "stale", behind=3)
        launcher = tmp_path / "bin" / "sloughgpt"
        launcher.parent.mkdir()
        os.symlink(stale / "sloughgpt", launcher)

        monkeypatch.setattr(tree, "launcher_path", lambda: launcher)
        monkeypatch.setattr(tree, "main_checkout", lambda *a, **k: good)
        monkeypatch.setattr(
            tree,
            "describe_all",
            lambda paths=None, **kw: [
                tree.tree_info(stale, dirty=True),
                tree.tree_info(good, dirty=True),
            ],
        )
        monkeypatch.setattr(tree, "active_claims", lambda: {})

        with pytest.raises(SystemExit) as exit_info:
            cmd_tree(self._args(set="latest"))

        assert exit_info.value.code == 0
        assert launcher.resolve() == (good / "sloughgpt").resolve()

    def test_latest_refuses_when_the_main_checkout_is_stale(self, tmp_path, monkeypatch):
        from commands.tree import cmd_tree
        from core import tree

        stale = make_checkout(tmp_path / "stale", behind=3)
        lane = make_checkout(tmp_path / "lane")
        launcher = tmp_path / "bin" / "sloughgpt"
        launcher.parent.mkdir()
        os.symlink(stale / "sloughgpt", launcher)

        monkeypatch.setattr(tree, "launcher_path", lambda: launcher)
        monkeypatch.setattr(tree, "main_checkout", lambda *a, **k: stale)
        monkeypatch.setattr(
            tree,
            "describe_all",
            lambda paths=None, **kw: [
                tree.tree_info(stale, dirty=True),
                tree.tree_info(lane, dirty=True),
            ],
        )
        monkeypatch.setattr(tree, "active_claims", lambda: {})

        with pytest.raises(SystemExit) as exit_info:
            cmd_tree(self._args(set="latest"))

        assert exit_info.value.code == 1
        assert launcher.resolve() == (stale / "sloughgpt").resolve()
