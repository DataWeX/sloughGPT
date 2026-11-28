"""Worktree lifecycle — create a lane for a card, reap it when the card closes.

Every git fact is built in ``tmp_path``: the gates must be correct for a
repository nobody is using, never for the checkout the suite happens to sit in.
The failure this file exists for is measured: 65 registered worktrees, 9 with
their own ``node_modules``, and a primary checkout whose working tree had
silently reverted a landed ``done`` commit for three hours.

The gates are the contract, so they are what is tested:

* create claims the lane and refuses a path/branch that already exists;
* classify says ``dead`` only when merged, clean and unclaimed;
* reap refuses dirty, unmerged, protected, primary and claimed trees;
* reap refuses while the card is still open — the whole point of the card.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "cli", "src"))


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def make_repo(root: Path) -> Path:
    """A repo with one commit and an ``origin/main`` ref to compare against."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "checkout", "-q", "-b", "main")
    (root / "file.txt").write_text("0\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    return root


def make_lane(repo: Path, name: str, *, merged: bool = True, dirty: bool = False) -> Path:
    """A lane worktree at ``.wt-<name>``; merged=False gives it its own commit."""
    path = repo / f".wt-{name}"
    _git(repo, "worktree", "add", "-q", str(path), "-b", f"feat/{name}", "origin/main")
    if not merged:
        (path / "lane.txt").write_text("work\n", encoding="utf-8")
        _git(path, "add", ".")
        _git(path, "commit", "-q", "-m", "lane work")
    if dirty:
        (path / "wip.txt").write_text("uncommitted\n", encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    """A throwaway repo with no claims and no card store interference."""
    from core import worktree as wt_mod

    root = make_repo(tmp_path / "repo")
    monkeypatch.setattr(wt_mod.tree_mod, "active_claims", lambda: {})
    monkeypatch.setattr(wt_mod, "_claim_event", lambda *a, **k: None)
    monkeypatch.setattr(wt_mod, "claimed_card", lambda path: None)
    return root


class TestCreate:
    def test_creates_lane_and_branch_from_base(self, repo: Path):
        from core import worktree as wt_mod

        path, branch = wt_mod.create("card-1", "Shared node_modules", root=repo)

        assert path == repo / ".wt-shared-node-modules"
        assert path.is_dir()
        assert branch == "feat/shared-node-modules"
        assert _git(repo, "branch", "--show-current") == "main"
        assert "feat/shared-node-modules" in _git(repo, "branch", "--list", "feat/*")

    def test_refuses_an_existing_path(self, repo: Path):
        from core import worktree as wt_mod

        wt_mod.create("card-1", "Twice", root=repo)
        with pytest.raises(wt_mod.WorktreeError, match="already exists"):
            wt_mod.create("card-1", "Twice", root=repo)

    def test_refuses_an_existing_branch(self, repo: Path):
        from core import worktree as wt_mod

        wt_mod.create("card-1", "Taken", root=repo)
        _git(repo, "worktree", "remove", "--force", str(repo / ".wt-taken"))
        with pytest.raises(wt_mod.WorktreeError, match="branch feat/taken already exists"):
            wt_mod.create("card-2", "Taken", root=repo)

    def test_slug_is_stable_for_a_title(self):
        from core import worktree as wt_mod

        assert wt_mod.slug_for("Worktree lifecycle: grib wt new / wt done") == wt_mod.slug_for(
            "Worktree lifecycle: grib wt new / wt done"
        )
        assert wt_mod.slug_for("!!!") == "card"


class TestClassify:
    def test_merged_clean_unclaimed_is_dead(self, repo: Path):
        from core import worktree as wt_mod

        lane = make_lane(repo, "finished")
        items = {i.name: i for i in wt_mod.classify(root=repo)}

        assert items[".wt-finished"].state == "dead"
        assert items[".wt-finished"].reason() == ""
        assert lane.exists()

    def test_primary_checkout_is_never_dead(self, repo: Path):
        from core import worktree as wt_mod

        items = {i.name: i for i in wt_mod.classify(root=repo)}

        assert items[repo.name].primary
        assert items[repo.name].state == "idle"
        assert items[repo.name].reason() == "primary checkout"

    def test_dirty_lane_says_why(self, repo: Path):
        from core import worktree as wt_mod

        make_lane(repo, "wip", dirty=True)
        items = {i.name: i for i in wt_mod.classify(root=repo)}

        assert items[".wt-wip"].state == "idle"
        assert items[".wt-wip"].reason() == "uncommitted changes"

    def test_unmerged_lane_says_why(self, repo: Path):
        from core import worktree as wt_mod

        make_lane(repo, "ahead", merged=False)
        items = {i.name: i for i in wt_mod.classify(root=repo)}

        assert items[".wt-ahead"].state == "idle"
        assert items[".wt-ahead"].reason() == "branch not merged into origin/main"

    def test_live_claim_blocks_even_a_merged_tree(self, repo: Path, monkeypatch):
        from core import worktree as wt_mod

        lane = make_lane(repo, "owned")
        monkeypatch.setattr(wt_mod.tree_mod, "active_claims", lambda: {lane.resolve(): "ses_other"})

        items = {i.name: i for i in wt_mod.classify(root=repo)}
        assert items[".wt-owned"].state == "live"
        assert items[".wt-owned"].reason() == "claimed by ses_other"

    def test_own_claim_does_not_block(self, repo: Path, monkeypatch):
        from core import worktree as wt_mod

        lane = make_lane(repo, "mine")
        monkeypatch.setattr(wt_mod.tree_mod, "active_claims", lambda: {lane.resolve(): "ses_me"})

        items = {i.name: i for i in wt_mod.classify(root=repo, session="ses_me")}
        assert items[".wt-mine"].state == "dead"


class TestReap:
    def test_removes_a_finished_lane_and_its_branch(self, repo: Path):
        from core import worktree as wt_mod

        lane = make_lane(repo, "done-lane")
        message = wt_mod.reap(lane, root=repo)

        assert not lane.exists()
        assert "removed .wt-done-lane" in message
        assert "feat/done-lane" not in _git(repo, "branch", "--list", "feat/*")

    def test_refuses_dirty(self, repo: Path):
        from core import worktree as wt_mod

        lane = make_lane(repo, "dirty", dirty=True)
        with pytest.raises(wt_mod.WorktreeError, match="uncommitted changes"):
            wt_mod.reap(lane, root=repo)
        assert lane.exists()

    def test_refuses_unmerged(self, repo: Path):
        from core import worktree as wt_mod

        lane = make_lane(repo, "ahead", merged=False)
        with pytest.raises(wt_mod.WorktreeError, match="not merged"):
            wt_mod.reap(lane, root=repo)
        assert lane.exists()

    def test_refuses_another_sessions_claim(self, repo: Path, monkeypatch):
        from core import worktree as wt_mod

        lane = make_lane(repo, "owned")
        monkeypatch.setattr(wt_mod.tree_mod, "active_claims", lambda: {lane.resolve(): "ses_other"})
        with pytest.raises(wt_mod.WorktreeError, match="claimed by another live session"):
            wt_mod.reap(lane, root=repo)

    def test_refuses_the_primary_checkout(self, repo: Path):
        from core import worktree as wt_mod

        with pytest.raises(wt_mod.WorktreeError, match="primary checkout"):
            wt_mod.reap(repo, root=repo)

    def test_refuses_an_unregistered_directory(self, repo: Path, tmp_path: Path):
        from core import worktree as wt_mod

        stray = tmp_path / "stray"
        stray.mkdir()
        with pytest.raises(wt_mod.WorktreeError, match="not a registered worktree"):
            wt_mod.reap(stray, root=repo)

    def test_refuses_protected_user_data(self, repo: Path):
        from core import worktree as wt_mod

        lane = make_lane(repo, "fbdrv")  # -> .wt-fbdrv, the AGENTS.md user-data tree
        with pytest.raises(wt_mod.WorktreeError, match="is protected"):
            wt_mod.reap(lane, root=repo)

    def test_refuses_the_machine_managed_sync_worktree(self, repo: Path):
        """kanban-sync-wt is clean, detached and merged — and still not ours.

        It is the mirror worktree ``scripts/kanban_sync.sh`` pushes from, so
        the generic "merged + clean" rule calls it dead. Dogfooding ``wt ls``
        on the real repo listed it as reapable; this pins the correction.
        """
        from core import worktree as wt_mod

        # The real shape: detached, clean, at a commit already on origin/main.
        sync = repo / "kanban-sync-wt"
        _git(repo, "worktree", "add", "-q", "--detach", str(sync), "origin/main")
        assert sync.name in wt_mod.PROTECTED_NAMES
        with pytest.raises(wt_mod.WorktreeError, match="machine-managed"):
            wt_mod.reap(sync, root=repo)
        assert sync.exists()

    def test_refuses_while_the_card_is_still_open(self, repo: Path, monkeypatch):
        """The card gate: a merged, clean lane is still not reapable until closed."""
        from core import worktree as wt_mod

        lane = make_lane(repo, "open-card")
        monkeypatch.setattr(wt_mod, "claimed_card", lambda path: "20261128_103632")
        monkeypatch.setattr(wt_mod, "card_column", lambda card: "wip")

        with pytest.raises(wt_mod.WorktreeError, match="is 'wip', not done"):
            wt_mod.reap(lane, root=repo)
        assert lane.exists()

    def test_card_gate_passes_once_done(self, repo: Path, monkeypatch):
        from core import worktree as wt_mod

        lane = make_lane(repo, "closed-card")
        monkeypatch.setattr(wt_mod, "claimed_card", lambda path: "20261128_103632")
        monkeypatch.setattr(wt_mod, "card_column", lambda card: "done")

        wt_mod.reap(lane, root=repo)
        assert not lane.exists()

    def test_unknown_card_does_not_block(self, repo: Path, monkeypatch):
        """No grib means no answer — an unknown card never strands a lane."""
        from core import worktree as wt_mod

        lane = make_lane(repo, "unknown")
        monkeypatch.setattr(wt_mod, "claimed_card", lambda path: "card-x")
        monkeypatch.setattr(wt_mod, "card_column", lambda card: None)

        wt_mod.reap(lane, root=repo)
        assert not lane.exists()

    def test_force_overrides_the_gates_and_says_so(self, repo: Path):
        from core import worktree as wt_mod

        lane = make_lane(repo, "forced", merged=False, dirty=True)
        message = wt_mod.reap(lane, force=True, root=repo)

        assert not lane.exists()
        assert "(forced)" in message


class TestClaimSeam:
    """The claim seam runs for real here — the ``repo`` fixture stubs it out.

    Regression (found by dogfooding ``wt done --all`` on the real repo, not by
    a test): ``_claim_event`` did not accept ``board_dir`` while both callers
    passed it, so ``wt new`` and ``wt done`` raised ``TypeError`` on their first
    real run. Every unit test monkeypatched the seam, so the claim half of the
    feature had never executed. These two tests pin the signature and the
    passthrough that the stubs hid.
    """

    def test_claim_event_writes_to_the_board_dir_it_is_given(self, tmp_path: Path):
        from core import worktree as wt_mod

        wt_mod._claim_event(
            "claim",
            card_id="card-1",
            path=tmp_path / "tree",
            branch="feat/x",
            session="sess",
            board_dir=tmp_path,
        )
        lines = (tmp_path / "claims.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert '"card_id": "card-1"' in lines[0]
        assert '"event": "claim"' in lines[0]

    def test_create_passes_the_board_dir_through(self, repo: Path, tmp_path: Path, monkeypatch):
        from core import worktree as wt_mod

        seen: dict = {}
        monkeypatch.setattr(wt_mod, "_claim_event", lambda event, **kw: seen.update(kw))
        wt_mod.create("card-1", "Claim seam", root=repo, board_dir=tmp_path)
        assert seen["board_dir"] == tmp_path


class TestCommands:
    def test_wt_ls_lists_every_tree(self, repo: Path, monkeypatch):
        from commands import worktree as wt_cmd

        # The real command runs under the CLI logger; the test process gets the
        # plain console logger. Record the rows instead of re-testing formatting.
        fake = _FakeLog()
        monkeypatch.setattr(wt_cmd, "log", fake)
        make_lane(repo, "finished")
        code = wt_cmd.cmd_wt(_ns(subcommand="ls", root=repo))

        assert code == 0
        rows = {name: detail for name, detail in fake.rows}
        assert ".wt-finished" in rows
        assert "reapable" in rows[".wt-finished"]
        assert repo.name in rows  # the primary checkout is listed, never reapable

    def test_wt_done_all_reaps_only_dead_trees(self, repo: Path):
        from commands.worktree import cmd_wt

        dead = make_lane(repo, "dead-one")
        alive = make_lane(repo, "dirty-one", dirty=True)
        code = cmd_wt(_ns(subcommand="done", all=True, force=False, root=repo, path=None))

        assert code == 0
        assert not dead.exists()
        assert alive.exists()

    def test_wt_done_refusal_exits_nonzero(self, repo: Path):
        from commands.worktree import cmd_wt

        lane = make_lane(repo, "dirty", dirty=True)
        code = cmd_wt(_ns(subcommand="done", all=False, force=False, root=repo, path=lane))

        assert code == 1
        assert lane.exists()


class _FakeLog:
    """Records the rows a command would draw (no terminal, no theme)."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []
        self.messages: list[str] = []

    def header(self, *args, **kwargs) -> None:
        pass

    def status(self, name: str, detail: str = "", *args, **kwargs) -> None:
        self.rows.append((name, detail))

    def _note(self, message: str, *args, **kwargs) -> None:
        self.messages.append(str(message))

    info = success = warning = error = _note


class _ns:
    """Minimal argparse-like namespace for command tests."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
