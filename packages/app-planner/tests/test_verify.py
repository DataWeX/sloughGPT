"""
Tests for planner verify — three-way diff of journal note <-> board card <-> git.

The note card (YAML frontmatter: status, board, landed) is the source of
truth; verify reports any drift against the board card and against the
commits actually present/landed in a git repo.
"""

import json
import subprocess
from pathlib import Path

import pytest
from app_planner.verify import parse_frontmatter, verify_note


def _write_note(
    path: Path,
    title: str = "Card X",
    status: str = "done",
    board: str | None = None,
    landed: str | None = None,
) -> Path:
    lines = ["---", f"title: {title}", f"status: {status}"]
    if board:
        lines.append(f"board: {board}")
    if landed:
        lines.append(f"landed: {landed}")
    lines += ["---", "", "# body", ""]
    path.write_text("\n".join(lines))
    return path


def _write_board(path: Path, cards: list[dict]) -> Path:
    with open(path, "w") as f:
        f.write(json.dumps({"_schema": "header"}) + "\n")  # schema line must be skipped
        for c in cards:
            f.write(json.dumps(c) + "\n")
    return path


def _card(cid: str, title: str, column: str) -> dict:
    return {"id": cid, "title": title, "column": column, "notes": []}


@pytest.fixture
def git_repo(tmp_path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    (repo / "a.txt").write_text("a")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "a"], check=True)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return repo, sha


@pytest.fixture
def note(tmp_path) -> Path:
    return _write_note(tmp_path / "n.md")


@pytest.fixture
def board(tmp_path) -> Path:
    return _write_board(tmp_path / "board.jsonl", [_card("uuid-1", "Card X", "done")])


def test_parse_frontmatter_keeps_extra_keys(tmp_path):
    p = _write_note(tmp_path / "n.md", board="uuid-1", landed="abc1234 + def5678")
    meta = parse_frontmatter(p)
    assert meta["status"] == "done"
    assert meta["board"] == "uuid-1"
    assert meta["landed"] == "abc1234 + def5678"
    assert meta["title"] == "Card X"


def test_verify_all_consistent(note, board, git_repo):
    repo, sha = git_repo
    _write_note(note.parent / "n.md", board="uuid-1", landed=sha)
    findings = verify_note(note, board, repo)
    assert findings, "must produce findings"
    assert all(f.ok for f in findings), [f for f in findings if not f.ok]


def test_verify_status_column_mismatch(note, board):
    _write_board(board, [_card("uuid-1", "Card X", "todo")])  # note says done
    findings = verify_note(note, board, Path("."))
    bad = [f for f in findings if f.kind == "status" and not f.ok]
    assert bad, f"status drift not detected: {findings}"
    assert "done" in bad[0].detail and "todo" in bad[0].detail


def test_verify_missing_board_uuid(note, board, tmp_path):
    _write_note(note.parent / "n.md", board="nope-999")
    findings = verify_note(note, board, Path("."))
    bad = [f for f in findings if f.kind == "board" and not f.ok]
    assert bad, "missing board uuid not detected"


def test_verify_matches_card_by_title_when_no_board_key(note, board):
    _write_note(note.parent / "n.md", status="done")  # no board: key
    findings = verify_note(note, board, Path("."))
    bad = [f for f in findings if not f.ok]
    assert not bad, f"title fallback should match the card: {bad}"


def test_verify_landed_sha_missing_from_repo(note, board):
    _write_note(note.parent / "n.md", board="uuid-1", landed="0" * 40)
    findings = verify_note(note, board, Path("."))
    bad = [f for f in findings if f.kind == "git" and not f.ok]
    assert bad, "missing sha not detected"


def test_verify_landed_sha_not_ancestor(note, board, tmp_path, git_repo):
    repo, _ = git_repo
    # side-branch commit: exists, but never merged into HEAD
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "side"], check=True)
    (repo / "b.txt").write_text("b")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "b"], check=True)
    side_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-"], check=True)

    _write_note(note.parent / "n.md", board="uuid-1", landed=side_sha)
    findings = verify_note(note, board, repo)
    bad = [f for f in findings if f.kind == "git" and not f.ok]
    assert bad, "unlanded (non-ancestor) sha not detected"
    assert "ancestor" in bad[0].detail or "main" in bad[0].detail or "HEAD" in bad[0].detail


def test_verify_branch_labeled_sha_needs_existence_not_ancestry(note, board, git_repo):
    repo, _ = git_repo
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "side"], check=True)
    (repo / "b.txt").write_text("b")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "b"], check=True)
    side_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-"], check=True)

    # provenance group: branch sha must exist but is NOT required to be an ancestor
    _write_note(note.parent / "n.md", board="uuid-1", landed=f"{side_sha} (feat/my-branch)")
    findings = verify_note(note, board, repo)
    bad = [f for f in findings if not f.ok]
    assert not bad, f"branch-labeled sha must not require ancestry: {bad}"


def test_verify_without_landed_key_skips_git(note, board):
    _write_note(note.parent / "n.md", board="uuid-1")  # no landed:
    findings = verify_note(note, board, Path("/nonexistent-repo"))
    git_findings = [f for f in findings if f.kind == "git"]
    assert not git_findings, f"git checks must be skipped without landed: {git_findings}"
    assert all(f.ok for f in findings)


def test_verify_parses_landed_with_side_labels_and_plus_separators(note, board, git_repo):
    repo, sha = git_repo
    # the real card-092 shape: shas mixed with branch labels/parens/plus signs
    _write_note(note.parent / "n.md", board="uuid-1", landed=f"{sha} + 66d752587 (feat/x)")
    findings = verify_note(note, board, repo)
    shas = [f for f in findings if f.kind == "git"]
    assert len(shas) == 2  # the repo sha and 66d752587 are both >=7-hex tokens
    # the repo sha is an ancestor -> ok; 66d752587 only exists in sloughGPT -> flagged
    assert sum(f.ok for f in shas) == 1


def test_cli_exit_codes(tmp_path, git_repo, monkeypatch):
    from app_planner.cli import main

    repo, sha = git_repo
    notes = tmp_path / "notes"
    board_dir = tmp_path / "boarddir"
    notes.mkdir()
    board_dir.mkdir()
    _write_note(notes / "n.md", board="uuid-1", landed=sha)
    _write_board(board_dir / "board.jsonl", [_card("uuid-1", "Card X", "done")])
    monkeypatch.setenv("APP_PLANNER_NOTES_DIR", str(notes))
    monkeypatch.setenv("APP_PLANNER_BOARD_DIR", str(board_dir))

    rc_ok = main(["verify", str(notes / "n.md"), "--repo", str(repo)])
    assert rc_ok == 0

    _write_board(board_dir / "board.jsonl", [_card("uuid-1", "Card X", "todo")])
    rc_bad = main(["verify", str(notes / "n.md"), "--repo", str(repo)])
    assert rc_bad == 1


def test_cli_resolves_title_in_user_fallback_notes_dir(tmp_path, git_repo, monkeypatch):
    """Title resolution must also search NOTES_FALLBACK (~/.config/dev-notes),
    where the real journal lives, not only the project's .dev-notes."""
    from app_planner import config
    from app_planner.cli import main

    repo, sha = git_repo
    project_notes = tmp_path / "project-notes"
    fallback = tmp_path / "fallback-notes"
    project_notes.mkdir()
    fallback.mkdir()
    board_dir = tmp_path / "boarddir"
    board_dir.mkdir()
    _write_note(fallback / "2026-09-29-card094-x.md", board="uuid-1", landed=sha)
    _write_board(board_dir / "board.jsonl", [_card("uuid-1", "Card X", "done")])

    monkeypatch.delenv("APP_PLANNER_NOTES_DIR", raising=False)
    monkeypatch.setenv("APP_PLANNER_BOARD_DIR", str(board_dir))
    monkeypatch.setattr(config, "NOTES_FALLBACK", fallback)
    monkeypatch.setattr(config, "find_project_root", lambda: tmp_path)
    (tmp_path / ".dev-notes").mkdir()

    rc = main(["verify", "card094", "--repo", str(repo)])
    assert rc == 0


# ── Hardening: unmappable vs legacy statuses (card 3ccc38b7) ─────────────


def test_verify_unmappable_status_is_data_error(tmp_path):
    p = _write_note(tmp_path / "n.md", status="partial")
    b = _write_board(tmp_path / "board.jsonl", [_card("uuid-1", "Card X", "todo")])
    findings = verify_note(p, b, Path("."))
    bad = [f for f in findings if f.kind == "status" and not f.ok]
    assert bad, f"unmappable status not flagged: {findings}"
    assert "partial" in bad[0].detail
    assert "not a known status" in bad[0].detail


def test_verify_legacy_alias_passes_with_canonical_hint(tmp_path):
    p = _write_note(tmp_path / "n.md", status="wip")
    b = _write_board(tmp_path / "board.jsonl", [_card("uuid-1", "Card X", "in_progress")])
    findings = verify_note(p, b, Path("."))
    status = [f for f in findings if f.kind == "status"]
    assert status and status[0].ok, findings
    assert "doing" in status[0].detail


def test_verify_canonical_doing_consistent(tmp_path):
    p = _write_note(tmp_path / "n.md", status="doing")
    b = _write_board(tmp_path / "board.jsonl", [_card("uuid-1", "Card X", "in_progress")])
    findings = verify_note(p, b, Path("."))
    status = [f for f in findings if f.kind == "status"]
    assert status and status[0].ok, findings
