"""Tests for scripts/check_surface_scope.py — the Surface Rule advisory gate.

Policy: AGENTS.md "Surface Rule (one change unit spans one shipping surface)".
The gate is ADVISORY: it prints a diff's surface span and returns 0, with
--strict as the ratchet that turns a multi-surface span into exit 1.

The contract under test is the one the script's own docstring states —
"Advisory by default: always exit 0" — so the failure paths matter as much as
the happy path. A gate that blocks CI on missing history or a typo'd range is
worse than no gate: it trains people to bypass it. Both of those paths are
pinned here (see ``test_missing_history_is_advisory`` and
``test_malformed_range_is_advisory``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_surface_scope as gate  # noqa: E402

# ── surface_of: the mapping the whole rule rests on ────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("apps/web/lib/navigation.ts", "web"),
        ("apps/web/app/(app)/chat/page.tsx", "web"),
        ("apps/mobile/src/App.tsx", "mobile"),
        ("apps/api/server/main.py", "api"),
        ("domain/training/_internal/train_pipeline.py", "domain"),
        ("packages/core-py/slo/__init__.py", "packages"),
        ("packages/strui/src/Button.tsx", "packages"),
        # Meta files are travel companions, never a second surface (clause 2).
        ("docs/AGENT_SYNC.md", None),
        ("docs/design/DESIGN_SYSTEM.md", None),
        (".github/workflows/ci_cd.yml", None),
        ("scripts/check_surface_scope.py", None),
        ("tests/test_check_surface_scope.py", None),
        ("AGENTS.md", None),
        ("package.json", None),
        ("README.md", None),
    ],
)
def test_surface_of(path: str, expected: str | None) -> None:
    assert gate.surface_of(path) == expected


def test_surface_of_is_boundary_anchored() -> None:
    """`apps/web` must not swallow a path that merely starts with the string."""
    assert gate.surface_of("apps/webapp/index.ts") is None
    assert gate.surface_of("domaine/foo.py") is None
    assert gate.surface_of("packagesx/foo.py") is None
    # A bare directory entry is itself a surface hit (the prefix + "/" branch
    # does not cover it, so equality must).
    assert gate.surface_of("domain") == "domain"
    assert gate.surface_of("apps/web") == "web"


def test_every_declared_surface_is_reachable() -> None:
    """Clause 1 names five surfaces; the table must implement exactly those."""
    assert len(gate.SURFACES) == 5
    names = [name for _prefix, name in gate.SURFACES]
    assert sorted(names) == ["api", "domain", "mobile", "packages", "web"]


# ── main(): exit-code matrix ───────────────────────────────────────────────


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["check_surface_scope.py", *args])
    return gate.main()


def _fake_git(
    monkeypatch: pytest.MonkeyPatch,
    *,
    files: str | None = "apps/web/lib/a.ts\n",
    merge_base: str | None = "abc123",
) -> None:
    """Stand in for `git diff` / `git merge-base` so these tests stay hermetic."""

    def fake(cmd: list[str], **_kwargs: object) -> str:
        if cmd[:2] == ["git", "merge-base"]:
            if merge_base is None:
                raise subprocess.CalledProcessError(128, cmd, stderr=b"no merge base")
            return f"{merge_base}\n"
        if cmd[:2] == ["git", "diff"]:
            if files is None:
                raise subprocess.CalledProcessError(128, cmd, stderr=b"bad revision")
            return files
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr(gate.subprocess, "check_output", fake)


def test_single_surface_is_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_git(monkeypatch, files="apps/web/lib/a.ts\napps/web/lib/b.ts\ndocs/x.md\n")
    assert _run(monkeypatch, "--range", "A..B") == 0


def test_doc_only_change_reports_no_shipping_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_git(monkeypatch, files="docs/AGENT_SYNC.md\nAGENTS.md\n")
    assert _run(monkeypatch, "--range", "A..B") == 0


def test_multi_surface_advises_but_does_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_git(monkeypatch, files="apps/web/lib/a.ts\ndomain/foo.py\n")
    assert _run(monkeypatch, "--range", "A..B") == 0


def test_multi_surface_strict_is_the_ratchet(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_git(monkeypatch, files="apps/web/lib/a.ts\ndomain/foo.py\npackages/p/x.py\n")
    assert _run(monkeypatch, "--range", "A..B", "--strict") == 1


def test_single_surface_strict_stays_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_git(monkeypatch, files="domain/foo.py\n")
    assert _run(monkeypatch, "--range", "A..B", "--strict") == 0


def test_warn_names_every_surface_in_the_span(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_git(monkeypatch, files="apps/web/a.ts\ndomain/b.py\n")
    _run(monkeypatch, "--range", "A..B")
    out = capsys.readouterr().out
    assert "WARN" in out
    assert "domain" in out and "web" in out
    assert "PR 258" in out  # the evidence the rule cites


# ── the advisory contract: never block ─────────────────────────────────────


def test_missing_history_is_advisory(monkeypatch: pytest.MonkeyPatch) -> None:
    """No origin/main to merge-base against -> exit 0, not a crash."""
    _fake_git(monkeypatch, merge_base=None)
    assert _run(monkeypatch) == 0


def test_malformed_range_is_advisory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd range must not traceback into exit 1 (docstring: always exit 0)."""
    _fake_git(monkeypatch, files=None)
    assert _run(monkeypatch, "--range", "nope..HEAD") == 0
    assert _run(monkeypatch, "--range", "nope..HEAD", "--strict") == 0


# ── real-git end-to-end ────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "gate@test")
    _git(tmp_path, "config", "user.name", "gate")
    (tmp_path / "README.md").write_text("base\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "base")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_real_range_single_surface(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / "apps").mkdir()
    (repo / "apps" / "web").mkdir()
    (repo / "apps" / "web" / "a.ts").write_text("x\n")
    _git(repo, "add", "apps/web/a.ts")
    _git(repo, "commit", "-q", "-m", "web")
    assert _run(monkeypatch, "--range", "HEAD~1..HEAD") == 0


def test_real_range_multi_surface_strict(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / "domain").mkdir()
    (repo / "domain" / "__init__.py").write_text("x = 1\n")
    (repo / "apps").mkdir()
    (repo / "apps" / "web").mkdir()
    (repo / "apps" / "web" / "a.ts").write_text("x\n")
    _git(repo, "add", "domain", "apps")
    _git(repo, "commit", "-q", "-m", "two surfaces")
    assert _run(monkeypatch, "--range", "HEAD~1..HEAD", "--strict") == 1


def test_real_bad_range_never_blocks(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(monkeypatch, "--range", "nope..HEAD") == 0
