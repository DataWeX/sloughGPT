"""Gate tests for scripts/check_kanban_freshness.py — the CI alarm that
keeps the kanban sync producer honest (card 20261031_084842).

The producer runs on this machine (kanban-sync.timer); CI only ever sees
the repo mirror. Without this gate a dead timer means GitHub board history
goes stale silently. Same load pattern as test_board_chain_gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_kanban_freshness as gate  # noqa: E402


def _manifest(tmp_path: Path, exported_at: datetime, **extra) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "exported_at": exported_at.isoformat(timespec="seconds"),
                "note_count": 2,
                "card_count": 5,
                "board_sha256": "00",
                "source": "/x",
                **extra,
            }
        ),
        encoding="utf-8",
    )
    return path


class TestFreshness:
    def test_fresh_manifest_passes(self, tmp_path: Path, capsys):
        manifest = _manifest(tmp_path, datetime.now(UTC))
        assert gate.main(["--manifest", str(manifest)]) == 0
        assert "ok:" in capsys.readouterr().out

    def test_stale_manifest_fails(self, tmp_path: Path, capsys):
        old = datetime.now(UTC) - timedelta(minutes=121)
        manifest = _manifest(tmp_path, old)
        assert gate.main(["--manifest", str(manifest)]) == 1
        out = capsys.readouterr().out
        assert "FAIL" in out and "stale" in out

    def test_age_budget_is_configurable(self, tmp_path: Path):
        old = datetime.now(UTC) - timedelta(minutes=30)
        manifest = _manifest(tmp_path, old)
        # stale under a 20-minute budget, fresh under a 60-minute one
        assert gate.main(["--manifest", str(manifest), "--max-age-minutes", "20"]) == 1
        assert gate.main(["--manifest", str(manifest), "--max-age-minutes", "60"]) == 0

    def test_naive_timestamp_treated_as_utc(self, tmp_path: Path):
        """exported_at is written with +00:00; a naive value must not
        crash the gate (pre-epoch manifests, hand edits)."""
        naive = datetime.now(UTC).replace(tzinfo=None)
        manifest = _manifest(tmp_path, naive)
        assert gate.main(["--manifest", str(manifest)]) == 0


class TestCorruptOrMissing:
    def test_missing_manifest_fails_with_pointing_message(self, tmp_path, capsys):
        missing = tmp_path / "manifest.json"
        assert gate.main(["--manifest", str(missing)]) == 1
        out = capsys.readouterr().out
        assert "never ran" in out and "kanban_sync.sh" in out

    def test_unreadable_manifest_fails(self, tmp_path: Path, capsys):
        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        assert gate.main(["--manifest", str(path)]) == 1
        assert "unreadable" in capsys.readouterr().out

    def test_manifest_missing_field_fails(self, tmp_path: Path, capsys):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"note_count": 1}), encoding="utf-8")
        assert gate.main(["--manifest", str(path)]) == 1
        assert "unreadable" in capsys.readouterr().out


def test_default_target_is_repo_manifest():
    """No-arg default must be the repo-side manifest, cwd-independent."""
    assert gate.DEFAULT_MANIFEST == REPO_ROOT / ".kanban" / "manifest.json"
    assert gate.DEFAULT_MAX_AGE_MINUTES == 120


class TestReadsOriginMain:
    """The default verdict is about main, not the checked-out tree.

    The producer pushes to main only, so a feature branch's copy of
    .kanban/manifest.json is stale by design — reading it reddened every
    branch CI run with "kanban-sync.timer looks dead" for a healthy timer.
    """

    @staticmethod
    def _payload(exported_at: datetime) -> str:
        return json.dumps(
            {
                "exported_at": exported_at.isoformat(timespec="seconds"),
                "note_count": 2,
                "card_count": 5,
            }
        )

    def _fake_git(self, monkeypatch, results):
        """Install a git stub; `results` is a list applied per call in order.

        Each entry is (returncode, stdout, stderr); the last entry repeats for
        any further call.
        """
        calls: list[list[str]] = []

        def run_git(args, cwd=None):
            calls.append(list(args))
            rc, out, err = results[min(len(calls) - 1, len(results) - 1)]
            return subprocess.CompletedProcess(["git", *args], rc, out, err)

        monkeypatch.setattr(gate, "run_git", run_git)
        return calls

    def test_default_reads_the_ref_not_the_worktree(self, monkeypatch, capsys):
        calls = self._fake_git(monkeypatch, [(0, self._payload(datetime.now(UTC)), "")])
        assert gate.main([]) == 0
        assert calls[0] == ["show", f"{gate.DEFAULT_REF}:{gate.MANIFEST_IN_TREE}"]
        out = capsys.readouterr().out
        assert "ok:" in out and "origin/main:.kanban/manifest.json" in out

    def test_stale_on_main_fails_however_fresh_the_branch_copy_is(
        self, monkeypatch, tmp_path, capsys
    ):
        old = datetime.now(UTC) - timedelta(minutes=300)
        self._fake_git(monkeypatch, [(0, self._payload(old), "")])
        # An explicit fresh file must NOT be consulted on the default path.
        fresh = _manifest(tmp_path, datetime.now(UTC))
        assert gate.main(["--manifest", str(fresh)]) == 0
        assert gate.main([]) == 1
        assert "FAIL" in capsys.readouterr().out

    def test_unreadable_ref_fails_open_loudly(self, monkeypatch, capsys):
        """No origin, no ref: 'cannot verify' is not 'the producer is dead'."""
        calls = self._fake_git(
            monkeypatch,
            [
                (1, "", "fatal: ambiguous argument"),
                (1, "", "fatal: could not read"),
            ],
        )
        assert gate.main([]) == 0
        out = capsys.readouterr().out
        assert out.startswith("WARN:") and "cannot verify" in out
        assert calls[0][0] == "show" and calls[1][:3] == ["fetch", "--no-tags", "origin"]

    def test_missing_ref_is_fetched_once_then_read(self, monkeypatch, capsys):
        calls = self._fake_git(
            monkeypatch,
            [
                (1, "", "fatal: unknown revision"),
                (0, self._payload(datetime.now(UTC)), ""),
            ],
        )
        assert gate.main([]) == 0
        assert calls[0][0] == "show" and calls[1][0] == "fetch" and calls[2][0] == "show"
        assert "ok:" in capsys.readouterr().out
