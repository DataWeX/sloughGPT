"""Gate tests for the single-writer lock in scripts/kanban_sync.sh.

The 13h stall this guards against was not the hung fetch — that is a fact of
networks. It was that every 5-minute tick *after* it exited 0 with "skipping":
flock reports no owner, contention was treated as success, and CI's freshness
alarm was the only thing that eventually noticed. So contention must name the
holder, and contention past a bound must be a FAILURE.

Sandboxed by env: KANBAN_SYNC_LOCK points at a tmp file and KANBAN_SYNC_WT at
a plain directory, so the script stops at its "not a git worktree — refusing"
guard. No fetch, no commit, no push, never the real timer's lock.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "kanban_sync.sh"
STALE_BOUND = 900  # keep in step with KANBAN_SYNC_LOCK_STALE's default


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env["KANBAN_SYNC_LOCK"] = str(tmp_path / "sync.lock")
    env["KANBAN_SYNC_WT"] = str(tmp_path / "not-a-worktree")
    (tmp_path / "not-a-worktree").mkdir(exist_ok=True)
    env.update(extra)
    return env


def _marker(tmp_path: Path) -> Path:
    return Path(f"{tmp_path / 'sync.lock'}.holder")


def _hold_lock(tmp_path: Path):
    """Take the lock in a background process and hand back the handle."""
    holder = subprocess.Popen(
        ["flock", str(tmp_path / "sync.lock"), "-c", "sleep 30"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        if holder.poll() is None and _locked(tmp_path):
            return holder
        time.sleep(0.02)
    holder.kill()
    pytest.fail("could not acquire the lock in the harness")


def _locked(tmp_path: Path) -> bool:
    probe = subprocess.run(
        ["flock", "-n", str(tmp_path / "sync.lock"), "-c", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode != 0


class TestHolderMarker:
    def test_a_run_publishes_then_removes_its_marker(self, tmp_path: Path):
        result = subprocess.run(
            ["bash", str(SCRIPT)], env=_env(tmp_path), capture_output=True, text=True
        )
        # Stops at the worktree guard, which is the point: no network, no push.
        assert result.returncode == 1
        assert "not a git worktree" in result.stderr
        assert not _marker(tmp_path).exists(), "marker must not outlive the run"

    def test_contention_reports_the_holder_and_skips(self, tmp_path: Path):
        holder = _hold_lock(tmp_path)
        try:
            _marker(tmp_path).write_text(f"{holder.pid} {int(time.time())} /some/repo\n")
            result = subprocess.run(
                ["bash", str(SCRIPT)], env=_env(tmp_path), capture_output=True, text=True
            )
            assert result.returncode == 0
            assert f"lock held by pid {holder.pid}" in result.stdout
            assert "skipping" in result.stdout
        finally:
            holder.kill()
            holder.wait()

    def test_contention_past_the_bound_is_a_failure(self, tmp_path: Path):
        holder = _hold_lock(tmp_path)
        try:
            wedged = int(time.time()) - (STALE_BOUND + 60)
            _marker(tmp_path).write_text(f"{holder.pid} {wedged} /some/repo\n")
            result = subprocess.run(
                ["bash", str(SCRIPT)], env=_env(tmp_path), capture_output=True, text=True
            )
            assert result.returncode == 1
            assert "producer is wedged" in result.stderr
            assert str(STALE_BOUND + 60) in result.stderr
        finally:
            holder.kill()
            holder.wait()

    def test_contention_without_a_marker_is_still_loud(self, tmp_path: Path):
        holder = _hold_lock(tmp_path)
        try:
            result = subprocess.run(
                ["bash", str(SCRIPT)], env=_env(tmp_path), capture_output=True, text=True
            )
            assert result.returncode == 0
            assert "no holder marker" in result.stdout
        finally:
            holder.kill()
            holder.wait()

    def test_stale_bound_is_configurable(self, tmp_path: Path):
        holder = _hold_lock(tmp_path)
        try:
            _marker(tmp_path).write_text(f"{holder.pid} {int(time.time()) - 30} /some/repo\n")
            env = _env(tmp_path, KANBAN_SYNC_LOCK_STALE="10")
            result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
            assert result.returncode == 1
            assert "wedged" in result.stderr
        finally:
            holder.kill()
            holder.wait()
