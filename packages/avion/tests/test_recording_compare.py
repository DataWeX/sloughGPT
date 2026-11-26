"""Recording compare tests — step diffs, screenshot diffs, run history, CLI.

Synthetic runs only (tmp dirs + Pillow-generated PNGs): no browser, no
network, no live stack.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from avion.core.task import Task, TaskStatus, TaskStep
from avion.recording.compare import (
    KEEP_RUNS,
    compare_runs,
    diff_shot,
    diff_steps,
    list_runs,
    new_run_id,
    prune_runs,
    render_summary,
    runs_root,
    shot_name,
    slugify,
)

PIL = pytest.importorskip("PIL", reason="Pillow needed for pixel diffs")
from PIL import Image  # noqa: E402


def _step(name: str, status: str = "passed") -> dict:
    return {"name": name, "status": status, "duration_ms": 1.0}


def _png(path: Path, color: tuple[int, int, int], size=(40, 30), mark=None) -> None:
    img = Image.new("RGB", size, color)
    if mark is not None:
        for x in range(10):
            for y in range(10):
                img.putpixel((x, y), mark)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _make_run(
    root: Path,
    run_id: str,
    flows: dict[str, dict],
    shots: dict[str, dict[str, tuple]] | None = None,
) -> Path:
    """Build a run dir: run.json + per-flow result.json (+ optional shots).

    ``flows`` maps flow-id -> result dict; ``shots`` maps flow-id ->
    {filename: (color, mark)}.
    """
    run = root / run_id
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"id": run_id}))
    for flow_id, result in flows.items():
        flow_dir = run / flow_id
        flow_dir.mkdir()
        (flow_dir / "result.json").write_text(json.dumps(result))
        for filename, spec in (shots or {}).get(flow_id, {}).items():
            _png(flow_dir / "shots" / filename, spec[0], mark=spec[1])
    return run


class TestDiffSteps:
    def test_identical_is_empty(self):
        steps = [_step("a"), _step("b")]
        assert diff_steps(steps, steps) == []

    def test_added_and_removed(self):
        old = [_step("a"), _step("b")]
        new = [_step("a"), _step("c")]
        kinds = {(c["kind"], c["step"]) for c in diff_steps(old, new)}
        assert kinds == {("removed", "b"), ("added", "c")}

    def test_moved_not_add_remove(self):
        old = [_step("a"), _step("b"), _step("c")]
        new = [_step("c"), _step("a"), _step("b")]
        changes = diff_steps(old, new)
        # one minimal move explains the reorder: c jumped ahead of a,b
        assert all(c["kind"] == "moved" for c in changes)
        assert {c["step"] for c in changes} == {"c"}

    def test_status_change_reported(self):
        old = [_step("a"), _step("b")]
        new = [_step("a"), _step("b", "failed")]
        changes = diff_steps(old, new)
        assert len(changes) == 1
        assert changes[0]["kind"] == "status"
        assert changes[0]["step"] == "b"
        assert "passed -> failed" in changes[0]["detail"]


class TestShotName:
    def test_index_prefix_and_key(self):
        filename, key = shot_name(3, "Send message", {})
        assert filename == "03-send-message.png"
        assert key == "send-message"

    def test_repeated_names_get_suffix(self):
        seen: dict[str, int] = {}
        _, key1 = shot_name(1, "wait", seen)
        _, key2 = shot_name(2, "wait", seen)
        assert key1 == "wait"
        assert key2 == "wait-2"

    def test_slugify(self):
        assert slugify("Go to /chat!") == "go-to-chat"
        assert slugify("///") == "step"


class TestDiffShot:
    def test_added_and_removed(self, tmp_path: Path):
        lone = tmp_path / "solo.png"
        _png(lone, (255, 255, 255))
        assert diff_shot("solo", None, lone)["status"] == "added"
        assert diff_shot("solo", lone, None)["status"] == "removed"

    def test_identical_unchanged(self, tmp_path: Path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        _png(a, (10, 20, 30))
        _png(b, (10, 20, 30))
        out = diff_shot("x", a, b)
        assert out["status"] == "unchanged"
        assert out["fraction"] == 0.0

    def test_changed_pixels_flagged(self, tmp_path: Path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        _png(a, (10, 20, 30))
        _png(b, (10, 20, 30), mark=(250, 0, 0))
        out = diff_shot("x", a, b)
        assert out["status"] == "changed"
        assert out["fraction"] > 0.05  # 10x10 of 40x30 = 8.3%

    def test_noise_below_tolerance_unchanged(self, tmp_path: Path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        _png(a, (100, 100, 100))
        _png(b, (104, 100, 96))  # per-channel delta <= default tol 8
        assert diff_shot("x", a, b)["status"] == "unchanged"

    def test_size_change_flagged(self, tmp_path: Path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        _png(a, (0, 0, 0), size=(40, 30))
        _png(b, (0, 0, 0), size=(41, 30))
        out = diff_shot("x", a, b)
        assert out["status"] == "size-changed"
        assert "40x30" in out["note"]

    def test_unreadable_degrades(self, tmp_path: Path):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not a png")
        good = tmp_path / "good.png"
        _png(good, (1, 2, 3))
        assert diff_shot("x", good, bad)["status"] == "unavailable"


class TestRunHistory:
    def test_list_runs_requires_meta(self, tmp_path: Path):
        _make_run(tmp_path, "20261126-000000-aaaaaa", {})
        (tmp_path / "not-a-run").mkdir()
        (tmp_path / "stray.txt").write_text("x")
        runs = list_runs(tmp_path)
        assert [p.name for p in runs] == ["20261126-000000-aaaaaa"]

    def test_list_runs_sorted_oldest_first(self, tmp_path: Path):
        for i in (3, 1, 2):
            _make_run(tmp_path, f"20261126-00000{i}-aaaaaa", {})
        assert [p.name for p in list_runs(tmp_path)] == [
            "20261126-000001-aaaaaa",
            "20261126-000002-aaaaaa",
            "20261126-000003-aaaaaa",
        ]

    def test_prune_keeps_newest_and_touches_only_runs(self, tmp_path: Path):
        for i in range(KEEP_RUNS + 3):
            _make_run(tmp_path, f"20261126-{i:06d}-aaaaaa", {})
        bystander = tmp_path / "user-notes"
        bystander.mkdir()
        (bystander / "keep.txt").write_text("precious")
        removed = prune_runs(tmp_path)
        assert len(removed) == 3
        assert len(list_runs(tmp_path)) == KEEP_RUNS
        assert (bystander / "keep.txt").exists()  # non-run dirs never touched

    def test_prune_noop_under_bound(self, tmp_path: Path):
        _make_run(tmp_path, "20261126-000000-aaaaaa", {})
        assert prune_runs(tmp_path) == []

    def test_new_run_id_sortable(self):
        ids = sorted(new_run_id() for _ in range(5))
        assert all(len(i) >= 15 for i in ids)
        assert ids == sorted(ids)

    def test_runs_root_env_override(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("SLO_JOURNEY_RUNS", str(tmp_path))
        assert runs_root() == tmp_path


class TestCompareRuns:
    def _pair(self, root: Path) -> tuple[Path, Path]:
        old = _make_run(
            root,
            "20261126-010000-aaaaaa",
            {
                "chat": {"task": "chat", "status": "passed", "steps": [_step("a"), _step("b")]},
                "souls": {"task": "souls", "status": "passed", "steps": [_step("x")]},
            },
            shots={
                "chat": {
                    "01-a.png": ((10, 20, 30), None),
                    "02-b.png": ((10, 20, 30), None),
                }
            },
        )
        new = _make_run(
            root,
            "20261126-020000-bbbbbb",
            {
                "chat": {
                    "task": "chat",
                    "status": "failed",
                    "steps": [_step("a"), _step("b", "failed"), _step("c")],
                },
                "settings": {"task": "settings", "status": "passed", "steps": [_step("y")]},
            },
            shots={
                "chat": {
                    "01-a.png": ((10, 20, 30), None),
                    "02-b.png": ((10, 20, 30), (250, 0, 0)),
                    "03-c.png": ((1, 1, 1), None),
                }
            },
        )
        return old, new

    def test_status_step_shot_and_flow_membership(self, tmp_path: Path):
        old, new = self._pair(tmp_path)
        cmp = compare_runs(old, new)
        assert cmp.changed
        assert cmp.flows_added == ["settings"]
        assert cmp.flows_removed == ["souls"]

        chat = next(f for f in cmp.flows if f.flow == "chat")
        assert chat.old_status == "passed" and chat.new_status == "failed"
        kinds = {(c["kind"], c["step"]) for c in chat.steps}
        assert ("status", "b") in kinds
        assert ("added", "c") in kinds
        shot_status = {s["shot"]: s["status"] for s in chat.shots}
        # diff-keys drop the NN- index prefix by design (inserting a step
        # must not renumber-match every shot)
        assert shot_status["b"] == "changed"
        assert shot_status["c"] == "added"
        assert shot_status["a"] == "unchanged"

    def test_identical_runs_not_changed(self, tmp_path: Path):
        old = _make_run(
            tmp_path,
            "20261126-010000-aaaaaa",
            {"chat": {"task": "chat", "status": "passed", "steps": [_step("a")]}},
        )
        new = _make_run(
            tmp_path,
            "20261126-020000-bbbbbb",
            {"chat": {"task": "chat", "status": "passed", "steps": [_step("a")]}},
        )
        cmp = compare_runs(old, new)
        assert not cmp.changed
        assert "no differences" in render_summary(cmp)

    def test_summary_is_concise(self, tmp_path: Path):
        old, new = self._pair(tmp_path)
        text = render_summary(compare_runs(old, new))
        lines = text.splitlines()
        assert lines[0].startswith("compare 20261126-010000")
        assert any("[!] chat:" in line for line in lines)
        # unchanged flows get no detail line — summarize, don't dump
        assert not any("[!] souls:" in line for line in lines)
        assert len(lines) <= 6

    def test_missing_run_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            compare_runs(tmp_path / "nope", tmp_path)

    def test_to_dict_roundtrips(self, tmp_path: Path):
        old, new = self._pair(tmp_path)
        data = compare_runs(old, new).to_dict()
        assert data["old"] == "20261126-010000-aaaaaa"
        assert data["changed"] is True
        assert json.loads(json.dumps(data)) == data


class TestCLICompare:
    def _runs(self, tmp_path: Path, changed: bool) -> tuple[str, str]:
        old = _make_run(
            tmp_path,
            "20261126-010000-aaaaaa",
            {"chat": {"task": "chat", "status": "passed", "steps": [_step("a")]}},
        )
        steps = [_step("a"), _step("b")] if changed else [_step("a")]
        new = _make_run(
            tmp_path,
            "20261126-020000-bbbbbb",
            {"chat": {"task": "chat", "status": "passed", "steps": steps}},
        )
        return str(old), str(new)

    def test_exit_zero_when_identical(self, tmp_path: Path, capsys):
        from avion.cli import main

        old, new = self._runs(tmp_path, changed=False)
        assert main(["compare", old, new]) == 0
        assert "no differences" in capsys.readouterr().out

    def test_exit_one_when_changed(self, tmp_path: Path, capsys):
        from avion.cli import main

        old, new = self._runs(tmp_path, changed=True)
        assert main(["compare", old, new]) == 1
        assert "step added b" in capsys.readouterr().out

    def test_exit_two_on_error(self, tmp_path: Path):
        from avion.cli import main

        assert main(["compare", str(tmp_path / "missing"), str(tmp_path)]) == 2

    def test_json_output(self, tmp_path: Path, capsys):
        from avion.cli import main

        old, new = self._runs(tmp_path, changed=True)
        assert main(["compare", old, new, "--json"]) == 1
        data = json.loads(capsys.readouterr().out)
        assert data["changed"] is True


class TestTaskCheckpoints:
    def test_step_records_populated(self):
        async def ok(ctx):
            pass

        async def boom(ctx):
            raise RuntimeError("kaput")

        task = Task(
            "t",
            steps=[
                TaskStep("a", ok),
                TaskStep("b", boom, optional=True),
                TaskStep("c", ok, precondition=lambda ctx: False),
            ],
        )
        result = asyncio.run(task.execute({}))
        assert [(s["name"], s["status"]) for s in result.step_records] == [
            ("a", "passed"),
            ("b", "failed"),
            ("c", "skipped"),
        ]
        assert all("duration_ms" in s for s in result.step_records)

    def test_to_dict_carries_steps(self):
        result = asyncio.run(Task("t", steps=[]).execute({}))
        assert result.to_dict()["steps"] == []

    def test_on_step_hook_receives_statuses(self):
        seen: list[tuple[str, str]] = []

        async def ok(ctx):
            pass

        task = Task(
            "t",
            steps=[TaskStep("a", ok), TaskStep("b", ok)],
            on_step=lambda name, status: seen.append((name, status)),
        )
        asyncio.run(task.execute({}))
        assert seen == [("a", "passed"), ("b", "passed")]

    def test_on_step_failure_never_fails_journey(self):
        def bad_hook(name, status):
            raise RuntimeError("screenshot exploded")

        async def ok(ctx):
            pass

        task = Task("t", steps=[TaskStep("a", ok)], on_step=bad_hook)
        result = asyncio.run(task.execute({}))
        assert result.status == TaskStatus.PASSED
        assert result.step_records[0]["status"] == "passed"

    def test_async_on_step_awaited(self):
        seen: list[str] = []

        async def ok(ctx):
            pass

        async def hook(name, status):
            seen.append(f"{name}:{status}")

        task = Task("t", steps=[TaskStep("a", ok)], on_step=hook)
        asyncio.run(task.execute({}))
        assert seen == ["a:passed"]
