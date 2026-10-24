"""Tests for the unified logs dashboard — card 8182563a (logs unification).

The 2026-08-27/28 rules this file locks down:

1. **No separate monitor command** — ``commands/monitor.py`` is deleted; one
   dashboard, one code path (``run_dashboard`` in ``commands/logs.py``) shared
   by ``logs --dashboard``, a hidden ``monitor`` migration alias, and
   ``train monitor``.
2. **Beautiful colored output** — metrics are color-graded (load thresholds,
   error counts, generation stats), not printed raw.
3. **Punchy info for ALL processes** — every process line says something
   (status word fallback when there is no detail), nothing renders blank.
4. **De-cluttered dashboard** — empty sections are omitted entirely (no
   ``(none active)`` / ``(no events yet)`` placeholders), and the health
   score folds into the SERVER line instead of its own section.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# Add CLI src to path (mirrors other CLI tests)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commands import logs as logs_mod
from commands.logs import (
    _load_color,
    _render_dashboard,
    _status_color,
    monitor,
)

pytestmark = pytest.mark.timeout(60)


# ── helpers ────────────────────────────────────────────────────────────


def _snap(**over):
    """Minimal dashboard snapshot; override health/processes/events."""
    health = {
        "model_type": "qwen",
        "model_loaded": True,
        "uptime_seconds": 7200,
        "cpu_percent": 45,
        "memory_percent": 55,
        "memory_used_mb": 1024,
        "request_count": 10,
        "error_count": 0,
        "tokens_per_sec": 12.5,
        "avg_latency_ms": 30,
    }
    health.update(over.pop("health", {}))
    data = {"health": health, "processes": {}, "events": [], "recent_errors": []}
    data.update(over)
    return {"data": data}


@pytest.fixture()
def colored(monkeypatch):
    """Emit ANSI codes in tests — _c() is a no-op when stdout is not a TTY."""
    monkeypatch.setattr(logs_mod, "_TTY", True)
    return logs_mod


# ── fold invariants: no separate monitor command ───────────────────────


def test_monitor_module_is_gone():
    """commands/monitor.py was deleted — the fold is guarded against resurrection."""
    assert importlib.util.find_spec("commands.monitor") is None


def test_root_cli_registers_monitor_hidden():
    """The migration alias dispatches but never appears in help listings."""
    from cli import cli as root_cli  # noqa: PLC0415 — heavy import, one test only

    alias = root_cli.commands["monitor"]
    assert alias.hidden is True
    assert alias.help and "logs --dashboard" in alias.help


def test_monitor_alias_forwards_to_logs_dashboard(monkeypatch, capsys):
    """`sloughgpt monitor ...` → same dashboard code path, with a merge notice."""
    captured = {}

    def fake_sse(host, port, interval, output_json, clear, compact):
        captured.update(
            host=host, port=port, interval=interval, output_json=output_json,
            clear=clear, compact=compact,
        )

    monkeypatch.setattr(logs_mod, "_consume_sse_dashboard", fake_sse)
    result = CliRunner().invoke(monitor, ["--interval", "1", "--port", "9000"])
    assert result.exit_code == 0, result.output
    assert captured == {
        "host": "localhost", "port": 9000, "interval": 1.0,
        "output_json": False, "clear": True, "compact": False,
    }
    notice = result.stderr + result.output
    assert "logs --dashboard" in notice, "alias must announce the merge"


def test_logs_dashboard_flag_routes_to_run_dashboard(monkeypatch):
    """`logs --dashboard` calls the shared entry, not an inline copy."""
    called = {}
    monkeypatch.setattr(
        logs_mod, "run_dashboard",
        lambda *a, **k: called.update(args=a, kwargs=k),
    )
    result = CliRunner().invoke(
        logs_mod.logs,
        ["--dashboard", "--interval", "3", "--host", "h1", "--port", "9", "--compact"],
    )
    assert result.exit_code == 0, result.output
    assert called["args"] == ("h1", 9, 3.0)
    assert called["kwargs"] == {
        "output_json": False, "no_clear": False, "compact": True,
    }


def test_train_monitor_delegates_to_run_dashboard(monkeypatch):
    """`train monitor` uses the same dashboard path (no commands.monitor import)."""
    from core.framework import Context, Group  # noqa: PLC0415
    from groups.train import register as register_train  # noqa: PLC0415

    called = {}
    monkeypatch.setattr(
        logs_mod, "run_dashboard",
        lambda *a, **k: called.update(args=a, kwargs=k),
    )

    root = Group("sloughgpt")
    register_train(root)
    cmd = root.groups["train"].commands["monitor"]

    ctx = Context()
    ctx.obj = {"host": "h1", "port": 1234}
    cmd.func(ctx=ctx, watch=False, interval=2)

    assert called["args"] == ("h1", 1234, 2.0)
    assert called["kwargs"] == {"output_json": False, "no_clear": False}


def test_system_category_lists_no_monitor():
    """Help catalog no longer advertises a separate monitor command."""
    from core.slo_cli import _CATEGORIES  # noqa: PLC0415

    assert "monitor" not in _CATEGORIES["System"]["cmds"]


# ── colors finished (rule 2) ───────────────────────────────────────────


def test_load_color_thresholds():
    assert _load_color(69) == logs_mod._GREEN
    assert _load_color(70) == logs_mod._YELLOW
    assert _load_color(90) == logs_mod._RED


def test_sys_metrics_color_graded(colored, capsys):
    _render_dashboard(
        _snap(health={"cpu_percent": 95, "memory_percent": 40}), clear=False
    )
    out = capsys.readouterr().out
    assert colored._c("95%", colored._RED) in out
    assert colored._c("40%", colored._GREEN) in out
    assert colored._c("err 0", colored._GREY) in out


def test_error_count_red_when_nonzero(colored, capsys):
    _render_dashboard(_snap(health={"error_count": 3}), clear=False)
    out = capsys.readouterr().out
    assert colored._c("err 3", colored._RED) in out


def test_gen_line_coloring(colored, capsys):
    _render_dashboard(_snap(), clear=False)
    out = capsys.readouterr().out
    assert colored._c("12.5", colored._CYAN) in out
    assert colored._c("30ms", colored._GREY) in out


# ── punchy info for ALL processes (rule 3) ─────────────────────────────


def test_process_without_detail_shows_status_word(colored, capsys):
    snap = _snap(processes={
        "t1": {"status": "queued", "label": "job", "detail": "", "progress": 0},
    })
    _render_dashboard(snap, clear=False)
    out = capsys.readouterr().out
    assert colored._c("queued", colored._YELLOW) in out
    # the status token fills the line — no trailing-whitespace-only tail
    job_line = next(ln for ln in out.splitlines() if "job" in ln)
    assert not job_line.endswith(" ")


def test_process_exited_shows_status_and_detail(colored, capsys):
    snap = _snap(processes={
        "s1": {"status": "exited", "label": "self-train", "detail": "exit code 1",
               "progress": 0},
    })
    _render_dashboard(snap, clear=False)
    out = capsys.readouterr().out
    assert colored._c("exited", colored._GREY) in out
    assert "exit code 1" in out


def test_running_process_keeps_detail_and_bar(colored, capsys):
    snap = _snap(processes={
        "d1": {"status": "running", "label": "dl:model", "detail": "45% 1.2MB/s",
               "progress": 45},
    })
    _render_dashboard(snap, clear=False)
    out = capsys.readouterr().out
    assert "45% 1.2MB/s" in out
    assert "█" in out  # progress bar retained
    assert colored._c("running", colored._GREEN) not in out  # not redundant w/ icon


def test_status_colors_match_icons():
    assert _status_color("running") == logs_mod._GREEN
    assert _status_color("queued") == logs_mod._YELLOW
    assert _status_color("starting") == logs_mod._CYAN
    assert _status_color("error") == logs_mod._RED
    assert _status_color("exited") == logs_mod._GREY


# ── de-cluttered dashboard (rule 4) ────────────────────────────────────


def test_empty_sections_omitted(colored, capsys):
    _render_dashboard(_snap(), clear=False)
    out = capsys.readouterr().out
    assert "PROCESSES" not in out
    assert "EVENTS" not in out
    assert "(none active)" not in out
    assert "(no events yet)" not in out


def test_health_score_folded_into_server_line(colored, capsys):
    snap = _snap(health={"health_score": {"score": 92, "status": "good"}})
    _render_dashboard(snap, clear=False)
    out = capsys.readouterr().out
    assert "92/100" in out
    assert "HEALTH" not in out  # no standalone section
    server_line = next(ln for ln in out.splitlines() if "SERVER" in ln)
    assert "92/100" in server_line


def test_compact_hides_processes_and_events(colored, capsys):
    snap = _snap(
        processes={"t1": {"status": "running", "label": "x", "detail": "d",
                          "progress": 0}},
        events=[{"ts": 1, "category": "SYSTEM", "msg": "hello"}],
    )
    _render_dashboard(snap, clear=False, compact=True)
    out = capsys.readouterr().out
    assert "PROCESSES" not in out
    assert "EVENTS" not in out
    assert "SERVER" in out  # compact still renders the basics
