"""Logs unification (card 8182563a) — monitor folded into logs, colors finished.

User rules 2026-08-27/28:
- no separate ``monitor`` command — it lives in ``logs --dashboard``
- beautiful colored output (TTY colored, piped output stays clean)
- punchy info for ALL processes (type-colored labels, rpm restored, health
  merged into the headline, dashboard de-cluttered)

Guards:
- ``commands/monitor.py`` retired; no registration left in cli.py / slo_cli.py
- ``logs`` keeps every flag ``monitor`` offered (interval/host/port/json/no-clear)
- dashboard render: rpm shown, health in SERVER line, sections de-cluttered
- chrome messages honor TTY (ANSI only when interactive)
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from commands import logs as logs_mod
from commands.logs import _format_line, _read_logs, _render_dashboard, logs

SRC = Path(__file__).resolve().parents[2] / "apps" / "cli" / "src"


# ── monitor retirement ───────────────────────────────────────────────────────


def test_monitor_module_removed():
    assert not (SRC / "commands" / "monitor.py").exists(), (
        "monitor command retired — dashboard lives in `logs --dashboard`"
    )


def test_cli_py_has_no_monitor_registration():
    text = (SRC / "cli.py").read_text()
    assert "commands.monitor" not in text
    assert '"monitor"' not in text


def test_system_command_group_dropped_monitor():
    text = (SRC / "core" / "slo_cli.py").read_text()
    assert '"monitor"' not in text, "System group must not list the retired monitor command"


# ── flag parity: everything monitor offered still exists on logs ─────────────


def test_logs_keeps_every_monitor_flag():
    names = {p.name for p in logs.params}
    for flag in ("interval", "host", "port", "output_json", "no_clear", "dashboard", "compact"):
        assert flag in names, f"logs lost the {flag} flag that monitor offered"


def test_logs_help_documents_dashboard():
    result = CliRunner().invoke(logs, ["--help"])
    assert result.exit_code == 0
    assert "--dashboard" in result.output
    assert "--interval" in result.output
    assert "--compact" in result.output


# ── dashboard render: punchy + de-cluttered ──────────────────────────────────


def _snapshot() -> dict:
    return {
        "data": {
            "health": {
                "model_loaded": True,
                "model_type": "SloNet-1B",
                "uptime_seconds": 3725,
                "cpu_percent": 12.5,
                "memory_percent": 40.2,
                "memory_used_mb": 512,
                "requests_per_minute": 7,
                "request_count": 42,
                "error_count": 1,
                "tokens_per_sec": 18.5,
                "avg_latency_ms": 44.0,
                "rpm_history": [1, 2, 3],
                "mem_history": [40, 41],
                "health_score": {"score": 87, "status": "good"},
            },
            "processes": {
                "train:abc12345": {
                    "type": "training",
                    "status": "running",
                    "label": "tiny-shakespeare",
                    "detail": "step 10/100",
                    "progress": 42,
                },
                "self-train": {
                    "type": "self-train",
                    "status": "running",
                    "label": "self-train",
                    "detail": "pid 4242",
                    "progress": 0,
                },
            },
            "events": [{"ts": 1790000000.0, "category": "TRAIN", "message": "epoch done"}],
            "recent_errors": [{"path": "/v1/chat", "message": "boom"}],
        }
    }


def test_render_dashboard_headline_is_punchy(capsys):
    _render_dashboard(_snapshot(), clear=False)
    out = capsys.readouterr().out
    assert "SERVER" in out and "online" in out
    assert "SloNet-1B" in out
    assert "rpm 7" in out, "requests-per-minute was dropped in the fold — restore it"
    assert "87/100" in out, "health score must show in the headline"
    assert "good" in out


def test_render_dashboard_shows_all_processes(capsys):
    _render_dashboard(_snapshot(), clear=False)
    out = capsys.readouterr().out
    assert "tiny-shakespeare" in out
    assert "step 10/100" in out
    assert "pid 4242" in out
    assert "PROCESSES" in out


def test_render_dashboard_shows_events_and_errors(capsys):
    _render_dashboard(_snapshot(), clear=False)
    out = capsys.readouterr().out
    assert "TRAIN" in out and "epoch done" in out
    assert "RECENT ERRORS" in out and "/v1/chat" in out


def test_render_dashboard_is_de_cluttered(capsys):
    """Health merges into the headline; per-section rules are dropped."""
    _render_dashboard(_snapshot(), clear=False)
    out = capsys.readouterr().out
    assert "  HEALTH" not in out, "health score belongs on the SERVER line, not its own section"
    divider_lines = sum(1 for line in out.splitlines() if "─" * 20 in line)
    assert divider_lines <= 1, "only the title divider remains (sections de-cluttered)"


def test_render_dashboard_compact_keeps_headline(capsys):
    _render_dashboard(_snapshot(), clear=False, compact=True)
    out = capsys.readouterr().out
    assert "SERVER" in out and "87/100" in out and "rpm 7" in out
    assert "PROCESSES" not in out
    assert "EVENTS" not in out


def test_render_dashboard_empty_snapshot_does_not_crash(capsys):
    _render_dashboard({"data": {}}, clear=False)
    out = capsys.readouterr().out
    assert "no model" in out
    assert "(none active)" in out


def test_process_labels_colored_by_type_on_tty(monkeypatch, capsys):
    monkeypatch.setattr(logs_mod, "_TTY", True)
    _render_dashboard(_snapshot(), clear=False)
    out = capsys.readouterr().out
    # training label rendered in green on an interactive terminal
    assert "\033[32mtiny-shakespeare" in out


# ── colors finished: TTY colored, piped stays clean ──────────────────────────


def _rec(level="ERROR", msg="bad thing"):
    return {"ts": "2026-09-30T00:00:00", "level": level, "logger": "slo.x", "msg": msg, "tag": "T"}


def test_format_line_level_colors():
    colored = _format_line(_rec(), use_color=True)
    assert "\033[31m" in colored and "ERROR" in colored
    plain = _format_line(_rec(), use_color=False)
    assert "\033" not in plain


def test_no_matching_lines_message_honors_tty(tmp_path, capsys, monkeypatch):
    logfile = tmp_path / "app.log"
    logfile.write_text("Sep 18 09:31:02 slo.x[1]: INFO hello\n")
    filters = {"search": "does-not-match-anywhere"}

    monkeypatch.setattr(logs_mod, "_TTY", False)
    _read_logs(logfile, 10, filters, output_json=False, use_color=True)
    assert "\033" not in capsys.readouterr().err, "piped output must stay ANSI-free"

    monkeypatch.setattr(logs_mod, "_TTY", True)
    _read_logs(logfile, 10, filters, output_json=False, use_color=True)
    assert "\033[" in capsys.readouterr().err, "interactive output gets colored chrome"


def test_json_and_plain_modes_stay_ansi_free():
    assert "\033" not in _format_line(_rec(), use_color=False)
