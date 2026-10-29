"""`sloughgpt logs --dashboard` must render a stalled snapshot as a stall.

`/dashboard/stream` now emits `phase=STALL` when `_build_snapshot` fails to
return within `DASHBOARD_SNAPSHOT_STALL_S` (routers/dashboard.py). A STALL
frame carries no health data on purpose — it is the signal itself.

Before this, `_render_dashboard` read only `data.health/processes/events` and
never looked at `phase`, so a stall frame rendered as an empty dashboard
saying "no model" with uptime 0: visually identical to a server that is idle
and perfectly healthy. That relocates the exact defect the server-side frame
exists to kill — silence reading as "nothing to report" — one hop downstream.
"""

from __future__ import annotations

from apps.cli.src.commands.logs import _render_dashboard


def _stall_frame() -> dict:
    """Shape emitted by routers/dashboard.py past the stall bound."""
    return {
        "stream": "dashboard",
        "phase": "STALL",
        "status": "working",
        "data": {"stalled": True, "waited_seconds": 6.0, "interval_seconds": 2.0},
        "message": "dashboard snapshot has not returned in 6s; a subsystem is likely blocked",
    }


def _error_frame() -> dict:
    return {
        "stream": "dashboard",
        "phase": "ERROR",
        "status": "error",
        "data": {"error": "connection refused"},
        "message": "connection refused",
    }


def test_stall_frame_renders_as_a_stall(capsys):
    _render_dashboard(_stall_frame(), clear=False)
    out = capsys.readouterr().out
    assert "SNAPSHOT STALLED" in out
    assert "subsystem is likely blocked" in out
    assert "cadence 2s" in out, "the observer needs the cadence to judge the stall"


def test_stall_frame_does_not_render_as_an_idle_server(capsys):
    """The defect this fixes: a stall must not look like a healthy idle box."""
    _render_dashboard(_stall_frame(), clear=False)
    out = capsys.readouterr().out
    assert "no model" not in out, "a stall rendered as an idle server"
    assert "up 0s" not in out, "a stall rendered as an idle server"


def test_stall_is_not_dressed_up_as_a_failure(capsys):
    """A stall is status=working: the stream is alive and the build may return."""
    _render_dashboard(_stall_frame(), clear=False)
    out = capsys.readouterr().out
    assert "SNAPSHOT FAILED" not in out


def test_stall_without_a_message_still_renders_as_a_stall(capsys):
    frame = {"phase": "STALL", "status": "working", "data": {"stalled": True}}
    _render_dashboard(frame, clear=False)
    out = capsys.readouterr().out
    assert "SNAPSHOT STALLED" in out
    assert "a subsystem is likely blocked" in out


def test_error_frame_renders_its_message(capsys):
    _render_dashboard(_error_frame(), clear=False)
    out = capsys.readouterr().out
    assert "SNAPSHOT FAILED" in out
    assert "connection refused" in out
    assert "no model" not in out


def test_real_snapshot_still_renders_normally(capsys):
    """Guard: the phase check must not swallow an ordinary frame."""
    frame = {
        "stream": "dashboard",
        "phase": "DASHBOARD",
        "status": "working",
        "data": {
            "health": {
                "model_loaded": True,
                "model_type": "SloNet-1B",
                "uptime_seconds": 3725,
                "cpu_percent": 12.5,
                "memory_percent": 40.2,
            },
            "processes": {},
            "events": [],
            "recent_errors": [],
        },
    }
    _render_dashboard(frame, clear=False)
    out = capsys.readouterr().out
    assert "SERVER" in out and "online" in out
    assert "SloNet-1B" in out
    assert "SNAPSHOT STALLED" not in out
    assert "SNAPSHOT FAILED" not in out


def test_empty_snapshot_without_phase_still_renders(capsys):
    """A frame with no `phase` key at all keeps the old empty-dashboard path."""
    _render_dashboard({"data": {}}, clear=False)
    out = capsys.readouterr().out
    assert "no model" in out
    assert "SNAPSHOT STALLED" not in out
