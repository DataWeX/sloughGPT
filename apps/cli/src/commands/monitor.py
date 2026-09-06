"""
Monitor command — live CLI dashboard for all server processes.

Connects to /dashboard/stream SSE endpoint and renders a compact
ANSI terminal view with server health, active processes, and
recent events.

Usage:
    sloughgpt monitor                  # default 2s refresh
    sloughgpt monitor --interval 1     # 1s refresh
    sloughgpt monitor --no-clear       # append mode (no screen clear)
    sloughgpt monitor --json           # raw JSON lines
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import click

from domains.logging import get_global

log = get_global()


# ── ANSI helpers ────────────────────────────────────────────────────────

_TTY = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"{code}{text}\033[0m" if _TTY else text

_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_GREY = "\033[90m"

_CLEAR = "\033[2J\033[H"


def _line(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _format_uptime(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s:02d}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h {m:02d}m"


def _format_ts(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%H:%M:%S")


def _category_color(cat: str) -> str:
    colors = {
        "TRAIN": _GREEN,
        "MODEL": _CYAN,
        "INFERENCE": "\033[35m",
        "SYSTEM": _GREY,
        "ERROR": _RED,
        "INFRA": _GREY,
        "CHAT": _CYAN,
        "SOUL": _YELLOW,
        "START": _GREY,
        "IDLE": _GREY,
        "UI": _BLUE,
    }
    return colors.get(cat, "")


def _status_icon(status: str) -> str:
    icons = {
        "running": _c("run", _GREEN),
        "queued": _c("wait", _YELLOW),
        "starting": _c("init", _CYAN),
        "complete": _c("ok", _GREEN),
        "completed": _c("ok", _GREEN),
        "error": _c("err", _RED),
        "exited": _c("stop", _GREY),
        "idle": _c("idle", _GREY),
        "stopped": _c("stop", _GREY),
    }
    return icons.get(status, _c("?", _YELLOW))


def _progress_bar(progress: float, width: int = 20) -> str:
    filled = int(width * progress / 100)
    empty = width - filled
    return _c("█" * filled, _GREEN) + _c("░" * empty, _GREY)


def _render_dashboard(snapshot: dict, clear: bool = True) -> None:
    """Render a single dashboard snapshot to the terminal."""
    data = snapshot.get("data", {})
    health = data.get("health", {})
    processes = data.get("processes", {})
    events = data.get("events", [])

    model = health.get("model_type", "")
    model_str = model if model else _c("no model", _DIM)
    uptime = health.get("uptime_seconds", 0)
    cpu = health.get("cpu_percent", 0)
    mem = health.get("memory_percent", 0)
    mem_mb = health.get("memory_used_mb", 0)
    rpm = health.get("requests_per_minute", 0)
    tps = health.get("tokens_per_sec", 0)
    reqs = health.get("request_count", 0)
    errs = health.get("error_count", 0)

    if clear:
        sys.stdout.write(_CLEAR)
    _line(f"  {_c('SloughGPT Monitor', _BOLD + _CYAN)}  {_c(_format_ts(time.time()), _DIM)}")
    _line(f"  {'─' * 56}")

    loaded = health.get("model_loaded", False)
    status_str = _c("online", _GREEN) if loaded else _c("no model", _YELLOW)
    _line(f"  {_c('SERVER', _BOLD)} {status_str}  {_c(model_str, _CYAN)}  uptime {_format_uptime(uptime)}")

    cpu_bar = f"{cpu:.0f}%"
    mem_bar = f"{mem:.0f}% ({mem_mb}MB)"
    _line(f"  {_c('SYS', _BOLD)}   cpu {cpu_bar}  mem {mem_bar}  reqs {reqs}  errs {errs}  rpm {rpm:.0f}")

    if tps > 0:
        _line(f"  {_c('GEN', _BOLD)}   {tps:.1f} tok/s")

    _line()
    _line(f"  {_c('PROCESSES', _BOLD)}")
    _line(f"  {'─' * 56}")

    if not processes:
        _line(f"  {_c('  (none active)', _DIM)}")
    else:
        for proc_id, proc in processes.items():
            status = proc.get("status", "unknown")
            label = proc.get("label", proc_id)
            detail = proc.get("detail", "")
            progress = proc.get("progress", 0)

            icon = _status_icon(status)
            name = _c(label.ljust(14), _BOLD)

            if progress > 0 and status == "running":
                bar = _progress_bar(progress)
                pct = f"{progress:.0f}%".rjust(4)
                detail_str = f"{detail}  {bar} {pct}" if detail else f"{bar} {pct}"
            else:
                detail_str = detail

            _line(f"  {icon} {name} {_c(detail_str, _DIM)}")

    _line()
    _line(f"  {_c('EVENTS', _BOLD)}")
    _line(f"  {'─' * 56}")

    if not events:
        _line(f"  {_c('  (no events yet)', _DIM)}")
    else:
        for ev in events[:10]:
            ts = ev.get("ts", 0)
            cat = ev.get("category", "")
            msg = ev.get("message", "")

            ts_str = _c(_format_ts(ts), _GREY)
            cat_str = _c(cat.ljust(9), _category_color(cat))
            max_msg = 44
            if len(msg) > max_msg:
                msg = msg[:max_msg - 1] + "…"
            msg_str = _c(msg, "")

            _line(f"  {ts_str} {cat_str} {msg_str}")

    _line()
    _line(f"  {_c('Ctrl+C to exit', _DIM)}")
    sys.stdout.flush()


def _consume_sse(host: str, port: int, interval: float, output_json: bool, clear: bool) -> None:
    """Consume /dashboard/stream SSE and render the dashboard."""
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/dashboard/stream"
    error_count = 0

    while True:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                error_count = 0
                buffer = ""
                for chunk in iter(lambda: resp.read(1), b""):
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        for line in event_str.strip().split("\n"):
                            if line.startswith("data: "):
                                payload = line[6:]
                                try:
                                    snapshot = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue
                                if output_json:
                                    click.echo(json.dumps(snapshot))
                                else:
                                    _render_dashboard(snapshot, clear=clear)
                                time.sleep(interval)
        except urllib.error.URLError as e:
            error_count += 1
            if not output_json:
                if clear:
                    sys.stdout.write(_CLEAR)
                _line(f"  {_c('SloughGPT Monitor', _BOLD + _CYAN)}")
                _line(f"  {'─' * 56}")
                _line()
                _line(f"  {_c('Cannot connect to server', _RED)}")
                _line(f"  {_c(str(e), _DIM)}")
                _line()
                _line(f"  {_c(f'Retrying in 3s... (attempt {error_count})', _DIM)}")
                sys.stdout.flush()
            time.sleep(3)
        except KeyboardInterrupt:
            if not output_json:
                log.show_cursor()
                sys.stdout.write("\n")
                sys.stdout.flush()
            break
        except Exception as e:
            error_count += 1
            if not output_json:
                if clear:
                    sys.stdout.write(_CLEAR)
                _line(f"  {_c('SloughGPT Monitor', _BOLD + _CYAN)}")
                _line(f"  {'─' * 56}")
                _line()
                _line(f"  {_c('Stream error', _RED)}: {e}")
                _line()
                _line(f"  {_c(f'Retrying in 3s... (attempt {error_count})', _DIM)}")
                sys.stdout.flush()
            time.sleep(3)


def _poll_fallback(host: str, port: int, interval: float, output_json: bool, clear: bool) -> None:
    """Fallback: poll individual endpoints when SSE is unavailable."""
    import urllib.request

    endpoints = {
        "health": f"http://{host}:{port}/health",
        "events": f"http://{host}:{port}/dashboard/events",
    }

    while True:
        snapshot = {"data": {"health": {}, "processes": {}, "events": []}}

        try:
            req = urllib.request.Request(endpoints["health"])
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = json.loads(resp.read())
                snapshot["data"]["health"] = raw.get("data", raw)
        except (urllib.error.URLError, OSError, ValueError):
            pass

        try:
            req = urllib.request.Request(endpoints["events"])
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = json.loads(resp.read())
                data = raw.get("data", raw)
                snapshot["data"]["events"] = data.get("events", [])
        except (urllib.error.URLError, OSError, ValueError):
            pass

        if output_json:
            click.echo(json.dumps(snapshot))
        else:
            _render_dashboard(snapshot, clear=clear)

        time.sleep(interval)


@click.command(help="Live dashboard for all server processes")
@click.option("--interval", "-i", default=2.0, type=float, help="Refresh interval in seconds", show_default=True)
@click.option("--host", default="localhost", help="API hostname", show_default=True)
@click.option("--port", default=8000, type=int, help="API port", show_default=True)
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON lines instead of dashboard")
@click.option("--no-clear", is_flag=True, help="Append mode — don't clear screen between refreshes")
def monitor(interval: float, host: str, port: int, output_json: bool, no_clear: bool):
    """Live dashboard — server health, processes, and event feed."""
    clear = not no_clear

    if _TTY and not output_json:
        log.hide_cursor()

    try:
        _consume_sse(host, port, interval, output_json, clear)
    except (urllib.error.URLError, OSError, ValueError):
        try:
            _poll_fallback(host, port, interval, output_json, clear)
        except KeyboardInterrupt:
            pass
    finally:
        if _TTY and not output_json:
            log.show_cursor()
