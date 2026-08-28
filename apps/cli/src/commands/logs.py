"""
Logs command — query, filter, and monitor server logs from the CLI.

Usage:
    sloughgpt logs                          # tail last 50 lines
    sloughgpt logs --tail 200               # tail last 200 lines
    sloughgpt logs --follow                 # tail -f mode
    sloughgpt logs --dashboard              # live dashboard (health, processes, events)
    sloughgpt logs --dashboard --interval 1 # live dashboard, 1s refresh
    sloughgpt logs --dashboard --no-clear   # live dashboard, append mode
    sloughgpt logs --request-id abc123      # filter by correlation ID
    sloughgpt logs --level ERROR            # filter by log level
    sloughgpt logs --since 1h               # logs from last hour
    sloughgpt logs --tag INF                # filter by log tag
    sloughgpt logs --path /chat             # filter by request path
    sloughgpt logs --search "timeout"       # free-text search
    sloughgpt logs --errors-only            # only ERROR and WARNING lines
    sloughgpt logs --json                   # output raw JSON lines
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click


# ── ANSI helpers (shared by log formatting and dashboard) ───────────────

_TTY = sys.stdout.isatty()

_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_GREY = "\033[90m"
_CLEAR = "\033[2J\033[H"


def _c(text: str, code: str) -> str:
    return f"{code}{text}\033[0m" if _TTY else text


def _line(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


# ── Log file helpers ────────────────────────────────────────────────────

def _parse_since(since: str) -> Optional[datetime]:
    """Parse a relative time string like '1h', '30m', '2d' or an ISO timestamp."""
    if not since:
        return None

    if since[-1] in ('m', 'h', 'd'):
        unit = since[-1]
        try:
            value = int(since[:-1])
        except ValueError:
            return None
        if unit == 'm':
            delta = timedelta(minutes=value)
        elif unit == 'h':
            delta = timedelta(hours=value)
        else:
            delta = timedelta(days=value)
        return datetime.now(timezone.utc) - delta

    try:
        return datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _format_line(record: dict, use_color: bool = True) -> str:
    """Format a log record for display."""
    ts = record.get('ts', '')
    level = record.get('level', 'INFO')
    logger_name = record.get('logger', '')
    msg = record.get('msg', '')
    tag = record.get('tag', '')
    request_id = record.get('request_id', '')

    if use_color:
        level_colors = {
            'DEBUG': '\033[36m',
            'INFO': '\033[32m',
            'WARNING': '\033[33m',
            'ERROR': '\033[31m',
            'CRITICAL': '\033[35m',
        }
        reset = '\033[0m'
        level_str = f"{level_colors.get(level, '')}{level:8s}{reset}"
        tag_str = f"\033[90m[{tag}]\033[0m" if tag else ''
        rid_str = f"\033[90mrid={request_id}\033[0m" if request_id else ''
        ts_str = f"\033[90m{ts}\033[0m"
        logger_str = f"\033[90m{logger_name}\033[0m"
    else:
        level_str = f"{level:8s}"
        tag_str = f"[{tag}]" if tag else ''
        rid_str = f"rid={request_id}" if request_id else ''
        ts_str = ts
        logger_str = logger_name

    parts = [ts_str, level_str]
    if tag_str:
        parts.append(tag_str)
    if rid_str:
        parts.append(rid_str)
    parts.append(logger_str)
    parts.append(msg)

    return ' '.join(p for p in parts if p)


def _matches_filters(record: dict, filters: dict) -> bool:
    """Check if a log record matches all active filters."""
    if filters.get('request_id'):
        if filters['request_id'] != record.get('request_id', ''):
            return False

    if filters.get('level'):
        if filters['level'].upper() != record.get('level', '').upper():
            return False

    if filters.get('tag'):
        if filters['tag'] != record.get('tag', ''):
            return False

    if filters.get('path'):
        ctx = record.get('ctx', {})
        if isinstance(ctx, dict):
            req_ctx = ctx.get('context', {})
            if isinstance(req_ctx, dict):
                req_path = req_ctx.get('path', '')
                if filters['path'] not in req_path:
                    return False
        if filters['path'] not in record.get('msg', ''):
            return False

    if filters.get('search'):
        search = filters['search'].lower()
        msg = record.get('msg', '').lower()
        if search not in msg:
            return False

    if filters.get('since'):
        ts_str = record.get('ts', '')
        if ts_str:
            try:
                record_ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                if record_ts < filters['since']:
                    return False
            except ValueError:
                pass

    if filters.get('errors_only'):
        level = record.get('level', '').upper()
        if level not in ('ERROR', 'WARNING', 'CRITICAL'):
            return False

    return True


# ── Dashboard helpers ──────────────────────────────────────────────────

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
    }
    return colors.get(cat, "")


def _status_icon(status: str) -> str:
    icons = {
        "running": _c("▶", _GREEN),
        "queued": _c("◆", _YELLOW),
        "starting": _c("↻", _CYAN),
        "complete": _c("✓", _GREEN),
        "completed": _c("✓", _GREEN),
        "error": _c("✗", _RED),
        "exited": _c("■", _GREY),
        "idle": _c("·", _GREY),
        "stopped": _c("■", _GREY),
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
    _line(f"  {_c('SloughGPT', _BOLD + _CYAN)}  {_c(_format_ts(time.time()), _DIM)}")
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


def _consume_sse_dashboard(host: str, port: int, interval: float, output_json: bool, clear: bool) -> None:
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
                _line(f"  {_c('SloughGPT', _BOLD + _CYAN)}")
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
                sys.stdout.write("\033[?25h\n")
                sys.stdout.flush()
            break
        except Exception as e:
            error_count += 1
            if not output_json:
                if clear:
                    sys.stdout.write(_CLEAR)
                _line(f"  {_c('SloughGPT', _BOLD + _CYAN)}")
                _line(f"  {'─' * 56}")
                _line()
                _line(f"  {_c('Stream error', _RED)}: {e}")
                _line()
                _line(f"  {_c(f'Retrying in 3s... (attempt {error_count})', _DIM)}")
                sys.stdout.flush()
            time.sleep(3)


def _poll_fallback_dashboard(host: str, port: int, interval: float, output_json: bool, clear: bool) -> None:
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
        except Exception:
            pass

        try:
            req = urllib.request.Request(endpoints["events"])
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = json.loads(resp.read())
                data = raw.get("data", raw)
                snapshot["data"]["events"] = data.get("events", [])
        except Exception:
            pass

        if output_json:
            click.echo(json.dumps(snapshot))
        else:
            _render_dashboard(snapshot, clear=clear)

        time.sleep(interval)


# ── Log file modes ─────────────────────────────────────────────────────

def _read_logs(log_path: Path, tail: int, filters: dict, output_json: bool, use_color: bool):
    """Read log file and apply filters."""
    matches = []

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if _matches_filters(record, filters):
                matches.append(record)

    matches = matches[-tail:]

    for record in matches:
        if output_json:
            click.echo(json.dumps(record))
        else:
            click.echo(_format_line(record, use_color=use_color))

    if not matches:
        click.echo("No matching log lines found.", err=True)


def _tail_follow(log_path: Path, filters: dict, output_json: bool, use_color: bool):
    """Follow log file like tail -f."""
    click.echo(f"Following {log_path} (Ctrl+C to stop)...", err=True)

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        f.seek(0, 2)

        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if _matches_filters(record, filters):
                    if output_json:
                        click.echo(json.dumps(record))
                    else:
                        click.echo(_format_line(record, use_color=use_color))
        except KeyboardInterrupt:
            click.echo("\nStopped following.", err=True)


# ── CLI entry point ───────────────────────────────────────────────────

@click.command(help="Query, filter, and monitor server logs")
@click.option('--tail', '-n', default=50, type=int, help='Number of lines to show (from end)')
@click.option('--request-id', '-r', default=None, help='Filter by correlation request ID')
@click.option('--level', '-l', default=None, help='Filter by log level (DEBUG, INFO, WARNING, ERROR)')
@click.option('--since', '-s', default=None, help='Show logs since: relative (30m, 1h, 2d) or ISO timestamp')
@click.option('--tag', '-t', default=None, help='Filter by log tag (REQ, INF, TRAIN, MODEL, etc.)')
@click.option('--path', '-p', default=None, help='Filter by request path (substring match)')
@click.option('--search', default=None, help='Free-text search in message')
@click.option('--errors-only', is_flag=True, help='Show only ERROR and WARNING lines')
@click.option('--json', 'output_json', is_flag=True, help='Output raw JSON lines')
@click.option('--file', 'log_file', default=None, type=click.Path(exists=True), help='Log file path (default: logs/sloughgpt.log)')
@click.option('--follow', '-f', is_flag=True, help='Follow (tail -f) new log lines')
@click.option('--dashboard', '-d', is_flag=True, help='Live dashboard: server health, processes, events')
@click.option('--interval', '-i', default=2.0, type=float, help='Dashboard refresh interval (s)', show_default=True)
@click.option('--no-clear', is_flag=True, help='Dashboard: append mode, no screen clear')
@click.option('--host', default='localhost', help='Dashboard: API hostname', show_default=True)
@click.option('--port', default=8000, type=int, help='Dashboard: API port', show_default=True)
def logs(tail, request_id, level, since, tag, path, search, errors_only, output_json, log_file, follow,
         dashboard, interval, no_clear, host, port):
    """Query, filter, and monitor server logs."""
    use_color = sys.stdout.isatty() and not output_json

    # ── Dashboard mode ──────────────────────────────────────────────
    if dashboard:
        clear = not no_clear
        if _TTY and not output_json:
            sys.stdout.write("\033[?25l")
        try:
            _consume_sse_dashboard(host, port, interval, output_json, clear)
        except Exception:
            try:
                _poll_fallback_dashboard(host, port, interval, output_json, clear)
            except KeyboardInterrupt:
                pass
        finally:
            if _TTY and not output_json:
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
        return

    # ── Log file mode ───────────────────────────────────────────────
    if log_file:
        log_path = Path(log_file)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        log_path = repo_root / 'logs' / 'sloughgpt.log'

    if not log_path.exists():
        click.echo(f"Log file not found: {log_path}", err=True)
        sys.exit(1)

    since_dt = _parse_since(since) if since else None

    filters = {
        'request_id': request_id,
        'level': level,
        'tag': tag,
        'path': path,
        'search': search,
        'since': since_dt,
        'errors_only': errors_only,
    }

    if follow:
        _tail_follow(log_path, filters, output_json, use_color)
    else:
        _read_logs(log_path, tail, filters, output_json, use_color)
