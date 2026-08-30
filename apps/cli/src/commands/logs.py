"""
Logs command — query, filter, and monitor server logs from the CLI.

Usage:
    sloughgpt logs                          # tail last 50 lines
    sloughgpt logs --follow                 # tail -f mode
    sloughgpt logs --dashboard              # live dashboard (health, processes, events)
    sloughgpt logs --dashboard --interval 1 # live dashboard, 1s refresh
    sloughgpt logs --dashboard --no-clear   # live dashboard, append mode
    sloughgpt logs --level ERROR            # filter by log level
    sloughgpt logs --tag TRAIN              # filter by log tag
    sloughgpt logs --search "timeout"       # free-text search
    sloughgpt logs --errors-only            # only ERROR and WARNING lines
    sloughgpt logs --stats                  # quick log summary
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click

from domains.logging import get_global

log = get_global()


# ── ANSI helpers ───────────────────────────────────────────────────────

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


# ── Sparkline ──────────────────────────────────────────────────────────

def _sparkline(values: list[float], width: int = 12) -> str:
    if not values:
        return ""
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    return "".join(blocks[min(int((v - mn) / rng * 8) + 1, 8)] for v in sampled)


# ── Log file helpers ───────────────────────────────────────────────────

def _parse_since(since: str) -> Optional[datetime]:
    if not since:
        return None
    if since[-1] in ('m', 'h', 'd'):
        unit = since[-1]
        try:
            value = int(since[:-1])
        except ValueError:
            return None
        delta = timedelta(minutes=value) if unit == 'm' else (
            timedelta(hours=value) if unit == 'h' else timedelta(days=value)
        )
        return datetime.now(timezone.utc) - delta
    try:
        return datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _format_line(record: dict, use_color: bool = True) -> str:
    ts = record.get('ts', '')
    level = record.get('level', 'INFO')
    logger_name = record.get('logger', '')
    msg = record.get('msg', '')
    tag = record.get('tag', '')
    request_id = record.get('request_id', '')

    if use_color:
        level_colors = {
            'DEBUG': '\033[36m', 'INFO': '\033[32m',
            'WARNING': '\033[33m', 'ERROR': '\033[31m', 'CRITICAL': '\033[35m',
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
        ts_str, logger_str = ts, logger_name

    parts = [ts_str, level_str]
    if tag_str: parts.append(tag_str)
    if rid_str: parts.append(rid_str)
    parts.extend([logger_str, msg])
    return ' '.join(p for p in parts if p)


def _matches_filters(record: dict, filters: dict) -> bool:
    if filters.get('request_id') and filters['request_id'] != record.get('request_id', ''):
        return False
    if filters.get('level') and filters['level'].upper() != record.get('level', '').upper():
        return False
    if filters.get('tag') and filters['tag'] != record.get('tag', ''):
        return False
    if filters.get('path'):
        ctx = record.get('ctx', {})
        if isinstance(ctx, dict):
            req_ctx = ctx.get('context', {})
            if isinstance(req_ctx, dict) and filters['path'] not in req_ctx.get('path', ''):
                return False
        if filters['path'] not in record.get('msg', ''):
            return False
    if filters.get('search') and filters['search'].lower() not in record.get('msg', '').lower():
        return False
    if filters.get('since'):
        ts_str = record.get('ts', '')
        if ts_str:
            try:
                if datetime.fromisoformat(ts_str.replace('Z', '+00:00')) < filters['since']:
                    return False
            except ValueError:
                pass
    if filters.get('errors_only') and record.get('level', '').upper() not in ('ERROR', 'WARNING', 'CRITICAL'):
        return False
    return True


# ── Dashboard ──────────────────────────────────────────────────────────

def _format_uptime(seconds: int) -> str:
    if seconds < 60: return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s:02d}s"
    h, m = seconds // 3600, (seconds % 3600) // 60
    return f"{h}h {m:02d}m"


def _format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def _category_color(cat: str) -> str:
    return {
        "TRAIN": _GREEN, "MODEL": _CYAN, "INFERENCE": "\033[35m",
        "SYSTEM": _GREY, "ERROR": _RED, "INFRA": _GREY, "CHAT": _CYAN,
        "SOUL": _YELLOW, "START": _GREY, "IDLE": _GREY, "DOWNLOAD": _CYAN,
        "SLOW": _YELLOW, "WORKFLOW": _GREY,
    }.get(cat, "")


def _status_icon(status: str) -> str:
    return {
        "running": _c("\u25b6", _GREEN), "queued": _c("\u25c6", _YELLOW),
        "starting": _c("\u21bb", _CYAN), "complete": _c("\u2713", _GREEN),
        "completed": _c("\u2713", _GREEN), "error": _c("\u2717", _RED),
        "exited": _c("\u25a0", _GREY), "idle": _c("\u00b7", _GREY), "stopped": _c("\u25a0", _GREY),
    }.get(status, _c("?", _YELLOW))


def _progress_bar(progress: float, width: int = 20) -> str:
    filled = int(width * progress / 100)
    return _c("\u2588" * filled, _GREEN) + _c("\u2591" * (width - filled), _GREY)


def _render_dashboard(snapshot: dict, clear: bool = True, compact: bool = False) -> None:
    data = snapshot.get("data", {})
    health = data.get("health", {})
    processes = data.get("processes", {})
    events = data.get("events", [])
    errors = data.get("recent_errors", [])

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
    lat = health.get("avg_latency_ms", 0)
    rpm_history = health.get("rpm_history", [])
    mem_history = health.get("mem_history", [])
    gpu = health.get("gpu", {})
    quant = health.get("quantization", {})
    health_score = health.get("health_score", {})

    if clear:
        sys.stdout.write(_CLEAR)

    _line(f"  {_c('SloughGPT', _BOLD + _CYAN)}  {_c(_format_ts(time.time()), _DIM)}")
    _line(f"  {'─' * 60}")

    loaded = health.get("model_loaded", False)
    status_str = _c("online", _GREEN) if loaded else _c("no model", _YELLOW)
    _line(f"  {_c('SERVER', _BOLD)} {status_str}  {_c(model_str, _CYAN)}  up {_format_uptime(uptime)}")

    # MODEL line — device, params, quantization
    if loaded:
        model_parts = []
        device = health.get("device", "")
        if device:
            model_parts.append(device)
        if gpu and gpu.get("backend") not in ("unknown", None):
            vram = gpu.get("vram_gb", 0)
            if vram:
                model_parts.append(f"{vram}GB")
        if quant:
            bits = quant.get("bits", 0)
            if bits:
                model_parts.append(f"{bits}bit")
        mode = quant.get("mode", "")
        if mode:
            model_parts.append(mode)
        if model_parts:
            _line(f"  {_c('MODEL', _BOLD)}   {'  '.join(model_parts)}")

    spark_rpm = _sparkline(rpm_history) if rpm_history else ""
    spark_mem = _sparkline(mem_history) if mem_history else ""
    sys_line = f"  {_c('SYS', _BOLD)}   cpu {cpu:.0f}%  mem {mem:.0f}% ({mem_mb}MB)  reqs {reqs}  err {errs}"
    if spark_rpm:
        sys_line += f"  {_c(spark_rpm, _GREY)}"
    if spark_mem:
        sys_line += f"  {_c(spark_mem, _GREY)}"
    _line(sys_line)

    gen_parts = []
    if tps > 0: gen_parts.append(f"{tps:.1f} tok/s")
    if lat > 0: gen_parts.append(f"{lat:.0f}ms avg")
    if gen_parts:
        _line(f"  {_c('GEN', _BOLD)}   {'  '.join(gen_parts)}")

    # Health score
    if health_score:
        score = health_score.get("score", 0)
        status = health_score.get("status", "")
        if score > 0:
            score_color = _GREEN if score >= 80 else _YELLOW if score >= 50 else _RED
            _line(f"  {_c('HEALTH', _BOLD)}  {_c(f'{score}/100', score_color)}  {_c(status, _DIM)}")

    if not compact:
        _line()
        _line(f"  {_c('PROCESSES', _BOLD)}")
        _line(f"  {'─' * 60}")

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
        _line(f"  {'─' * 60}")

        if not events:
            _line(f"  {_c('  (no events yet)', _DIM)}")
        else:
            for ev in events[:8]:
                ts_str = _c(_format_ts(ev.get("ts", 0)), _GREY)
                cat = ev.get("category", "")
                msg = ev.get("message", "")
                cat_str = _c(cat.ljust(10), _category_color(cat))
                max_msg = 42
                if len(msg) > max_msg:
                    msg = msg[:max_msg - 1] + "\u2026"
                _line(f"  {ts_str} {cat_str} {msg}")

        if errors:
            _line()
            _line(f"  {_c('RECENT ERRORS', _BOLD + _RED)}")
            _line(f"  {'─' * 60}")
            for err in errors[:3]:
                path = err.get("path", "")
                msg = err.get("message", "")[:40]
                _line(f"  {_c(path.ljust(20), _DIM)} {_c(msg, _RED)}")

    _line()
    _line(f"  {_c('Ctrl+C to exit', _DIM)}")
    sys.stdout.flush()


def _consume_sse_dashboard(host: str, port: int, interval: float, output_json: bool, clear: bool, compact: bool = False) -> None:
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
                                try:
                                    snapshot = json.loads(line[6:])
                                except json.JSONDecodeError:
                                    continue
                                if output_json:
                                    click.echo(json.dumps(snapshot))
                                else:
                                    _render_dashboard(snapshot, clear=clear, compact=compact)
                                time.sleep(interval)
        except urllib.error.URLError as e:
            error_count += 1
            if not output_json:
                if clear: sys.stdout.write(_CLEAR)
                _line(f"  {_c('SloughGPT', _BOLD + _CYAN)}")
                _line(f"  {'─' * 60}")
                _line(f"  {_c('Cannot connect to server', _RED)}")
                _line(f"  {_c(str(e), _DIM)}")
                _line(f"  {_c(f'Retrying... ({error_count})', _DIM)}")
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
                if clear: sys.stdout.write(_CLEAR)
                _line(f"  {_c('Stream error', _RED)}: {e}")
                sys.stdout.flush()
            time.sleep(3)


def _poll_fallback_dashboard(host: str, port: int, interval: float, output_json: bool, clear: bool, compact: bool = False) -> None:
    import urllib.request
    while True:
        snapshot = {"data": {"health": {}, "processes": {}, "events": [], "recent_errors": []}}
        for key, path in [("health", "/health"), ("events", "/dashboard/events")]:
            try:
                req = urllib.request.Request(f"http://{host}:{port}{path}")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    raw = json.loads(resp.read())
                    snapshot["data"][key] = raw.get("data", raw) if key == "health" else raw.get("data", {}).get("events", [])
            except Exception:
                pass
        try:
            req = urllib.request.Request(f"http://{host}:{port}/health/detailed")
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = json.loads(resp.read())
                data = raw.get("data", raw)
                snapshot["data"]["recent_errors"] = data.get("recent_errors", [])
        except Exception:
            pass
        if output_json:
            click.echo(json.dumps(snapshot))
        else:
            _render_dashboard(snapshot, clear=clear, compact=compact)
        time.sleep(interval)


# ── Stats mode ─────────────────────────────────────────────────────────

def _show_stats(log_path: Path, output_json: bool, use_color: bool) -> None:
    """Show quick log file statistics."""
    level_counts = {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}
    tag_counts: dict[str, int] = {}
    total = 0
    first_ts = None
    last_ts = None
    recent_errors = []

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            level = record.get("level", "INFO")
            tag = record.get("tag", "")
            ts = record.get("ts", "")

            if level in level_counts:
                level_counts[level] += 1
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
            if level in ("ERROR", "CRITICAL") and len(recent_errors) < 5:
                recent_errors.append(record)

    if output_json:
        click.echo(json.dumps({
            "total": total, "levels": level_counts, "tags": tag_counts,
            "first_ts": first_ts, "last_ts": last_ts,
            "recent_errors": [_format_line(e, use_color=False) for e in recent_errors],
        }))
        return

    _line(f"  {_c('Log Stats', _BOLD + _CYAN)}  {_c(str(log_path), _DIM)}")
    _line(f"  {'─' * 50}")
    _line(f"  Total lines:  {_c(str(total), _BOLD)}")
    if first_ts and last_ts:
        _line(f"  Time range:   {_c(first_ts[:19], _DIM)} → {_c(last_ts[:19], _DIM)}")
    _line()

    _line(f"  {_c('Levels', _BOLD)}")
    for lvl in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        count = level_counts[lvl]
        if count > 0:
            color = {"DEBUG": _GREY, "INFO": _GREEN, "WARNING": _YELLOW, "ERROR": _RED, "CRITICAL": _RED + _BOLD}.get(lvl, "")
            _line(f"    {_c(lvl.ljust(12), color)} {count}")

    if tag_counts:
        _line()
        _line(f"  {_c('Tags', _BOLD)}")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:10]:
            _line(f"    {tag.ljust(12)} {count}")

    if recent_errors:
        _line()
        _line(f"  {_c('Recent Errors', _BOLD + _RED)}")
        for rec in recent_errors[-3:]:
            _line(f"    {_c(_format_line(rec, use_color=use_color), _RED)}")

    _line()


# ── Log file modes ─────────────────────────────────────────────────────

def _read_logs(log_path: Path, tail: int, filters: dict, output_json: bool, use_color: bool):
    matches = []
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _matches_filters(record, filters):
                matches.append(record)
    for record in matches[-tail:]:
        if output_json:
            click.echo(json.dumps(record))
        else:
            click.echo(_format_line(record, use_color=use_color))
    if not matches:
        click.echo("No matching log lines found.", err=True)


def _tail_follow(log_path: Path, filters: dict, output_json: bool, use_color: bool):
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
                if not line: continue
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
@click.option('--level', '-l', default=None, help='Filter by log level')
@click.option('--since', '-s', default=None, help='Show logs since: relative (30m, 1h, 2d) or ISO timestamp')
@click.option('--tag', '-t', default=None, help='Filter by log tag')
@click.option('--path', '-p', default=None, help='Filter by request path')
@click.option('--search', default=None, help='Free-text search in message')
@click.option('--errors-only', is_flag=True, help='Show only ERROR and WARNING lines')
@click.option('--json', 'output_json', is_flag=True, help='Output raw JSON lines')
@click.option('--file', 'log_file', default=None, type=click.Path(exists=True), help='Log file path')
@click.option('--follow', '-f', is_flag=True, help='Follow (tail -f) new log lines')
@click.option('--dashboard', '-d', is_flag=True, help='Live dashboard: health, processes, events')
@click.option('--compact', is_flag=True, help='Dashboard: compact mode (less detail)')
@click.option('--interval', '-i', default=2.0, type=float, help='Dashboard refresh interval (s)')
@click.option('--no-clear', is_flag=True, help='Dashboard: append mode, no screen clear')
@click.option('--host', default='localhost', help='Dashboard: API hostname')
@click.option('--port', default=8000, type=int, help='Dashboard: API port')
@click.option('--stats', is_flag=True, help='Show log file statistics')
def logs(tail, request_id, level, since, tag, path, search, errors_only, output_json, log_file, follow,
         dashboard, compact, interval, no_clear, host, port, stats):
    """Query, filter, and monitor server logs."""
    use_color = sys.stdout.isatty() and not output_json

    if dashboard:
        clear = not no_clear
        if _TTY and not output_json:
            log.hide_cursor()
        try:
            _consume_sse_dashboard(host, port, interval, output_json, clear, compact)
        except Exception:
            try:
                _poll_fallback_dashboard(host, port, interval, output_json, clear, compact)
            except KeyboardInterrupt:
                pass
        finally:
            if _TTY and not output_json:
                log.show_cursor()
        return

    # Resolve log file
    if log_file:
        log_path = Path(log_file)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        log_path = repo_root / 'logs' / 'sloughgpt.log'

    if not log_path.exists():
        click.echo(f"Log file not found: {log_path}", err=True)
        sys.exit(1)

    if stats:
        _show_stats(log_path, output_json, use_color)
        return

    since_dt = _parse_since(since) if since else None
    filters = {
        'request_id': request_id, 'level': level, 'tag': tag,
        'path': path, 'search': search, 'since': since_dt, 'errors_only': errors_only,
    }

    if follow:
        _tail_follow(log_path, filters, output_json, use_color)
    else:
        _read_logs(log_path, tail, filters, output_json, use_color)
