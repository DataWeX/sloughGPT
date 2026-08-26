"""
Logs command — query and filter server logs from the CLI.

Usage:
    sloughgpt logs                          # tail last 50 lines
    sloughgpt logs --tail 200               # tail last 200 lines
    sloughgpt logs --request-id abc123      # filter by correlation ID
    sloughgpt logs --level ERROR            # filter by log level
    sloughgpt logs --since 1h               # logs from last hour
    sloughgpt logs --since 2026-08-26T15:00 # logs since timestamp
    sloughgpt logs --tag INF                # filter by log tag
    sloughgpt logs --path /chat             # filter by request path
    sloughgpt logs --search "timeout"       # free-text search
    sloughgpt logs --errors-only            # only ERROR and WARNING lines
    sloughgpt logs --json                   # output raw JSON lines
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click


def _parse_since(since: str) -> Optional[datetime]:
    """Parse a relative time string like '1h', '30m', '2d' or an ISO timestamp."""
    if not since:
        return None

    # Relative: 30m, 1h, 2d
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

    # ISO timestamp
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

    # Level coloring
    if use_color:
        level_colors = {
            'DEBUG': '\033[36m',    # cyan
            'INFO': '\033[32m',     # green
            'WARNING': '\033[33m',  # yellow
            'ERROR': '\033[31m',    # red
            'CRITICAL': '\033[35m', # magenta
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
        # Also check the message for path references
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


@click.command(help="Query and filter server logs")
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
def logs(tail, request_id, level, since, tag, path, search, errors_only, output_json, log_file, follow):
    """Query and filter server logs."""
    use_color = sys.stdout.isatty() and not output_json

    # Resolve log file
    if log_file:
        log_path = Path(log_file)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        log_path = repo_root / 'logs' / 'sloughgpt.log'

    if not log_path.exists():
        click.echo(f"Log file not found: {log_path}", err=True)
        sys.exit(1)

    # Parse since filter
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
        # Tail -f mode
        _tail_follow(log_path, filters, output_json, use_color)
    else:
        # Read and filter
        _read_logs(log_path, tail, filters, output_json, use_color)


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

    # Take last N
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
    import time

    click.echo(f"Following {log_path} (Ctrl+C to stop)...", err=True)

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        # Seek to end
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
