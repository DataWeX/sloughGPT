"""
Error logging router — accepts frontend JS errors for server-side monitoring.

- POST /errors/log: log one or more client-side errors
- GET /errors/recent: retrieve recent errors (newest first)
- GET /errors/grouped: errors grouped by message fingerprint
- GET /errors/trends: error counts per hour for last 24h
- GET /errors/export: dump full error log as JSON
- DELETE /errors/clear: clear all errors
- GET /errors/unread: unread count
"""

import json
import logging
import hashlib
import uuid
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("man.errors")

router = APIRouter(prefix="/errors", tags=["errors"])

# In-memory ring buffer for recent errors
MAX_ERRORS = 500
_error_buffer: list[dict] = []
_error_count_since_clear = 0

# Disk persistence
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ERROR_LOG_DIR = _REPO_ROOT / "data" / "error_log"
_ERROR_LOG_FILE = _ERROR_LOG_DIR / "errors.jsonl"


def _ensure_dir():
    _ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _persist_to_disk(record: dict):
    """Append a single error record to the JSONL file on disk."""
    try:
        _ensure_dir()
        with open(_ERROR_LOG_FILE, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass  # best-effort — never block the request


def _load_from_disk() -> list[dict]:
    """Load all errors from disk (for startup / restart recovery)."""
    try:
        if _ERROR_LOG_FILE.exists():
            records = []
            for line in _ERROR_LOG_FILE.read_text().splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
            return records[-MAX_ERRORS:]
    except Exception:
        pass
    return []


def _clear_disk():
    """Clear the error log file on disk."""
    try:
        if _ERROR_LOG_FILE.exists():
            _ERROR_LOG_FILE.unlink()
    except Exception:
        pass


def _fingerprint(message: str) -> str:
    """Create a stable fingerprint for deduplication (ignore numbers/IDs)."""
    import re
    normalized = re.sub(r'\d+', 'N', message.lower())
    normalized = re.sub(r'[a-f0-9]{8,}', 'ID', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


# Load persisted errors on module init
_error_buffer = _load_from_disk()
_error_count_since_clear = 0  # reset on restart — only tracks current session


class ErrorEntry(BaseModel):
    message: str
    source: str = "web"
    stack: Optional[str] = None
    url: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None
    timestamp: Optional[str] = None
    metadata: Optional[dict] = None


class ErrorBatch(BaseModel):
    errors: List[ErrorEntry]


@router.post("/log")
async def log_errors(batch: ErrorBatch, request: Request):
    """
    Accept frontend JS errors and store them in the ring buffer + disk.

    Returns a 201 with the count of logged errors.
    """
    global _error_count_since_clear
    now = datetime.now(timezone.utc).isoformat()
    client_host = request.client.host if request.client else "unknown"

    for entry in batch.errors:
        error_record = {
            "id": uuid.uuid4().hex[:12],
            "message": entry.message,
            "source": entry.source or "web",
            "stack": entry.stack,
            "url": entry.url,
            "line": entry.line,
            "col": entry.col,
            "client_host": client_host,
            "timestamp": entry.timestamp or now,
            "metadata": entry.metadata or {},
            "fingerprint": _fingerprint(entry.message),
        }
        _error_buffer.append(error_record)
        _error_count_since_clear += 1
        _persist_to_disk(error_record)
        logger.error(
            "CLIENT ERROR [%s] %s | %s:%s %s",
            error_record["id"],
            entry.message[:120],
            entry.url or "?",
            entry.line or 0,
            entry.col or 0,
        )

    # Trim buffer
    while len(_error_buffer) > MAX_ERRORS:
        _error_buffer.pop(0)

    return {"status": "ok", "logged": len(batch.errors)}


@router.get("/recent")
async def get_recent_errors(limit: int = 50, offset: int = 0):
    """Return paginated client errors (newest first)."""
    total = len(_error_buffer)
    start = max(0, total - offset - limit)
    end = max(0, total - offset)
    return {
        "errors": list(reversed(_error_buffer[start:end])),
        "unread_count": _error_count_since_clear,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.get("/grouped")
async def get_grouped_errors():
    """Return errors grouped by message fingerprint with counts."""
    groups: dict[str, dict] = {}
    for entry in reversed(_error_buffer):
        fp = entry.get("fingerprint") or _fingerprint(entry.get("message", ""))
        if fp in groups:
            groups[fp]["count"] += 1
            groups[fp]["latest"] = entry.get("timestamp", "")
        else:
            groups[fp] = {
                "fingerprint": fp,
                "message": entry.get("message", "")[:200],
                "source": entry.get("source", ""),
                "count": 1,
                "latest": entry.get("timestamp", ""),
                "sample_id": entry.get("id", ""),
                "sample_url": entry.get("url", ""),
                "sample_line": entry.get("line"),
            }
    result = sorted(groups.values(), key=lambda g: g["count"], reverse=True)
    return {"groups": result, "total_groups": len(result)}


@router.get("/trends")
async def get_error_trends(hours: int = 24):
    """Return error counts per hour for the last N hours."""
    now = datetime.now(timezone.utc)
    buckets: dict[str, int] = {}

    # Initialize all buckets
    for h in range(hours):
        t = now.replace(minute=0, second=0, microsecond=0)
        from datetime import timedelta
        t = t - timedelta(hours=h)
        key = t.strftime("%Y-%m-%dT%H:00")
        buckets[key] = 0

    # Fill from buffer
    for entry in _error_buffer:
        ts = entry.get("timestamp", "")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            key = t.strftime("%Y-%m-%dT%H:00")
            if key in buckets:
                buckets[key] += 1
        except (ValueError, TypeError):
            pass

    # Also count from disk file
    try:
        if _ERROR_LOG_FILE.exists():
            for line in _ERROR_LOG_FILE.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts = rec.get("timestamp", "")
                    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    key = t.strftime("%Y-%m-%dT%H:00")
                    if key in buckets:
                        buckets[key] += 1
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass
    except Exception:
        pass

    result = [{"hour": k, "count": v} for k, v in sorted(buckets.items())]
    return {"trends": result, "hours": hours}


@router.get("/export")
async def export_errors(limit: int = 500):
    """Dump the full error log as a downloadable JSON response."""
    errors = list(reversed(_error_buffer[-limit:]))
    return JSONResponse(
        content={"errors": errors, "total": len(_error_buffer), "exported": len(errors)},
        headers={
            "Content-Disposition": f'attachment; filename="errors-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}.json"'
        },
    )


@router.delete("/clear")
async def clear_errors():
    """Clear all stored errors and reset unread counter."""
    _error_buffer.clear()
    _clear_disk()
    global _error_count_since_clear
    _error_count_since_clear = 0
    return {"status": "ok", "cleared": True}


@router.get("/unread")
async def unread_count():
    """Return the number of errors logged since last clear."""
    return {"unread_count": _error_count_since_clear}


@router.get("/log")
async def get_opencode_log():
    """
    Read error logs from the opencode log-monitor and ui-error-watcher plugins.
    Returns a unified list of log entries from both ~/.opencode-error-log.json
    and ~/.opencode-ui-error-log.json, sorted newest-first.
    """
    import os
    home = os.path.expanduser("~")
    entries: list[dict] = []

    # Read CLI error log (log-monitor plugin)
    cli_log_path = os.path.join(home, ".opencode-error-log.json")
    try:
        if os.path.exists(cli_log_path):
            with open(cli_log_path) as f:
                data = json.load(f)
            for rec in data:
                entries.append({
                    "timestamp": rec.get("timestamp", ""),
                    "command": rec.get("command", ""),
                    "category": "cli",
                    "pattern": rec.get("pattern", ""),
                    "snippet": rec.get("snippet", ""),
                    "cwd": rec.get("cwd", ""),
                })
    except Exception:
        pass

    # Read UI error log (ui-error-watcher plugin)
    ui_log_path = os.path.join(home, ".opencode-ui-error-log.json")
    try:
        if os.path.exists(ui_log_path):
            with open(ui_log_path) as f:
                data = json.load(f)
            for rec in data:
                entries.append({
                    "timestamp": rec.get("timestamp", ""),
                    "command": rec.get("command", ""),
                    "category": rec.get("category", "unknown"),
                    "pattern": rec.get("pattern", ""),
                    "snippet": rec.get("snippet", ""),
                    "cwd": rec.get("cwd", ""),
                })
    except Exception:
        pass

    # Sort newest first
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return {"entries": entries[:200], "total": len(entries)}
