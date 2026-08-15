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
import re

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from schemas.common import success_response

logger = logging.getLogger("slo.errors")

# ── Constants ─────────────────────────────────────────────────────────────

MAX_ERRORS = 500
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ERROR_LOG_DIR = _REPO_ROOT / "data" / "error_log"
_ERROR_LOG_FILE = _ERROR_LOG_DIR / "errors.jsonl"


# ── Pydantic models ──────────────────────────────────────────────────────

class ErrorEntry(BaseModel):
    message: str = Field(max_length=5000)
    source: str = Field(default="web", max_length=100)
    stack: Optional[str] = Field(default=None, max_length=20000)
    url: Optional[str] = Field(default=None, max_length=2000)
    line: Optional[int] = None
    col: Optional[int] = None
    timestamp: Optional[str] = None
    metadata: Optional[dict] = None


class ErrorBatch(BaseModel):
    errors: List[ErrorEntry] = Field(max_length=100)


class FrontendLogEntry(BaseModel):
    level: str = Field(default="info", max_length=20)
    logger: str = Field(default="web", max_length=100)
    message: str = Field(default="", max_length=5000)
    timestamp: float = 0.0
    context: Optional[dict] = None
    exception: Optional[str] = Field(default=None, max_length=20000)


class FrontendLogBatch(BaseModel):
    logs: List[FrontendLogEntry] = Field(max_length=100)


# ── Router class ─────────────────────────────────────────────────────────

class ErrorsRouter:
    """OOP router for error-logging endpoints."""

    # Dedup: message -> last-seen timestamp.  Identical errors within this
    # window are collapsed into a count bump instead of new records.
    _DEDUP_WINDOW_S = 10

    def __init__(self):
        self.router = APIRouter(prefix="/errors", tags=["errors"])
        self._error_buffer: list[dict] = []
        self._error_count_since_clear = 0
        self._dedup_map: dict[str, float] = {}

        self._error_buffer = self._load_from_disk()
        self._error_count_since_clear = 0

        self._register_routes()

    # ── Route registration ──────────────────────────────────────────────

    def _register_routes(self):
        self.router.add_api_route("/log", self.log_errors, methods=["POST"])
        self.router.add_api_route("/logs/ingest", self.ingest_frontend_logs, methods=["POST"])
        self.router.add_api_route("/recent", self.get_recent_errors, methods=["GET"])
        self.router.add_api_route("/grouped", self.get_grouped_errors, methods=["GET"])
        self.router.add_api_route("/trends", self.get_error_trends, methods=["GET"])
        self.router.add_api_route("/export", self.export_errors, methods=["GET"])
        self.router.add_api_route("/clear", self.clear_errors, methods=["DELETE"])
        self.router.add_api_route("/unread", self.unread_count, methods=["GET"])
        self.router.add_api_route("/log", self.get_opencode_log, methods=["GET"])

    # ── Helpers ─────────────────────────────────────────────────────────

    def _ensure_dir(self):
        _ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _persist_to_disk(self, record: dict):
        try:
            self._ensure_dir()
            with open(_ERROR_LOG_FILE, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="errors_persist")
            pass

    def _load_from_disk(self) -> list[dict]:
        try:
            if _ERROR_LOG_FILE.exists():
                records = []
                for line in _ERROR_LOG_FILE.read_text().splitlines():
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
                return records[-MAX_ERRORS:]
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="errors_load_from_disk")
            pass
        return []

    def _clear_disk(self):
        try:
            if _ERROR_LOG_FILE.exists():
                _ERROR_LOG_FILE.unlink()
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="errors_clear_disk")
            pass

    def _fingerprint(self, message: str) -> str:
        normalized = re.sub(r'\d+', 'N', message.lower())
        normalized = re.sub(r'[a-f0-9]{8,}', 'ID', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    # ── Route handlers ──────────────────────────────────────────────────

    async def log_errors(self, batch: ErrorBatch, request: Request):
        now_ts = datetime.now(timezone.utc)
        now_iso = now_ts.isoformat()
        now_epoch = now_ts.timestamp()
        client_host = request.client.host if request.client else "unknown"

        logged = 0
        batch_messages: set = set()
        for entry in batch.errors:
            message = entry.message or ""
            fp = self._fingerprint(message)

            # Dedup: skip if the same exact message arrived in a PRIOR
            # request within the window.  Never dedup within the current
            # batch — each entry in a batch is a distinct error event.
            last_seen = self._dedup_map.get(message)
            if (message not in batch_messages
                    and last_seen is not None
                    and (now_epoch - last_seen) < self._DEDUP_WINDOW_S):
                # Bump count on the existing record instead of creating a new one
                for rec in reversed(self._error_buffer):
                    if rec.get("message") == message:
                        rec["count"] = rec.get("count", 1) + 1
                        rec["timestamp"] = entry.timestamp or now_iso
                        break
                continue

            batch_messages.add(message)
            self._dedup_map[message] = now_epoch
            # Prune old dedup entries
            if len(self._dedup_map) > 500:
                self._dedup_map = {k: v for k, v in self._dedup_map.items()
                                   if (now_epoch - v) < self._DEDUP_WINDOW_S * 10}

            error_record = {
                "id": uuid.uuid4().hex[:12],
                "message": entry.message,
                "source": entry.source or "web",
                "stack": entry.stack,
                "url": entry.url,
                "line": entry.line,
                "col": entry.col,
                "client_host": client_host,
                "timestamp": entry.timestamp or now_iso,
                "metadata": entry.metadata or {},
                "fingerprint": fp,
                "count": 1,
            }
            self._error_buffer.append(error_record)
            self._error_count_since_clear += 1
            self._persist_to_disk(error_record)

            is_extension = any(x in (entry.url or "") for x in ("chrome-extension://", "moz-extension://", "extension://"))
            is_vague = entry.message in ("error", "Script error.", "Non-Error promise rejection")
            log_fn = logger.debug if (is_extension or is_vague) else logger.error
            log_fn(
                "CLIENT ERROR [%s] %s | %s:%s %s",
                error_record["id"],
                entry.message[:120],
                entry.url or "?",
                entry.line or 0,
                entry.col or 0,
            )
            logged += 1

        while len(self._error_buffer) > MAX_ERRORS:
            self._error_buffer.pop(0)

        return success_response(data={"status": "ok", "logged": logged})

    async def ingest_frontend_logs(self, batch: FrontendLogBatch):
        from domains.infrastructure.output_buffer import get_server_buffer
        from domains.logging import LogLevel as _LL

        buf = get_server_buffer()
        level_map = {
            "debug": "debug", "info": "info", "warning": "warning",
            "error": "error", "critical": "critical",
        }

        for entry in batch.logs:
            lvl = level_map.get(entry.level, "info")
            context: dict = {}
            if entry.context:
                context.update(entry.context)
            if entry.exception:
                context["exception"] = entry.exception

            buf.append_log(
                text=f"{entry.logger} {entry.message}",
                level=lvl,
                source=f"web.{entry.logger}",
                tag="WEB",
                context=context,
            )

        return success_response(data={"status": "ok", "ingested": len(batch.logs)})

    async def get_recent_errors(self, limit: int = 50, offset: int = 0):
        total = len(self._error_buffer)
        start = max(0, total - offset - limit)
        end = max(0, total - offset)
        return success_response(data={
            "errors": list(reversed(self._error_buffer[start:end])),
            "unread_count": self._error_count_since_clear,
            "total": total,
            "offset": offset,
            "limit": limit,
        })

    async def get_grouped_errors(self):
        groups: dict[str, dict] = {}
        for entry in reversed(self._error_buffer):
            fp = entry.get("fingerprint") or self._fingerprint(entry.get("message", ""))
            count = entry.get("count", 1)
            if fp in groups:
                groups[fp]["count"] += count
                groups[fp]["latest"] = entry.get("timestamp", "")
            else:
                groups[fp] = {
                    "fingerprint": fp,
                    "message": entry.get("message", "")[:200],
                    "source": entry.get("source", ""),
                    "count": count,
                    "latest": entry.get("timestamp", ""),
                    "sample_id": entry.get("id", ""),
                    "sample_url": entry.get("url", ""),
                    "sample_line": entry.get("line"),
                }
        result = sorted(groups.values(), key=lambda g: g["count"], reverse=True)
        return success_response(data={"groups": result, "total_groups": len(result)})

    async def get_error_trends(self, hours: int = 24):
        now = datetime.now(timezone.utc)
        buckets: dict[str, int] = {}

        for h in range(hours):
            t = now.replace(minute=0, second=0, microsecond=0)
            from datetime import timedelta
            t = t - timedelta(hours=h)
            key = t.strftime("%Y-%m-%dT%H:00")
            buckets[key] = 0

        for entry in self._error_buffer:
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
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="errors_trends")
            pass

        result = [{"hour": k, "count": v} for k, v in sorted(buckets.items())]
        return success_response(data={"trends": result, "hours": hours})

    async def export_errors(self, limit: int = 500):
        errors = list(reversed(self._error_buffer[-limit:]))
        return JSONResponse(
            content={"errors": errors, "total": len(self._error_buffer), "exported": len(errors)},
            headers={
                "Content-Disposition": f'attachment; filename="errors-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}.json"'
            },
        )

    async def clear_errors(self):
        self._error_buffer.clear()
        self._clear_disk()
        self._error_count_since_clear = 0
        self._dedup_map.clear()
        return success_response(data={"status": "ok", "cleared": True})

    async def unread_count(self):
        return success_response(data={"unread_count": self._error_count_since_clear})

    async def get_opencode_log(self):
        import os
        home = os.path.expanduser("~")
        entries: list[dict] = []

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
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="errors_opencode_log_cli")
            pass

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
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="errors_opencode_log_ui")
            pass

        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return success_response(data={"entries": entries[:200], "total": len(entries)})


# ── Singleton + backward-compat re-exports ──────────────────────────────────

_errors_instance = ErrorsRouter()
router = _errors_instance.router
_error_buffer = _errors_instance._error_buffer
_error_count_since_clear = _errors_instance._error_count_since_clear
_dedup_map = _errors_instance._dedup_map


def clear_errors():
    return _errors_instance.clear_errors()
