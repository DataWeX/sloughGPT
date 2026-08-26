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

import asyncio
import json
import logging
import hashlib
import uuid
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
import re

from infrastructure.auth import require_auth_if_enabled
from schemas.common import success_response, safe_audit_log, classify_and_raise

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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

    def _persist_batch(self, records: list[dict]):
        try:
            self._ensure_dir()
            with open(_ERROR_LOG_FILE, "a") as f:
                for record in records:
                    f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="errors_persist_batch")

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
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]

    # ── Route handlers ──────────────────────────────────────────────────

    async def log_errors(self, batch: ErrorBatch, request: Request, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Log one or more client-side JavaScript errors for server-side monitoring."""
        try:
            now_ts = datetime.now(timezone.utc)
            now_iso = now_ts.isoformat()
            now_epoch = now_ts.timestamp()
            client_host = request.client.host if request.client else "unknown"

            logged = 0
            batch_fps: set = set()
            records_to_persist: list = []
            for entry in batch.errors:
                message = entry.message or ""
                fp = self._fingerprint(message)

                last_seen = self._dedup_map.get(fp)
                if (fp not in batch_fps
                        and last_seen is not None
                        and (now_epoch - last_seen) < self._DEDUP_WINDOW_S):
                    for rec in reversed(self._error_buffer):
                        if rec.get("fingerprint") == fp:
                            rec["count"] = rec.get("count", 1) + 1
                            rec["timestamp"] = entry.timestamp or now_iso
                            break
                    continue

                batch_fps.add(fp)
                self._dedup_map[fp] = now_epoch
                if len(self._dedup_map) > 500:
                    cutoff = now_epoch - self._DEDUP_WINDOW_S * 10
                    self._dedup_map = {k: v for k, v in self._dedup_map.items() if v > cutoff}

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
                records_to_persist.append(error_record)

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

            if records_to_persist:
                await asyncio.to_thread(self._persist_batch, records_to_persist)

            return success_response(data={"status": "ok", "logged": logged})
        except Exception as e:
            classify_and_raise(e, source="errors.log")

    async def ingest_frontend_logs(self, batch: FrontendLogBatch, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Ingest frontend logs into the server buffer."""
        try:
            from domains.infrastructure.output_buffer import get_server_buffer

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
        except Exception as e:
            classify_and_raise(e, source="errors.ingest")

    async def get_recent_errors(self, limit: int = 50, offset: int = 0) -> dict:
        """Get recent errors from the buffer."""
        try:
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
        except Exception as e:
            classify_and_raise(e, source="errors.recent")

    async def get_grouped_errors(self) -> dict:
        """Get errors grouped by fingerprint."""
        try:
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
        except Exception as e:
            classify_and_raise(e, source="errors.grouped")

    async def get_error_trends(self, hours: int = 24) -> dict:
        """Get error counts per hour for the last N hours."""
        try:
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
                def _read_trends():
                    if _ERROR_LOG_FILE.exists():
                        return _ERROR_LOG_FILE.read_text().splitlines()
                    return []
                lines = await asyncio.to_thread(_read_trends)
                for line in lines:
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
        except Exception as e:
            classify_and_raise(e, source="errors.trends")

    async def export_errors(self, limit: int = 500) -> dict:
        """Export errors as JSON."""
        try:
            errors = list(reversed(self._error_buffer[-limit:]))
            return JSONResponse(
                content={"errors": errors, "total": len(self._error_buffer), "exported": len(errors)},
                headers={
                    "Content-Disposition": f'attachment; filename="errors-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}.json"'
                },
            )
        except Exception as e:
            classify_and_raise(e, source="errors.export")

    async def clear_errors(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Clear all errors."""
        try:
            self._error_buffer.clear()
            await asyncio.to_thread(self._clear_disk)
            self._error_count_since_clear = 0
            self._dedup_map.clear()
            safe_audit_log("errors.clear", resource="all")
            return success_response(data={"status": "ok", "cleared": True})
        except Exception as e:
            classify_and_raise(e, source="errors.clear")

    async def unread_count(self) -> dict:
        """Get unread error count."""
        try:
            return success_response(data={"unread_count": self._error_count_since_clear})
        except Exception as e:
            classify_and_raise(e, source="errors.unread")

    async def get_opencode_log(self) -> dict:
        """Get opencode CLI and UI error logs."""
        try:
            import os
            home = os.path.expanduser("~")
            entries: list[dict] = []

            cli_log_path = os.path.join(home, ".opencode-error-log.json")
            try:
                if os.path.exists(cli_log_path):
                    def _read_cli_log():
                        with open(cli_log_path) as f:
                            return json.load(f)
                    data = await asyncio.to_thread(_read_cli_log)
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
                    def _read_ui_log():
                        with open(ui_log_path) as f:
                            return json.load(f)
                    data = await asyncio.to_thread(_read_ui_log)
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
        except Exception as e:
            classify_and_raise(e, source="errors.opencode_log")


# ── Singleton + backward-compat re-exports ──────────────────────────────────

_errors_instance = ErrorsRouter()
router = _errors_instance.router
_error_buffer = _errors_instance._error_buffer
_error_count_since_clear = _errors_instance._error_count_since_clear
_dedup_map = _errors_instance._dedup_map


def clear_errors() -> dict:
    """clear_errors."""
    return _errors_instance.clear_errors()
