"""
Error logging router — accepts frontend JS errors for server-side monitoring.

Uses MogDB as the storage engine with automatic JSON sync.
Errors are stored in a capped MogDB collection and synced to JSON.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from domains.infrastructure.output_buffer import get_server_buffer
from domains.logging.base import LogTag
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, safe_audit_log, success_response

logger = logging.getLogger("slo.errors")

# ── Constants ─────────────────────────────────────────────────────────────

MAX_ERRORS = 500
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ERROR_DB_PATH = os.path.join(_REPO_ROOT, "data", "errors_mogdb")
_ERROR_SYNC_PATH = os.path.join(_REPO_ROOT, "data", "errors_json")


def _get_error_db():
    from mogdb import MogDB

    return MogDB(_ERROR_DB_PATH, sync_dir=_ERROR_SYNC_PATH)


# ── Pydantic models ──────────────────────────────────────────────────────


class ErrorEntry(BaseModel):
    message: str = Field(max_length=5000)
    source: str = Field(default="web", max_length=100)
    stack: str | None = Field(default=None, max_length=20000)
    url: str | None = Field(default=None, max_length=2000)
    line: int | None = None
    col: int | None = None
    timestamp: str | None = None
    metadata: dict | None = None


class ErrorBatch(BaseModel):
    errors: list[ErrorEntry] = Field(max_length=100)


class FrontendLogEntry(BaseModel):
    level: str = Field(default="info", max_length=20)
    logger: str = Field(default="web", max_length=100)
    message: str = Field(default="", max_length=5000)
    timestamp: float = 0.0
    context: dict | None = None
    exception: str | None = Field(default=None, max_length=20000)


class FrontendLogBatch(BaseModel):
    logs: list[FrontendLogEntry] = Field(max_length=100)


# ── Router class ─────────────────────────────────────────────────────────


class ErrorsRouter:
    """OOP router for error-logging endpoints."""

    _DEDUP_WINDOW_S = 10

    def __init__(self):
        self.router = APIRouter(prefix="/errors", tags=["errors"])
        self._errors_lock = threading.Lock()
        self._error_buffer: list[dict] = []
        self._error_count_since_clear = 0
        self._dedup_map: dict[str, float] = {}

        self._error_buffer = self._load_from_mogdb()
        self._error_count_since_clear = 0

        self._register_routes()

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
        self.router.add_api_route(
            "/stream", self.error_stream, methods=["GET"], response_model=None
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _load_from_mogdb(self) -> list[dict]:
        """Load recent errors from MogDB."""
        try:
            db = _get_error_db()
            col = db.collection("errors")
            docs = col.find(sort=[("_created", -1)], limit=MAX_ERRORS)
            return docs
        except Exception as e:
            logger.warning("failed to load errors from mogdb: %s", e)
            return []

    def _persist_to_mogdb(self, records: list[dict]):
        """Persist error records to MogDB."""
        try:
            db = _get_error_db()
            col = db.collection("errors")
            col.insert_many(records)
        except Exception as e:
            logger.warning("failed to persist errors to mogdb: %s", e)

    def _clear_mogdb(self):
        """Clear all errors from MogDB."""
        try:
            db = _get_error_db()
            col = db.collection("errors")
            col.delete_many({})
        except Exception as e:
            logger.warning("failed to clear mogdb errors: %s", e)

    def _fingerprint(self, message: str) -> str:
        normalized = re.sub(r"\d+", "N", message.lower())
        normalized = re.sub(r"[a-f0-9]{8,}", "ID", normalized)
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]

    # ── Route handlers ──────────────────────────────────────────────────

    async def log_errors(
        self,
        batch: ErrorBatch,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Log one or more client-side JavaScript errors for server-side monitoring."""
        try:
            now_ts = datetime.now(timezone.utc)
            now_iso = now_ts.isoformat()
            now_epoch = now_ts.timestamp()
            client_host = request.client.host if request.client else "unknown"

            logged = 0
            batch_fps: set = set()
            records_to_persist: list = []
            with self._errors_lock:
                for entry in batch.errors:
                    message = entry.message or ""
                    fp = self._fingerprint(message)

                    last_seen = self._dedup_map.get(fp)
                    if (
                        fp not in batch_fps
                        and last_seen is not None
                        and (now_epoch - last_seen) < self._DEDUP_WINDOW_S
                    ):
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

                    is_extension = any(
                        x in (entry.url or "")
                        for x in ("chrome-extension://", "moz-extension://", "extension://")
                    )
                    is_vague = entry.message in (
                        "error",
                        "Script error.",
                        "Non-Error promise rejection",
                    )
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
                await asyncio.to_thread(self._persist_to_mogdb, records_to_persist)

            return success_response(data={"status": "ok", "logged": logged})
        except Exception as e:
            classify_and_raise(e, source="errors.log")

    async def ingest_frontend_logs(
        self, batch: FrontendLogBatch, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Ingest frontend logs through the core logging pipeline."""
        try:
            user_id = (
                (auth_user or {}).get("id") or (auth_user or {}).get("username") or "anonymous"
            )

            for entry in batch.logs:
                context: dict = {}
                if entry.context:
                    context.update(entry.context)
                if entry.exception:
                    context["exception"] = entry.exception

                raw_tag = context.pop("tag", None)
                tag = raw_tag if raw_tag and raw_tag in LogTag._value2member_map_ else "WEB"
                context["user"] = user_id
                level = getattr(logging, entry.level.upper(), logging.INFO)
                logger_name = (
                    f"slo.web.{entry.logger}"
                    if entry.logger and not entry.logger.startswith("slo.")
                    else entry.logger or "slo.web"
                )

                _log = logging.getLogger(logger_name)
                _log.log(
                    level,
                    entry.message,
                    extra={"tag": tag, "context": context, "source": f"web.{entry.logger}"},
                )

            return success_response(data={"status": "ok", "ingested": len(batch.logs)})
        except Exception as e:
            classify_and_raise(e, source="errors.ingest")

    async def get_recent_errors(self, limit: int = 50, offset: int = 0) -> dict:
        """Get recent errors from the buffer."""
        try:
            with self._errors_lock:
                total = len(self._error_buffer)
                start = max(0, total - offset - limit)
                end = max(0, total - offset)
                errors = list(reversed(self._error_buffer[start:end]))
                unread_count = self._error_count_since_clear
            return success_response(
                data={
                    "errors": errors,
                    "unread_count": unread_count,
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                }
            )
        except Exception as e:
            classify_and_raise(e, source="errors.recent")

    async def get_grouped_errors(self) -> dict:
        """Get errors grouped by fingerprint."""
        try:
            with self._errors_lock:
                buffer_snapshot = list(self._error_buffer)
            groups: dict[str, dict] = {}
            for entry in reversed(buffer_snapshot):
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

            with self._errors_lock:
                buffer_snapshot = list(self._error_buffer)
            for entry in buffer_snapshot:
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
                db = _get_error_db()
                col = db.collection("errors")
                all_errors = col.find()
                for rec in all_errors:
                    ts = rec.get("timestamp", "")
                    if not ts:
                        continue
                    try:
                        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        key = t.strftime("%Y-%m-%dT%H:00")
                        if key in buckets:
                            buckets[key] += 1
                    except (ValueError, TypeError):
                        pass
            except Exception as e:
                logger.warning("failed to read trends from mogdb: %s", e)

            result = [{"hour": k, "count": v} for k, v in sorted(buckets.items())]
            return success_response(data={"trends": result, "hours": hours})
        except Exception as e:
            classify_and_raise(e, source="errors.trends")

    async def export_errors(self, limit: int = 500) -> dict:
        """Export errors as JSON."""
        try:
            with self._errors_lock:
                errors = list(reversed(self._error_buffer[-limit:]))
                total = len(self._error_buffer)
            return JSONResponse(
                content={"errors": errors, "total": total, "exported": len(errors)},
                headers={
                    "Content-Disposition": f'attachment; filename="errors-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}.json"'
                },
            )
        except Exception as e:
            classify_and_raise(e, source="errors.export")

    async def clear_errors(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Clear all errors."""
        try:
            with self._errors_lock:
                self._error_buffer.clear()
                self._error_count_since_clear = 0
                self._dedup_map.clear()
            await asyncio.to_thread(self._clear_mogdb)
            safe_audit_log("errors.clear", resource="all")
            return success_response(data={"status": "ok", "cleared": True})
        except Exception as e:
            classify_and_raise(e, source="errors.clear")

    async def unread_count(self) -> dict:
        """Get unread error count."""
        try:
            with self._errors_lock:
                count = self._error_count_since_clear
            return success_response(data={"unread_count": count})
        except Exception as e:
            classify_and_raise(e, source="errors.unread")

    async def error_stream(self, request: Request) -> StreamingResponse:
        """SSE endpoint pushing real-time error events to the frontend."""
        import time as _time

        from fastapi.responses import StreamingResponse

        buf = get_server_buffer()
        subscriber = buf.subscribe("error-stream")
        seen_seq = buf.seq

        async def generate():
            nonlocal seen_seq
            try:
                while True:
                    if await request.is_disconnected():
                        break

                    lines = await subscriber.async_read(timeout=1.0)

                    now = _time.time()
                    if not hasattr(generate, "_last_yield"):
                        generate._last_yield = now
                    if now - generate._last_yield >= 30.0:
                        yield ": heartbeat\n\n"
                        generate._last_yield = now

                    for line in lines:
                        if line.level in ("error", "critical"):
                            ctx = line.context or {}
                            corr_id = ctx.get("corr") or ctx.get("correlation_id") or ""
                            http_method = ctx.get("method", "")
                            http_path = ctx.get("path", "")
                            http_status = ctx.get("status")
                            dur_ms = ctx.get("dur_ms")

                            event = {
                                "stream": "errors",
                                "phase": "ERROR",
                                "status": "working",
                                "data": {
                                    "message": line.text,
                                    "level": line.level,
                                    "source": line.source,
                                    "tag": line.tag,
                                    "context": ctx,
                                    "ts": line.timestamp,
                                    "correlation_id": corr_id,
                                    "http_method": http_method,
                                    "http_path": http_path,
                                    "http_status": http_status,
                                    "duration_ms": dur_ms,
                                },
                                "meta": {"ts": _time.time()},
                            }
                            yield "data: " + json.dumps(event, default=str) + "\n\n"
                            generate._last_yield = _time.time()

                    with self._errors_lock:
                        current_count = len(self._error_buffer)

                    if current_count > seen_seq:
                        with self._errors_lock:
                            new_errors = list(self._error_buffer[seen_seq:current_count])
                        seen_seq = current_count
                        for err in new_errors:
                            event = {
                                "stream": "errors",
                                "phase": "CLIENT_ERROR",
                                "status": "working",
                                "data": {
                                    "id": err.get("id", ""),
                                    "message": err.get("message", ""),
                                    "source": err.get("source", ""),
                                    "stack": err.get("stack"),
                                    "url": err.get("url"),
                                    "line": err.get("line"),
                                    "col": err.get("col"),
                                    "fingerprint": err.get("fingerprint", ""),
                                    "count": err.get("count", 1),
                                    "timestamp": err.get("timestamp", ""),
                                    "metadata": err.get("metadata", {}),
                                },
                                "meta": {"ts": _time.time()},
                            }
                            yield "data: " + json.dumps(event, default=str) + "\n\n"
                            generate._last_yield = _time.time()

            except Exception as e:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "stream": "errors",
                            "phase": "ERROR",
                            "status": "error",
                            "data": {"error": str(e)},
                            "message": str(e),
                        }
                    )
                    + "\n\n"
                )
            finally:
                buf.unsubscribe("error-stream")

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def get_opencode_log(self) -> dict:
        """Get opencode CLI and UI error logs."""
        try:
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
                        entries.append(
                            {
                                "timestamp": rec.get("timestamp", ""),
                                "command": rec.get("command", ""),
                                "category": "cli",
                                "pattern": rec.get("pattern", ""),
                                "snippet": rec.get("snippet", ""),
                                "cwd": rec.get("cwd", ""),
                            }
                        )
            except Exception as e:
                logger.warning("failed to read cli log: %s", e)

            ui_log_path = os.path.join(home, ".opencode-ui-error-log.json")
            try:
                if os.path.exists(ui_log_path):

                    def _read_ui_log():
                        with open(ui_log_path) as f:
                            return json.load(f)

                    data = await asyncio.to_thread(_read_ui_log)
                    for rec in data:
                        entries.append(
                            {
                                "timestamp": rec.get("timestamp", ""),
                                "command": rec.get("command", ""),
                                "category": rec.get("category", "unknown"),
                                "pattern": rec.get("pattern", ""),
                                "snippet": rec.get("snippet", ""),
                                "cwd": rec.get("cwd", ""),
                            }
                        )
            except Exception as e:
                logger.warning("failed to read ui log: %s", e)

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


async def clear_errors() -> dict:
    """clear_errors — async backward-compat re-export."""
    return await _errors_instance.clear_errors()
