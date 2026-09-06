"""
Authentication and audit logging module.

Provides JWT auth middleware, API key validation, and structured audit logging.
All config is derived from ``ServerConfig``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone

from domains.infrastructure.errors import AppError
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from schemas.common import raise_error

from config import ServerConfig

logger = logging.getLogger("slo.auth")

_security = HTTPBearer(auto_error=False)


class JWTAuth:
    """Minimal JWT auth for API endpoints.

    Uses HMAC-SHA256 for signing. Intended for development/tooling
    auth; production deployments should integrate OAuth2/OIDC.
    """

    def __init__(self, config: ServerConfig | None = None):
        cfg = config or ServerConfig.from_env()
        self._secret = cfg.jwt_secret
        self._algorithm = cfg.jwt_algorithm
        self._expiration_hours = cfg.jwt_expiration_hours

    def create_token(self, user_id: str, extra_payload: dict | None = None) -> str:
        """Create a signed JWT token.

        Args:
            user_id: Subject identifier.
            extra_payload: Optional additional claims.

        Returns:
            Serialized JWT string.
        """
        import jwt as pyjwt

        payload = {
            "sub": user_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + self._expiration_hours * 3600,
        }
        if extra_payload:
            payload.update(extra_payload)
        return pyjwt.encode(payload, self._secret, algorithm=self._algorithm)

    def verify_token(self, token: str) -> dict:
        """Validate and decode a JWT token.

        Args:
            token: Raw JWT string.

        Returns:
            Decoded payload dict.

        Raises:
            HTTPException 401: If token is invalid or expired.
        """
        import jwt as pyjwt

        try:
            return pyjwt.decode(token, self._secret, algorithms=[self._algorithm])
        except pyjwt.ExpiredSignatureError:
            raise_error("Token expired", "E_AUTH_MISSING", status_code=401)
        except pyjwt.InvalidTokenError as e:
            raise_error(f"Invalid token: {e}", "E_AUTH_MISSING", status_code=401)

    def refresh_token(self, token: str) -> str | None:
        """Validate a token and issue a new one with a fresh expiry.

        Args:
            token: Raw JWT string to refresh.

        Returns:
            A new signed JWT for the same subject, or None if invalid/expired.

        Side effects:
            None.
        """
        try:
            payload = self.verify_token(token)
        except AppError:
            return None
        if not payload:
            return None
        return self.create_token(payload.get("sub", ""))

    async def require_user(
        self, credentials: HTTPAuthorizationCredentials | None = Depends(_security)
    ) -> dict:
        """FastAPI dependency — extracts and validates bearer token.

        Returns:
            Decoded token payload.

        Raises:
            HTTPException 401: If no token or invalid.
        """
        if credentials is None:
            raise_error("Missing Authorization header", "E_AUTH_MISSING", status_code=401)
        return self.verify_token(credentials.credentials)

    async def optional_user(
        self, credentials: HTTPAuthorizationCredentials | None = Depends(_security)
    ) -> dict | None:
        """FastAPI dependency — like require_user but returns None on missing token."""
        if credentials is None:
            return None
        try:
            return self.verify_token(credentials.credentials)
        except AppError:
            return None


class APIKeyAuth:
    """API key authentication using HMAC signature.

    Validates requests by comparing HMAC-SHA256(request_body + timestamp, shared_secret).
    """

    def __init__(self, api_key: str | None = None):
        self._key = api_key or ServerConfig.from_env().jwt_secret

    def validate_request(self, body: bytes, timestamp: str, signature: str) -> bool:
        """Validate a signed API request.

        Args:
            body: Raw request body.
            timestamp: ISO 8601 timestamp from header.
            signature: Expected HMAC signature.

        Returns:
            True if valid.
        """
        expected = hmac.new(
            self._key.encode(),
            body + timestamp.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def sign_request(self, body: bytes) -> tuple[str, str]:
        """Create timestamp + signature for a request.

        Returns:
            (timestamp, signature) tuple.
        """
        ts = datetime.now(timezone.utc).isoformat()
        sig = hmac.new(self._key.encode(), body + ts.encode(), hashlib.sha256).hexdigest()
        return ts, sig


class AuditLogger:
    """Structured audit log for security-relevant operations.

    Writes events to ``audit.log`` as JSON lines.
    """

    _TAIL_BYTES = 256 * 1024

    def __init__(self, log_path: str = "audit.log"):
        self._log_path = log_path
        self._handler = None
        self._logs: deque = deque(maxlen=1000)
        self._setup()

    @property
    def logs(self) -> list:
        """In-memory audit records, newest last.

        Returns:
            List of the most recent records as dicts.
        """
        return list(self._logs)

    def file_query(
        self,
        limit: int = 100,
        event_type: str | None = None,
        before: str | None = None,
    ) -> list:
        """Query persisted audit records from ``audit.log``, newest last.

        Reads the tail of the on-disk log so records written before the current
        process started remain queryable. Falls back to the in-memory ring
        buffer when the file is unavailable (e.g. handler setup failed).

        Args:
            limit: Max records to return. 0 returns all, negative mirrors
                ``logs[-limit:]`` slicing.
            event_type: Only return records with this event_type.
            before: ISO-8601 timestamp cursor; only records strictly older
                than it are returned.

        Returns:
            List of audit records as dicts, newest last.
        """
        try:
            with open(self._log_path, "rb") as fh:
                size = fh.seek(0, os.SEEK_END)
                start = max(0, size - self._TAIL_BYTES)
                fh.seek(start)
                tail = fh.read()
        except OSError:
            events = list(self._logs)
        else:
            text = tail.decode("utf-8", errors="replace")
            events = []
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    events.append(json.loads(line))
                except (ValueError, TypeError):
                    continue
            events.reverse()
            if not events:
                events = list(self._logs)
        if before is not None:
            events = [e for e in events if e.get("timestamp") and e["timestamp"] < before]
        if event_type:
            events = [e for e in events if e.get("event_type") == event_type]
        if limit == 0:
            return events
        if limit < 0:
            return events[-limit:]
        return events[:limit]

    def _setup(self):
        try:
            import logging.handlers

            handler = logging.handlers.RotatingFileHandler(
                self._log_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
            )
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._handler = handler
        except Exception:
            logger.debug("Audit log handler setup failed", exc_info=True)
            self._handler = None

    def log(
        self,
        event: str,
        user: str = "anonymous",
        resource: str = "",
        detail: str = "",
        extra: dict | None = None,
    ):
        """Record an audit event.

        Args:
            event: Event name (e.g. "model.load", "chat.send").
            user: User identifier.
            resource: Target resource identifier.
            detail: Human-readable description.
            extra: Additional structured data.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event,
            "user": user,
            "resource": resource,
            "detail": detail,
        }
        if extra:
            record["extra"] = extra
        self._logs.append(record)
        line = json.dumps(record, default=str)
        if self._handler:
            self._handler.emit(logging.LogRecord("audit", logging.INFO, "", 0, line, (), None))
        else:
            logger.info("AUDIT: %s", line, extra={"tag": "AUTH"})


# Singleton instances
_jwt_auth_instance: JWTAuth | None = None
_audit_logger_instance: AuditLogger | None = None


def get_jwt_auth() -> JWTAuth:
    global _jwt_auth_instance
    if _jwt_auth_instance is None:
        _jwt_auth_instance = JWTAuth()
    return _jwt_auth_instance


def get_audit_logger() -> AuditLogger:
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger()
    return _audit_logger_instance


def audit_user(auth_user) -> str:
    """Extract a stable actor identifier from an auth payload.

    Args:
        auth_user: Decoded token payload dict from ``require_auth_if_enabled``,
            or None when auth is disabled.

    Returns:
        Subject identifier if present, otherwise ``"anonymous"``.
    """
    if not auth_user:
        return "anonymous"
    sub = auth_user.get("sub") or auth_user.get("username") or auth_user.get("user_id")
    return str(sub) if sub else "anonymous"


async def require_auth_if_enabled(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> dict | None:
    """FastAPI dependency — enforces auth only when ``SLO_AUTH_REQUIRED=true``.

    When disabled, returns None (anonymous). When enabled, validates bearer token.

    Returns:
        Decoded token payload, or None if auth is disabled.
    """
    import os

    if os.environ.get("SLO_AUTH_REQUIRED", "false").lower() not in ("true", "1", "yes"):
        return None
    if credentials is None:
        raise_error(
            "Authorization required (set SLO_AUTH_REQUIRED=false to disable)",
            "E_AUTH_MISSING",
            status_code=401,
        )
    return get_jwt_auth().verify_token(credentials.credentials)
