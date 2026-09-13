"""
Shell audit logger — structured JSONL event log for every command.

Every command execution (interactive, programmatic, piped) is logged
with timestamp, command, args, exit code, and context.  This lets you
monitor what the shell does across applications.

Output: ``~/.config/sloughgpt/shell_audit.jsonl`` (rotated at 10MB).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_DEFAULT_LOG_DIR = Path.home() / ".config" / "sloughgpt"
_DEFAULT_LOG_FILE = "shell_audit.jsonl"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5


class ShellAuditLogger:
    """Structured audit log for shell command execution.

    Writes one JSON line per event to a rotating log file.  Events include:
      - shell.command   — every command dispatched
      - shell.eval      — every ``py`` expression evaluated
      - shell.error     — command that raised an exception
      - shell.unknown   — unrecognized command
      - shell.pipeline  — pipeline execution (one event per pipeline)
      - shell.background — background job spawn
      - shell.startup   — shell session started
      - shell.shutdown  — shell session ended
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        log_file: str = _DEFAULT_LOG_FILE,
    ) -> None:
        self._log_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / log_file
        self._handler: Optional[logging.handlers.RotatingFileHandler] = None
        self._session_id = f"{int(time.time() * 1000)}"
        self._cmd_count = 0
        self._setup()

    def _setup(self) -> None:
        try:
            self._handler = logging.handlers.RotatingFileHandler(
                str(self._log_path),
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
            )
            self._handler.setLevel(logging.INFO)
            self._handler.setFormatter(logging.Formatter("%(message)s"))
        except Exception:
            self._handler = None

    def _emit(self, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self._session_id,
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, default=str, ensure_ascii=False)
        if self._handler:
            self._handler.emit(
                logging.LogRecord("audit", logging.INFO, "", 0, line, (), None)
            )

    def command(
        self,
        line: str,
        cmd: str,
        args: str,
        exit_code: int,
        *,
        elapsed_ms: float | None = None,
        expanded: str | None = None,
        is_background: bool = False,
        is_pipeline: bool = False,
    ) -> None:
        """Log a command execution."""
        self._cmd_count += 1
        self._emit(
            "shell.command",
            line=line,
            cmd=cmd,
            args=args,
            exit_code=exit_code,
            cmd_num=self._cmd_count,
            elapsed_ms=round(elapsed_ms, 1) if elapsed_ms is not None else None,
            expanded=expanded,
            is_background=is_background,
            is_pipeline=is_pipeline,
        )

    def eval(self, expression: str, result: str, exit_code: int) -> None:
        """Log a ``py`` expression evaluation."""
        self._cmd_count += 1
        self._emit(
            "shell.eval",
            expression=expression,
            result_preview=result[:200],
            exit_code=exit_code,
            cmd_num=self._cmd_count,
        )

    def error(self, line: str, error: str) -> None:
        """Log a command that raised an exception."""
        self._emit("shell.error", line=line, error=error)

    def unknown(self, cmd: str) -> None:
        """Log an unrecognized command."""
        self._cmd_count += 1
        self._emit("shell.unknown", cmd=cmd, cmd_num=self._cmd_count)

    def background(self, line: str, bg_id: int) -> None:
        """Log a background job spawn."""
        self._emit("shell.background", line=line, bg_id=bg_id)

    def startup(self) -> None:
        """Log shell session start."""
        self._emit("shell.startup", pid=os.getpid())

    def shutdown(self) -> None:
        """Log shell session end."""
        self._emit("shell.shutdown", total_commands=self._cmd_count)

    @property
    def log_path(self) -> Path:
        return self._log_path


# Singleton
_audit: Optional[ShellAuditLogger] = None


def get_shell_audit_logger(**kwargs: Any) -> ShellAuditLogger:
    global _audit
    if _audit is None:
        _audit = ShellAuditLogger(**kwargs)
    return _audit
