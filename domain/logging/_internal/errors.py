"""Concise error logging — stdlib only, no external dependencies.

One-line errors on the console, full tracebacks in the file log:

- :func:`format_concise` — ``Type: msg at file:line in func [source]``,
  pointing at the last app frame (site-packages/venv frames skipped).
- :func:`log_error` — logs the one-liner at ERROR (console + file) and
  the full traceback at DEBUG (file only, since the console handler
  runs at INFO+ while the file handler runs at DEBUG).

Usage::

    import logging
    from domain.logging._internal.errors import log_error

    logger = logging.getLogger("slo.mymodule")
    try:
        ...
    except Exception as exc:
        log_error(logger, exc, source="GET /souls/current",
                  code="E_UNHANDLED", corr="abc1", status=500)
"""

from __future__ import annotations

import logging
import traceback
from types import TracebackType
from typing import Any

# Frame filenames containing any of these are third-party/runtime noise,
# never the place to point the user at.
_SKIP_FRAME_SUBSTRINGS = (
    "site-packages",
    ".venv/",
    "/uvicorn/",
    "/starlette/",
    "/fastapi/",
    "/anyio/",
    "asyncio/",
    "/pydantic/",
    "importlib/",
    "logging/__init__",
)


def _is_app_frame(filename: str | None) -> bool:
    """True when a traceback filename looks like our own code."""
    if not filename:
        return False
    return not any(skip in filename for skip in _SKIP_FRAME_SUBSTRINGS)


def _short_path(filename: str) -> str:
    """Repo-relative path for display (falls back to tail of path)."""
    if "sloughGPT/" in filename:
        return filename.split("sloughGPT/")[-1]
    return "/".join(filename.split("/")[-3:])


def find_app_frame(exc: BaseException) -> tuple[str, int | None, str | None]:
    """Return (path, lineno, func) of the deepest app frame in exc.

    Falls back to the deepest frame of any kind, or ("?", None, None)
    when there is no traceback (e.g. freshly constructed exception).
    """
    tb: TracebackType | None = exc.__traceback__
    fallback: tuple[str, int | None, str | None] = ("?", None, None)
    last_app: tuple[str, int | None, str | None] | None = None
    while tb is not None:
        frame = tb.tb_frame
        filename = frame.f_code.co_filename
        fallback = (_short_path(filename), tb.tb_lineno, frame.f_code.co_name)
        if _is_app_frame(filename):
            last_app = fallback
        tb = tb.tb_next
    return last_app if last_app is not None else fallback


def format_concise(exc: BaseException, source: str = "") -> str:
    """One-line summary: ``Type: msg at path:line in func [source]``."""
    path, lineno, func = find_app_frame(exc)
    location = path
    if lineno is not None:
        location += f":{lineno}"
    if func:
        location += f" in {func}"
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else "(no detail)"
    if len(msg) > 300:
        msg = msg[:297] + "..."
    line = f"{type(exc).__name__}: {msg} at {location}"
    if source:
        line += f" [{source}]"
    return line


def log_error(
    logger: logging.Logger,
    exc: BaseException,
    *,
    source: str = "",
    code: str = "E_UNHANDLED",
    corr: str = "-",
    status: int = 500,
    extra: dict[str, Any] | None = None,
) -> str:
    """Log an exception concisely.

    - ERROR: one line (console + file).
    - DEBUG: full traceback (file only — the console handler filters
      DEBUG out, the file handler keeps it).

    Returns the concise line (handy for tests and API details).
    """
    concise = format_concise(exc, source=source)
    context: dict[str, Any] = {"corr": corr, "code": code, "status": status}
    if extra:
        context.update(extra)
    logger.error(
        "%s",
        concise,
        extra={"tag": "REQ", "context": context},
    )
    full = "".join(traceback.TracebackException.from_exception(exc).format())
    logger.debug(
        "Full traceback %s corr=%s:\n%s",
        code,
        corr,
        full.rstrip(),
        extra={"tag": "REQ", "context": context},
    )
    return concise
