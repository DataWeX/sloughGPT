"""
Logger group — live error catch for the agent fix flow.

Polls the server error buffer and writes agent-ready bundles the
``error-fixer`` agent already reads:

- ``~/.opencode-autofix-log.json`` — backend / Python / build errors
- ``~/.opencode-ui-error-log.json`` — frontend / React errors

Each bundle entry carries ``{id, message, category, source, snippet,
file, timestamp, resolved, resolvedAt}`` so the fixer can start from the
``snippet`` without re-asking for context.

Usage:
    sloughgpt logger watch                # poll every 3s until Ctrl-C
    sloughgpt logger watch --once         # single poll (agent use)
    sloughgpt logger watch --interval 5   # poll every 5s
    sloughgpt logger status               # bundle counts + newest entry
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from core.framework import click
from core.helpers import api_get, output_json

from domain.logging import get_global
from domain.shared import utc_now_iso

log = get_global()

# Must match the filenames in .opencode/agents/error-fixer.md
AUTOFIX_FILE = ".opencode-autofix-log.json"
UI_FILE = ".opencode-ui-error-log.json"
MAX_ENTRIES = 200
DEDUP_WINDOW_S = 10

_PY_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')
_JS_PAREN_RE = re.compile(r"at [^(]*\((\S+?):(\d+)(?::\d+)?\)")
_JS_BARE_RE = re.compile(r"at (\S+?):(\d+)(?::\d+)?\s*$")


def _now_iso() -> str:
    return utc_now_iso()


def _normalize(message: str) -> str:
    normalized = re.sub(r"\d+", "N", message.lower())
    return re.sub(r"[a-f0-9]{8,}", "ID", normalized)


def fingerprint(record: dict) -> str:
    """Stable id for an error record — backend fingerprint when present."""
    fp = record.get("fingerprint")
    if fp:
        return str(fp)
    return hashlib.sha256(_normalize(record.get("message", "")).encode()).hexdigest()[:12]


def is_frontend(record: dict) -> bool:
    """Route to the UI bundle when the error smells like the browser."""
    source = str(record.get("source") or "")
    if source in ("window.onerror", "unhandledrejection"):
        return True
    url = str(record.get("url") or "")
    if url.startswith("http") or url.endswith((".tsx", ".ts", ".jsx", ".js")):
        return True
    stack = str(record.get("stack") or "")
    if ".tsx" in stack or "window.onerror" in stack:
        return True
    return False


def classify(record: dict) -> str:
    """Playbook category from the error-fixer's fix table."""
    text = f"{record.get('message', '')}\n{record.get('stack', '')}"
    if re.search(r"\bTS\d+\b|error TS", text):
        return "typescript"
    if "Traceback (most recent call last)" in text or re.search(r'\.py("|$)', text):
        return "python-error"
    if "ECONNREFUSED" in text or "Failed to fetch" in text:
        return "network"
    if " 500" in text or "status 500" in text:
        return "server-error"
    if "hydrat" in text.lower():
        return "hydration"
    if is_frontend(record):
        return "frontend"
    return "unknown"


def extract_file(record: dict) -> str | None:
    """Best-effort ``path:line`` for the fixer to open."""
    url = record.get("url")
    line = record.get("line")
    if url and line:
        return f"{url}:{line}"
    stack = str(record.get("stack") or "")
    m = _PY_FRAME_RE.search(stack)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    for raw_line in stack.splitlines():
        line = raw_line.strip()
        m = _JS_PAREN_RE.search(line) or _JS_BARE_RE.search(line)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
    return None


def build_entry(record: dict) -> dict:
    """Build an error-fixer bundle entry from a server error record."""
    message = record.get("message", "")
    stack = record.get("stack")
    url = record.get("url")
    snippet_parts = [message]
    if url:
        loc = url
        if record.get("line"):
            loc += f":{record['line']}"
        snippet_parts.append(f"at {loc}")
    if stack:
        snippet_parts.append(str(stack)[:2000])
    return {
        "id": fingerprint(record),
        "message": message,
        "category": classify(record),
        "source": record.get("source") or "web",
        "snippet": "\n".join(snippet_parts)[:3000],
        "file": extract_file(record),
        "timestamp": record.get("timestamp") or _now_iso(),
        "resolved": False,
        "resolvedAt": None,
    }


def bundle_path(kind: str, home: Path | None = None) -> Path:
    """Bundle file for ``kind`` (``autofix`` or ``ui``)."""
    base = home or Path.home()
    return base / (UI_FILE if kind == "ui" else AUTOFIX_FILE)


def load_entries(path: Path) -> list[dict]:
    """Load bundle entries — tolerant of missing / corrupt files."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def append_entry(path: Path, entry: dict, cap: int = MAX_ENTRIES) -> bool:
    """Append unless the fingerprint is already open. Returns True if added."""
    entries = load_entries(path)
    open_fps = {e.get("id") for e in entries if not e.get("resolved")}
    if entry["id"] in open_fps:
        return False
    entries.append(entry)
    del entries[: max(0, len(entries) - cap)]
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return True


def _fetch_recent(ctx, limit: int) -> list[dict]:
    r = api_get(ctx, f"/errors/recent?limit={limit}")
    if r.status_code != 200:
        log.error(f"logger: failed to fetch errors: {r.text[:120]}")
        return []
    return r.json().get("data", {}).get("errors", [])


def _catch_batch(ctx, limit: int, home: Path | None, seen: set[str]) -> int:
    """One poll: bundle every unseen error. Returns new-error count."""
    caught = 0
    for record in reversed(_fetch_recent(ctx, limit)):
        fp = fingerprint(record)
        if fp in seen:
            continue
        seen.add(fp)
        entry = build_entry(record)
        path = bundle_path("ui" if is_frontend(record) else "autofix", home)
        if append_entry(path, entry):
            caught += 1
            log.error(entry["message"][:120], fp=fp, bundle=path.name)
    return caught


def register(cli):
    """Register logger commands with the CLI group."""

    @cli.group(help="Live error catch — watch streams and write agent bundles")
    def logger():
        pass

    @logger.command("watch", help="Catch errors live into fixer bundles")
    @click.option("--interval", default=3, type=int, help="Poll interval in seconds")
    @click.option("--limit", "-n", default=50, type=int, help="Errors per poll")
    @click.option("--once", is_flag=True, default=False, help="Single poll then exit")
    @click.option("--dir", "home", default=None, type=str, help="Bundle dir (default: ~)")
    @click.pass_context
    def logger_watch(ctx, interval, limit, once, home):
        home_path = Path(home).expanduser() if home else None
        seen: set[str] = set()
        if once:
            caught = _catch_batch(ctx, limit, home_path, seen)
            log.info(f"logger: caught {caught} new error(s)")
            return
        log.info("logger: watching — Ctrl-C to stop")
        try:
            while True:
                _catch_batch(ctx, limit, home_path, seen)
                time.sleep(max(1, interval))
        except KeyboardInterrupt:
            log.info("logger: stopped")

    @logger.command("status", help="Bundle counts + newest entry")
    @click.option("--dir", "home", default=None, type=str, help="Bundle dir (default: ~)")
    @click.pass_context
    def logger_status(ctx, home):
        home_path = Path(home).expanduser() if home else None
        report = {}
        for kind in ("autofix", "ui"):
            entries = load_entries(bundle_path(kind, home_path))
            open_entries = [e for e in entries if not e.get("resolved")]
            newest = max((e.get("timestamp", "") for e in entries), default=None)
            report[kind] = {
                "total": len(entries),
                "open": len(open_entries),
                "newest": newest,
            }
        if output_json(ctx, report):
            return
        for kind, info in report.items():
            log.info(
                f"  {kind}: {info['open']} open / {info['total']} total (newest: {info['newest']})"
            )
