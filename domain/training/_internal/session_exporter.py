"""OpenCode session → chat-sft trajectory exporter (card 20261123_135823).

Reads sessions through the documented OpenCode HTTP API — ``opencode api``
handles service discovery and authentication, so this is the SDK/session
read path, never a storage scrape. Each user turn maps to the seed's
canonical trajectory shape (plan → tool_calls → observation → synthesis),
``filter_records`` applies the step-2 seed discipline, and the runner emits
chat-sft adapter JSONL (``{messages, source: "session", split}``).
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from hashlib import blake2b
from pathlib import Path

from domain.agents._internal.run_history import looks_like_refusal

# Curation caps (chars): session logs carry megabyte-scale reasoning and
# command output — training strings stay bounded, heads kept.
PLAN_MAX = 500
OBSERVATION_MAX = 2000
SYNTHESIS_MAX = 2000
HELDOUT_PERCENT = 16  # matches the seed's 63/12 train/heldout ratio


def api_get(path: str) -> dict:
    """One documented OpenCode API call (discovery/auth handled by the CLI)."""
    proc = subprocess.run(
        ["opencode", "api", "get", path],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"opencode api {path} failed: {proc.stderr.strip()[:300]}")
    return json.loads(proc.stdout)


def list_sessions(*, api=api_get) -> list[dict]:
    """Every session the server knows, following the cursor to exhaustion.

    The server's cursor is ``{previous, next}`` (a bare string is also
    accepted); forward pagination rides ``next``.
    """
    items: list[dict] = []
    next_cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        path = "/api/session" + (f"?cursor={next_cursor}" if next_cursor else "")
        page = api(path)
        items.extend(page.get("data", []))
        cursor = page.get("cursor")
        next_cursor = cursor.get("next") if isinstance(cursor, dict) else cursor
        if not next_cursor or next_cursor in seen_cursors:
            return items
        seen_cursors.add(str(next_cursor))


def export_session(session_id: str, *, api=api_get) -> list[dict]:
    """Full transcript in one export call (``{data: {info, messages}}``)."""
    payload = api(f"/api/experimental/session/{session_id}/export")
    return (payload.get("data") or {}).get("messages") or []


def is_repo_session(info: dict, repo_dirs) -> bool:
    """True for sessions run inside this repo's checkout family.

    ``repo_dirs`` is one root or a sequence (main checkout + every worktree —
    session directories anchor at the main checkout or a worktree path, never
    at a ``.wt-*`` copy).
    """
    directory = (info.get("location") or {}).get("directory") or ""
    if not directory:
        return False
    if isinstance(repo_dirs, (str, Path)):
        roots = [str(repo_dirs)]
    else:
        roots = [str(root) for root in repo_dirs]
    roots = [root.rstrip("/") for root in roots if root]
    return any(directory == root or directory.startswith(root + "/") for root in roots)


def repo_roots() -> list[str]:
    """Main checkout + every worktree of this repo (deduped, main first)."""
    roots: list[str] = []
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True
    )
    if common.returncode == 0 and common.stdout.strip():
        git_dir = Path(common.stdout.strip())
        if not git_dir.is_absolute():
            git_dir = Path.cwd() / git_dir
        roots.append(str(git_dir.resolve().parent))
    listing = subprocess.run(
        ["git", "worktree", "list", "--porcelain"], capture_output=True, text=True
    )
    if listing.returncode == 0:
        for line in listing.stdout.splitlines():
            if line.startswith("worktree "):
                roots.append(line.split(" ", 1)[1].strip())
    deduped: list[str] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def extract_trajectories(session_id: str, messages: list[dict]) -> list[dict]:
    """Every user turn → one candidate record (``filter_records`` judges it).

    Non-conversational message types (idle/system/synthetic/compaction/…)
    are ignored; turns with no visible prompt are not candidates.
    """
    records: list[dict] = []
    turn_idx = 0
    prompt: str | None = None
    body: list[dict] = []

    def flush() -> None:
        nonlocal prompt, body, turn_idx
        if prompt is None:
            return
        records.append(
            {
                "id": f"session-{session_id}-{turn_idx}",
                "source": "session",
                "messages": [{"role": "user", "content": _truncate(prompt, SYNTHESIS_MAX)}] + body,
            }
        )
        turn_idx += 1
        prompt, body = None, []

    for message in messages:
        kind = message.get("type")
        if kind == "user":
            flush()
            text = str(message.get("text") or "").strip()
            prompt = text or None  # empty prompt = no visible turn
            body = []
        elif kind == "assistant" and prompt is not None:
            parts = message.get("content") or []
            plan = _truncate(
                " ".join(p.get("text", "") for p in parts if p.get("type") == "reasoning").strip(),
                PLAN_MAX,
            )
            texts = _truncate(
                " ".join(p.get("text", "") for p in parts if p.get("type") == "text").strip(),
                SYNTHESIS_MAX,
            )
            tools = [p for p in parts if p.get("type") == "tool"]
            if tools:
                if texts:
                    body.append({"role": "assistant", "content": texts})
                for index, tool in enumerate(tools):
                    state = tool.get("state") or {}
                    if state.get("status") not in ("completed", "error"):
                        continue  # partial states are not observations
                    observation = "\n".join(
                        str(part.get("text", ""))
                        for part in (state.get("content") or [])
                        if isinstance(part, dict) and part.get("type") == "text"
                    ) or str(state.get("error") or "")
                    body.append(
                        {
                            "role": "assistant",
                            "content": plan if index == 0 else "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": str(tool.get("name") or ""),
                                        "arguments": json.dumps(
                                            state.get("input") or {}, ensure_ascii=False
                                        ),
                                    }
                                }
                            ],
                        }
                    )
                    if observation.strip():
                        body.append(
                            {
                                "role": "tool",
                                "content": _truncate(observation, OBSERVATION_MAX),
                            }
                        )
            elif texts:
                body.append({"role": "assistant", "content": texts})
    flush()
    return records


def _is_degenerate(text: str) -> bool:
    """Poison guard: tiny answers and character-collapse loops (step-2 discipline)."""
    stripped = text.strip()
    if len(stripped) < 8:
        return True
    counts = Counter(stripped.lower())
    return max(counts.values()) / len(stripped) >= 0.9


def filter_records(candidates: list[dict]) -> tuple[list[dict], dict]:
    """Step-2 seed discipline: tool_calls required, no refusal/degenerate/dupes."""
    stats = {
        "total": 0,
        "no_tool": 0,
        "no_synthesis": 0,
        "refusal": 0,
        "degenerate": 0,
        "duplicate": 0,
        "kept": 0,
    }
    seen: set[str] = set()
    kept: list[dict] = []
    for record in candidates:
        stats["total"] += 1
        messages = record["messages"]
        if not any(message.get("tool_calls") for message in messages):
            stats["no_tool"] += 1
            continue
        last = messages[-1]
        synthesis = str(last.get("content") or "") if last.get("role") == "assistant" else ""
        if not synthesis.strip():
            stats["no_synthesis"] += 1
            continue
        if looks_like_refusal(synthesis):
            stats["refusal"] += 1
            continue
        if _is_degenerate(synthesis):
            stats["degenerate"] += 1
            continue
        fingerprint = blake2b(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        if fingerprint in seen:
            stats["duplicate"] += 1
            continue
        seen.add(fingerprint)
        stats["kept"] += 1
        kept.append(record)
    return kept, stats


def assign_split(record_id: str) -> str:
    """Deterministic ~16% heldout split (content-independent, stable across runs)."""
    digest = blake2b(record_id.encode("utf-8"), digest_size=8).digest()
    return "heldout" if digest[0] % 100 < HELDOUT_PERCENT else "train"
