"""Task-backed memory: producers write facts through the infrastructure task queue.

The chat loop calls ``MemoryService`` inline; any other producer (agents,
pipelines, the future persistent-task layer) can instead enqueue a
``memory.remember`` / ``memory.store`` task and get the queue's guarantees:
priority scheduling, retries, cancel/pause control, and EventBus observability.

Each successfully processed task also appends one JSONL record to the durable
task-backed store at ``MemoryConfig.store_path`` (default ``data/memory/``),
so task-mined facts have an inspectable home independent of the learner's
KnowledgeMemory index. Retrieval is unaffected - ``retrieve()`` keeps reading
the shared store; the archive is provenance, not a second index.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from domains.memory.consolidation import plan_consolidation
from domains.memory.memory_config import MemoryConfig
from domains.memory.memory_service import MemoryService, get_memory_service

logger = logging.getLogger("slo.memory_task")

TASK_REMEMBER = "memory.remember"
TASK_STORE = "memory.store"
TASK_CONSOLIDATE = "memory.consolidate"

_ARCHIVE_FILENAME = "facts.jsonl"


def _archive_path() -> Path:
    """Resolve the task-backed store file from ``MemoryConfig.store_path``."""
    return Path(MemoryConfig.get().store_path) / _ARCHIVE_FILENAME


def _append_archive(record: Dict[str, Any]) -> None:
    """Append one durable record to the task-backed store (fail-closed)."""
    try:
        path = _archive_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Task memory archive append failed: %s", e)


def _read_archive() -> List[Dict[str, Any]]:
    """Read every archive record in file order (oldest first).

    Corrupt or non-JSON lines are skipped so one bad append never breaks
    the whole audit trail.

    Args:
        none.

    Returns:
        List of decoded record dicts.

    Side effects:
        - none; read-only.
    """
    path = _archive_path()
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                logger.debug("Task memory archive line skipped (not JSON): %r", line[:80])
    return records


def list_archive(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Return the most recent task-backed archive records, newest first.

    Args:
        limit: max records to return (clamped to at least 1).

    Returns:
        Archive records ordered newest -> oldest.

    Side effects:
        - none; read-only.
    """
    records = _read_archive()
    return records[-(max(int(limit), 1)):][::-1]


def archive_stats() -> Dict[str, Any]:
    """
    Summarize the task-backed provenance archive.

    Args:
        none.

    Returns:
        dict with ``path``, ``records`` (count), ``bytes`` (file size),
        ``task_types`` (per-task-type record counts), and ``oldest_ts`` /
        ``newest_ts`` (epoch seconds) when any record carries a timestamp.

    Side effects:
        - none; read-only.
    """
    records = _read_archive()
    by_type: Dict[str, int] = {}
    for r in records:
        task_type = r.get("task_type") or "unknown"
        by_type[task_type] = by_type.get(task_type, 0) + 1
    ts = [r["ts"] for r in records if r.get("ts") is not None]
    path = _archive_path()
    return {
        "path": str(path),
        "records": len(records),
        "bytes": path.stat().st_size if path.exists() else 0,
        "task_types": by_type,
        "oldest_ts": min(ts) if ts else None,
        "newest_ts": max(ts) if ts else None,
    }


def prune_archive(retain_days: Optional[float] = None) -> int:
    """
    Delete archive records older than ``retain_days``, keeping the file valid.

    Rewrites ``facts.jsonl`` atomically (tmp file + replace) with only the
    records still inside the retention window. Records without a ``ts`` are
    treated as oldest and pruned. On any write failure the original file is
    left untouched and zero is returned (fail-closed).

    Args:
        retain_days: retention window in days; records with ``ts`` older than
            this are removed. ``0`` prunes everything. Defaults to
            ``MemoryConfig.archive_retention_days``.

    Returns:
        Number of records removed.

    Side effects:
        - rewrites the archive file when anything is pruned.
    """
    path = _archive_path()
    if not path.exists():
        return 0
    if retain_days is None:
        retain_days = MemoryConfig.get().archive_retention_days
    cutoff = time.time() - max(float(retain_days), 0.0) * 86400
    records = _read_archive()
    kept = [r for r in records if (r.get("ts") or 0) >= cutoff]
    removed = len(records) - len(kept)
    if removed == 0:
        return 0
    tmp = path.with_name(path.name + ".tmp")
    try:
        body = "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in kept
        )
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        logger.warning("Task memory archive prune failed, original intact: %s", e)
        return 0
    return removed


async def remember_handler(task) -> Dict[str, Any]:
    """
    TaskQueue handler for ``memory.remember``.

    Expects ``task.payload`` with ``user_message`` and ``assistant_response``.
    Distills the turn into facts exactly like the chat loop's post-gen path.

    Args:
        task: a Task with task_type ``memory.remember``.

    Returns:
        ``{"stored": bool}`` - True when at least one fact was persisted.

    Side effects:
        - writes facts into the shared memory store.
        - appends a provenance record to the task-backed store on success.
    """
    svc: MemoryService = get_memory_service()
    user_message = str(task.payload.get("user_message", "") or "")
    assistant_response = str(task.payload.get("assistant_response", "") or "")
    stored = await svc.remember_async(user_message, assistant_response)
    if stored:
        _append_archive({
            "ts": time.time(),
            "task_id": task.id,
            "task_type": TASK_REMEMBER,
            "stored": True,
            "user_message": user_message,
            "assistant_response": assistant_response,
        })
    return {"stored": stored}


async def store_handler(task) -> Dict[str, Any]:
    """
    TaskQueue handler for ``memory.store``.

    Expects ``task.payload`` with ``content`` and optional ``topic``/``source``.

    Args:
        task: a Task with task_type ``memory.store``.

    Returns:
        ``{"stored": bool}`` - True when the fact was newly persisted.

    Side effects:
        - writes the explicit fact into the shared memory store.
        - appends a provenance record to the task-backed store on success.
    """
    svc: MemoryService = get_memory_service()
    content = str(task.payload.get("content", "") or "")
    topic = str(task.payload.get("topic", "task") or "task")
    source = str(task.payload.get("source", "task") or "task")
    stored = svc.store(content, topic, source)
    if stored:
        _append_archive({
            "ts": time.time(),
            "task_id": task.id,
            "task_type": TASK_STORE,
            "stored": True,
            "content": content,
            "topic": topic,
            "source": source,
        })
    return {"stored": stored}


async def consolidate_handler(task) -> Dict[str, Any]:
    """
    TaskQueue handler for ``memory.consolidate``.

    Scans the shared memory store for near-duplicate facts (same topic,
    n-gram cosine similarity >= threshold) and removes the shorter copies,
    keeping the longest fact in each cluster.

    Args:
        task: a Task with task_type ``memory.consolidate``. Optional payload
            ``threshold`` (float) overrides
            ``MemoryConfig.consolidation_threshold``.

    Returns:
        ``{"removed": int, "kept": int, "threshold": float}`` - deleted
        duplicates and surviving facts, plus the effective threshold.

    Side effects:
        - deletes near-duplicate facts from the shared memory store.
        - appends a provenance record to the task-backed store.
    """
    svc: MemoryService = get_memory_service()
    threshold = float(
        task.payload.get("threshold") or MemoryConfig.get().consolidation_threshold
    )
    facts = svc.list_all(limit=5000)
    plan = plan_consolidation(facts, threshold=threshold)
    removed = svc.delete(plan["remove_ids"]) if plan["remove_ids"] else 0
    result = {"removed": removed, "kept": len(plan["keep_ids"]), "threshold": threshold}
    _append_archive({
        "ts": time.time(),
        "task_id": task.id,
        "task_type": TASK_CONSOLIDATE,
        "removed": removed,
        "kept": result["kept"],
        "threshold": threshold,
    })
    return result


def register_memory_handlers(queue=None) -> None:
    """
    Register memory task handlers with a task queue. Call once at startup.

    Args:
        queue: target queue; defaults to the global task queue.
    """
    from domains.infrastructure.task_queue import get_task_queue
    tq = queue or get_task_queue()
    tq.register_handler(TASK_REMEMBER, remember_handler)
    tq.register_handler(TASK_STORE, store_handler)
    tq.register_handler(TASK_CONSOLIDATE, consolidate_handler)
    logger.info("Memory handlers registered with task queue", extra={"tag": "INFRA"})


def unregister_memory_handlers(queue=None) -> None:
    """
    Unregister memory task handlers. Call at shutdown.

    Args:
        queue: target queue; defaults to the global task queue.
    """
    from domains.infrastructure.task_queue import get_task_queue
    tq = queue or get_task_queue()
    tq.unregister_handler(TASK_REMEMBER)
    tq.unregister_handler(TASK_STORE)
    tq.unregister_handler(TASK_CONSOLIDATE)


async def submit_memory_remember(
    user_message: str,
    assistant_response: str,
    queue=None,
    priority=None,
) -> str:
    """
    Enqueue a ``memory.remember`` task and return its task id.

    Args:
        user_message: the user's prompt/instruction text.
        assistant_response: the assistant's reply to mine facts from.
        queue: target task queue; defaults to the global queue.
        priority: task priority; defaults to Priority.NORMAL.

    Returns:
        The enqueued task's id.

    Side effects:
        - enqueues work on the queue; the handler persists facts later.
    """
    from domains.infrastructure.task_queue import Priority, Task, get_task_queue
    q = queue or get_task_queue()
    priority = priority if priority is not None else Priority.NORMAL
    task = Task(
        name="memory.remember",
        task_type=TASK_REMEMBER,
        payload={
            "user_message": user_message,
            "assistant_response": assistant_response,
        },
        priority=priority,
    )
    return await q.enqueue(task)


async def submit_memory_store(
    content: str,
    topic: str = "task",
    source: str = "task",
    queue=None,
    priority=None,
) -> str:
    """
    Enqueue a ``memory.store`` task and return its task id.

    Args:
        content: the fact text to persist.
        topic: knowledge topic label.
        source: provenance label.
        queue: target task queue; defaults to the global queue.
        priority: task priority; defaults to Priority.NORMAL.

    Returns:
        The enqueued task's id.

    Side effects:
        - enqueues work on the queue; the handler persists the fact later.
    """
    from domains.infrastructure.task_queue import Priority, Task, get_task_queue
    q = queue or get_task_queue()
    priority = priority if priority is not None else Priority.NORMAL
    task = Task(
        name="memory.store",
        task_type=TASK_STORE,
        payload={
            "content": content,
            "topic": topic,
            "source": source,
        },
        priority=priority,
    )
    return await q.enqueue(task)


async def submit_memory_consolidate(
    threshold: Optional[float] = None,
    queue=None,
    priority=None,
) -> str:
    """
    Enqueue a ``memory.consolidate`` task and return its task id.

    Args:
        threshold: min n-gram cosine for near-dup merge; defaults to
            ``MemoryConfig.consolidation_threshold`` when omitted.
        queue: target task queue; defaults to the global queue.
        priority: task priority; defaults to Priority.NORMAL.

    Returns:
        The enqueued task's id.

    Side effects:
        - enqueues work on the queue; the handler consolidates later.
    """
    from domains.infrastructure.task_queue import Priority, Task, get_task_queue
    q = queue or get_task_queue()
    priority = priority if priority is not None else Priority.NORMAL
    payload: Dict[str, Any] = {}
    if threshold is not None:
        payload["threshold"] = float(threshold)
    task = Task(
        name="memory.consolidate",
        task_type=TASK_CONSOLIDATE,
        payload=payload,
        priority=priority,
    )
    return await q.enqueue(task)
