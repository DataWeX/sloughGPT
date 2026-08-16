"""Memory commands - inspect, search, store, consolidate, archive, and clear the auto-memory layer.

Thin wrappers over ``domains.memory.memory_service`` (infrastructure before
endpoints): the chat loop writes facts automatically, and these commands give
operators visibility and manual control over that store.
"""
import sys
import time

from domains.logging import get_global

log = get_global()

from domains.memory.consolidation import plan_consolidation
from domains.memory.memory_config import MemoryConfig
from domains.memory.memory_service import get_memory_service
from domains.memory.task_memory import (
    TASK_CONSOLIDATE,
    TASK_REMEMBER,
    TASK_STORE,
)


def _service():
    """Return the process-wide MemoryService, or exit when import fails."""
    try:
        return get_memory_service()
    except Exception as e:  # pragma: no cover - import environment dependent
        log.error(f"Memory layer unavailable: {e}")
        sys.exit(2)


def cmd_memory_stats(args) -> None:
    """Print memory statistics (total facts, topics, enabled state).

    Args:
        args: SimpleNamespace (unused beyond parity).

    Side effects:
        - Prints the stats block.
    """
    svc = _service()
    stats = svc.stats() or {}
    log.header("Memory")
    log.status(
        "enabled", "on" if svc.enabled else "off",
        "ok" if svc.enabled else "warn",
    )
    log.key_value("Facts", str(stats.get("total_facts", 0)))
    topics = stats.get("topics")
    if isinstance(topics, (list, tuple)):
        log.key_value("Topics", ", ".join(str(t) for t in topics) if topics else "-")
    else:
        log.key_value("Topic buckets", str(topics or 0))


def cmd_memory_enable(args) -> None:
    """Enable the memory master switch at runtime.

    Args:
        args: SimpleNamespace with ``enabled`` (bool).

    Side effects:
        - updates the shared MemoryConfig singleton.
    """
    svc = _service()
    enabled = bool(getattr(args, "enabled", True))
    svc.set_enabled(enabled)
    log.success(f"Memory {'enabled' if enabled else 'disabled'}")


def cmd_memory_list(args) -> None:
    """List stored memory items, most recent first.

    Args:
        args: SimpleNamespace with ``limit``.

    Side effects:
        - Prints a fact table.
    """
    svc = _service()
    limit = int(getattr(args, "limit", 50))
    items = svc.list_all(limit=limit)
    if not items:
        log.info("No memory stored yet. Chat turns auto-save after enough text.")
        return
    log.header(f"Memory ({len(items)} shown)")
    rows = []
    for item in items:
        topic = item.get("topic") or ""
        source = item.get("source") or ""
        content = (item.get("content") or "").replace("\n", " ")[:90]
        rows.append([topic, source, content])
    log.table(["topic", "source", "content"], rows)


def cmd_memory_search(args) -> None:
    """Semantic-search stored memory for facts relevant to a query.

    Args:
        args: SimpleNamespace with ``query`` and ``limit``.

    Side effects:
        - Prints a ranked fact table.
    """
    svc = _service()
    query = getattr(args, "query", "")
    if not query:
        log.error("Query required: sloughgpt memory search <query>")
        sys.exit(2)
    limit = int(getattr(args, "limit", 5))
    results = svc.retrieve(query, limit=limit)
    if not results:
        log.info(f"No memory matches {query!r}")
        return
    log.header(f"Matches for {query!r} ({len(results)})")
    rows = []
    for r in results:
        content = (r.get("content") or "").replace("\n", " ")[:90]
        rows.append([f"{r.get('score', 0.0):.3f}", r.get("topic") or "", content])
    log.table(["score", "topic", "content"], rows)


def cmd_memory_store(args) -> None:
    """Persist one explicit fact with a topic and source label.

    Args:
        args: SimpleNamespace with ``content``, ``topic``, ``source``.

    Side effects:
        - Writes the fact into the underlying knowledge store.
    """
    svc = _service()
    content = getattr(args, "content", "")
    if not content:
        log.error("Content required: sloughgpt memory store <content>")
        sys.exit(2)
    topic = getattr(args, "topic", "manual")
    source = getattr(args, "source", "cli")
    if svc.store(content, topic, source):
        log.success(f"Stored fact under topic {topic!r}")
    else:
        log.warning("Not stored (disabled, duplicate, or already present)")


def cmd_memory_remember(args) -> None:
    """Manually persist one completed turn (user + assistant).

    Args:
        args: SimpleNamespace with ``user_message`` and ``assistant_response``.

    Side effects:
        - Extracts and stores facts from the turn.
    """
    svc = _service()
    user_message = getattr(args, "user_message", "")
    assistant_response = getattr(args, "assistant_response", "")
    if not user_message or not assistant_response:
        log.error("Both a user message and assistant response are required")
        sys.exit(2)
    if svc.remember(user_message, assistant_response):
        log.success("Turn stored as memory")
    else:
        log.warning("Turn not stored (disabled, too short, or nothing new)")


def cmd_memory_clear(args) -> None:
    """Remove every stored memory item.

    Args:
        args: SimpleNamespace with ``yes`` (skip confirmation).

    Side effects:
        - Wipes the underlying knowledge store.
    """
    svc = _service()
    if not getattr(args, "yes", False):
        import click
        if not click.confirm("Delete all stored memory?", abort=True):
            return
    removed = svc.clear()
    log.success(f"Cleared {removed} memory items")


def cmd_memory_consolidate(args) -> None:
    """Merge near-duplicate facts, keeping the longest in each cluster.

    Runs the same planning the ``memory.consolidate`` task uses: facts in the
    same topic whose n-gram cosine similarity is at or above the threshold are
    collapsed, deleting the shorter copies.

    Args:
        args: SimpleNamespace with optional ``threshold`` (float); defaults to
            ``MemoryConfig.consolidation_threshold``.

    Side effects:
        - Deletes near-duplicate facts from the underlying knowledge store.
    """
    svc = _service()
    threshold = getattr(args, "threshold", None)
    if threshold is None:
        threshold = MemoryConfig.get().consolidation_threshold
    threshold = float(threshold)
    facts = svc.list_all(limit=5000)
    if not facts:
        log.info("No memory to consolidate.")
        return
    plan = plan_consolidation(facts, threshold=threshold)
    removed = svc.delete(plan["remove_ids"]) if plan["remove_ids"] else 0
    kept = len(plan["keep_ids"])
    if removed:
        log.success(f"Consolidated {removed} duplicate fact(s), kept {kept}")
    else:
        log.info(f"No near-duplicates found at threshold {threshold:.3f} "
                     f"({kept} facts kept)")


def _archive_summary(record) -> str:
    """Short one-line description of an archive record for display."""
    task_type = record.get("task_type") or ""
    if task_type == TASK_REMEMBER:
        return (record.get("user_message") or "")[:60]
    if task_type == TASK_STORE:
        return (record.get("content") or "")[:60]
    if task_type == TASK_CONSOLIDATE:
        return (f"removed {record.get('removed', 0)}, "
                f"kept {record.get('kept', 0)}")
    return (record.get("content") or "")[:60]


def cmd_memory_archive(args) -> None:
    """Inspect or prune the task-backed provenance archive.

    Without ``prune_days`` this prints archive statistics and the most
    recent records. With ``prune_days`` it deletes records older than the
    retention window (after confirmation).

    Args:
        args: SimpleNamespace with ``limit`` (int, recent records to show)
            and optional ``prune_days`` (float retention window).

    Side effects:
        - With ``prune_days`` set, rewrites ``facts.jsonl`` keeping only
          records inside the retention window.
    """
    from domains.memory.task_memory import archive_stats, list_archive, prune_archive
    prune_days = getattr(args, "prune_days", None)
    if prune_days is not None:
        import click
        if not click.confirm(
            f"Delete archive records older than {float(prune_days):g} days?",
            abort=True,
        ):
            return
        removed = prune_archive(retain_days=float(prune_days))
        if removed:
            log.success(f"Pruned {removed} archive record(s)")
        else:
            log.info("Nothing to prune")
        return
    stats = archive_stats()
    log.header("Memory archive")
    log.key_value("Path", stats.get("path") or "-")
    log.key_value("Records", str(stats.get("records", 0)))
    log.key_value("Size", f"{stats.get('bytes', 0)} bytes")
    task_types = stats.get("task_types") or {}
    if task_types:
        log.key_value("Task types", ", ".join(
            f"{k} ({v})" for k, v in sorted(task_types.items())
        ))
    limit = int(getattr(args, "limit", 10))
    if limit > 0:
        records = list_archive(limit=limit)
        if records:
            log.header(f"Recent archive ({len(records)})")
            rows = []
            for r in records:
                ts = r.get("ts")
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"
                rows.append([when, r.get("task_type") or "-",
                             r.get("task_id") or "-", _archive_summary(r)])
            log.table(["when", "task", "task_id", "summary"], rows)
