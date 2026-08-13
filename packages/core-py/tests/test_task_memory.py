"""Tests for task-backed memory (task_memory.py).

Covers the producer layer: ``memory.remember`` / ``memory.store`` queue
handlers, the durable task-backed store archive, submit helpers, and
handler registration.
"""

import asyncio
import json
import time

import pytest

from domains.infrastructure.task_queue import InProcessTaskQueue, Task
from domains.learner.knowledge import KnowledgeMemory
from domains.memory import (
    register_memory_handlers,
    submit_memory_consolidate,
    submit_memory_remember,
    submit_memory_store,
    unregister_memory_handlers,
)
from domains.memory import task_memory
from domains.memory.memory_config import MemoryConfig
from domains.memory.memory_provider import KnowledgeMemoryProvider
from domains.memory.memory_service import MemoryService


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Keep KnowledgeMemory persistence off the real data dir."""
    from domains.learner import knowledge as K
    monkeypatch.setattr(K, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(K, "FEED_STATE_PATH", tmp_path / "feeds.json")
    monkeypatch.setattr(K, "VISITED_PATH", tmp_path / "visited.json")
    monkeypatch.setattr(K, "ENTRIES_PATH", tmp_path / "entries.json")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Isolated MemoryService + config; wire them into task_memory module."""
    store = KnowledgeMemory(load_persisted=False)
    provider = KnowledgeMemoryProvider(store=store)
    config = MemoryConfig(enabled=True, min_chars=0, store_path=str(tmp_path))
    service = MemoryService(provider=provider, config=config)
    monkeypatch.setattr(task_memory, "get_memory_service", lambda: service)
    monkeypatch.setattr(task_memory.MemoryConfig, "get", lambda: config)
    return service


def _task(task_type, payload, task_id="t1"):
    return Task(id=task_id, name=task_type, task_type=task_type, payload=payload)


class TestStoreHandler:
    async def test_persists_fact_and_archives(self, isolated, tmp_path):
        task = _task("memory.store", {"content": "Jellyfish predate dinosaurs by 100m years", "topic": "biology", "source": "task"})
        result = await task_memory.store_handler(task)
        assert result == {"stored": True}
        assert any("Jellyfish" in r["content"] for r in isolated.retrieve("jellyfish predate", limit=5))
        archive = json.loads((tmp_path / "facts.jsonl").read_text())
        assert archive["task_type"] == "memory.store"
        assert archive["content"] == "Jellyfish predate dinosaurs by 100m years"
        assert archive["topic"] == "biology"
        assert archive["task_id"] == "t1"

    async def test_duplicate_fact_stored_false_no_archive(self, isolated, tmp_path):
        fact = "Redwoods can live over 2000 years"
        first = _task("memory.store", {"content": fact}, task_id="a")
        dup = _task("memory.store", {"content": fact}, task_id="b")
        assert await task_memory.store_handler(first) == {"stored": True}
        assert await task_memory.store_handler(dup) == {"stored": False}
        lines = (tmp_path / "facts.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    async def test_empty_content_stored_false_no_archive(self, isolated, tmp_path):
        task = _task("memory.store", {"content": "  "})
        assert await task_memory.store_handler(task) == {"stored": False}
        assert not (tmp_path / "facts.jsonl").exists()

    async def test_defaults_topic_and_source(self, isolated, tmp_path):
        task = _task("memory.store", {"content": "Venus spins backwards relative to most planets"})
        result = await task_memory.store_handler(task)
        assert result == {"stored": True}
        archive = json.loads((tmp_path / "facts.jsonl").read_text())
        assert archive["topic"] == "task"
        assert archive["source"] == "task"


class TestRememberHandler:
    async def test_mines_turn_and_archives(self, isolated, tmp_path):
        task = _task("memory.remember", {
            "user_message": "Explain how learning works.",
            "assistant_response": "Machine learning learns patterns from data. Gradient descent is the optimizer that minimizes the loss.",
        })
        result = await task_memory.remember_handler(task)
        assert result["stored"] is True
        assert any("Gradient descent" in r["content"] for r in isolated.retrieve("gradient descent optimizer", limit=5))
        archive = json.loads((tmp_path / "facts.jsonl").read_text())
        assert archive["task_type"] == "memory.remember"
        assert archive["user_message"].startswith("Explain how")

    async def test_short_turn_stored_false_no_archive(self, isolated, tmp_path):
        task = _task("memory.remember", {"user_message": "hi", "assistant_response": "yes"})
        result = await task_memory.remember_handler(task)
        assert result == {"stored": False}
        assert not (tmp_path / "facts.jsonl").exists()

    async def test_disabled_memory_no_archive(self, tmp_path, monkeypatch):
        store = KnowledgeMemory(load_persisted=False)
        config = MemoryConfig(enabled=False, min_chars=0, store_path=str(tmp_path))
        service = MemoryService(provider=KnowledgeMemoryProvider(store=store), config=config)
        monkeypatch.setattr(task_memory, "get_memory_service", lambda: service)
        monkeypatch.setattr(task_memory.MemoryConfig, "get", lambda: config)
        task = _task("memory.remember", {
            "user_message": "long enough question about preferences",
            "assistant_response": "The user likes the tea brand Yorkshire Gold above all others for breakfast.",
        })
        result = await task_memory.remember_handler(task)
        assert result == {"stored": False}
        assert not (tmp_path / "facts.jsonl").exists()


class TestRegistration:
    def test_register_and_unregister(self, monkeypatch):
        queue = InProcessTaskQueue(num_workers=1)
        monkeypatch.setattr("domains.infrastructure.task_queue.get_task_queue", lambda: queue)
        register_memory_handlers()
        assert "memory.remember" in queue._handlers
        assert "memory.store" in queue._handlers
        assert "memory.consolidate" in queue._handlers
        unregister_memory_handlers()
        assert "memory.remember" not in queue._handlers
        assert "memory.store" not in queue._handlers
        assert "memory.consolidate" not in queue._handlers


class TestSubmit:
    async def test_submit_remember_builds_payload(self, isolated):
        queue = InProcessTaskQueue(num_workers=1)
        task_id = await submit_memory_remember("a question", "a long enough answer", queue=queue)
        task = queue.get_task(task_id)
        assert task.task_type == "memory.remember"
        assert task.payload["user_message"] == "a question"

    async def test_submit_store_builds_payload(self, isolated):
        queue = InProcessTaskQueue(num_workers=1)
        task_id = await submit_memory_store("A fact about nebulae", topic="space", source="pipeline", queue=queue)
        task = queue.get_task(task_id)
        assert task.task_type == "memory.store"
        assert task.payload["content"] == "A fact about nebulae"
        assert task.payload["topic"] == "space"
        assert task.payload["source"] == "pipeline"

    async def test_submit_defaults_to_global_queue(self, isolated, monkeypatch):
        queue = InProcessTaskQueue(num_workers=1)
        monkeypatch.setattr("domains.infrastructure.task_queue.get_task_queue", lambda: queue)
        task_id = await submit_memory_remember("question", "answer text long enough to matter")
        assert queue.get_task(task_id) is not None

    async def test_submit_consolidate_builds_payload(self, isolated):
        queue = InProcessTaskQueue(num_workers=1)
        task_id = await submit_memory_consolidate(threshold=0.8, queue=queue)
        task = queue.get_task(task_id)
        assert task.task_type == "memory.consolidate"
        assert task.payload == {"threshold": 0.8}

    async def test_submit_consolidate_default_threshold_omitted(self, isolated):
        queue = InProcessTaskQueue(num_workers=1)
        task_id = await submit_memory_consolidate(queue=queue)
        assert queue.get_task(task_id).payload == {}


class TestConsolidateHandler:
    async def test_removes_near_duplicate_keeps_longest(self, isolated, tmp_path):
        isolated.store(
            "Machine learning learns patterns from data.", "ml", "task",
        )
        isolated.store(
            "Machine learning learns patterns from data very effectively.", "ml", "task",
        )
        result = await task_memory.consolidate_handler(
            _task("memory.consolidate", {"threshold": 0.8}, task_id="c1")
        )
        assert result["removed"] == 1
        assert result["kept"] == 1
        assert result["threshold"] == 0.8
        remaining = isolated.list_all(limit=10)
        assert len(remaining) == 1
        assert "very effectively" in remaining[0]["content"]
        archive = json.loads((tmp_path / "facts.jsonl").read_text())
        assert archive["task_type"] == "memory.consolidate"
        assert archive["removed"] == 1
        assert archive["kept"] == 1
        assert archive["threshold"] == 0.8
        assert archive["task_id"] == "c1"

    async def test_no_duplicates_nothing_removed_archives_zero(self, isolated, tmp_path):
        isolated.store("Machine learning learns patterns from data.", "ml", "task")
        isolated.store("The octopus has three hearts and blue blood.", "biology", "task")
        result = await task_memory.consolidate_handler(
            _task("memory.consolidate", {"threshold": 0.8})
        )
        assert result == {"removed": 0, "kept": 2, "threshold": 0.8}
        archive = json.loads((tmp_path / "facts.jsonl").read_text())
        assert archive["removed"] == 0
        assert archive["kept"] == 2

    async def test_default_threshold_from_config(self, isolated, tmp_path):
        isolated.store(
            "Machine learning learns patterns from data.", "ml", "task",
        )
        isolated.store(
            "Machine learning learns patterns from data very effectively.", "ml", "task",
        )
        result = await task_memory.consolidate_handler(
            _task("memory.consolidate", {})
        )
        assert result["threshold"] == MemoryConfig().consolidation_threshold


class TestQueueIntegration:
    async def test_full_queue_flow(self, isolated, tmp_path):
        queue = InProcessTaskQueue(num_workers=1)
        task_memory.register_memory_handlers(queue)
        try:
            await queue.start()
            task_id = await submit_memory_store(
                "Octopuses have three hearts and blue blood", topic="biology", queue=queue
            )
            deadline = loop_time() + 5.0
            while loop_time() < deadline:
                task = queue.get_task(task_id)
                if task and task.status.value == "completed":
                    break
                await asyncio.sleep(0.01)
            task = queue.get_task(task_id)
            assert task is not None and task.status.value == "completed"
            assert task.result == {"stored": True}
            assert any("Octopuses" in r["content"] for r in isolated.retrieve("octopus hearts", limit=5))
            archive = json.loads((tmp_path / "facts.jsonl").read_text())
            assert archive["content"].startswith("Octopuses")
        finally:
            await queue.stop()
            task_memory.unregister_memory_handlers(queue)

    async def test_unhandled_type_fails(self, isolated):
        queue = InProcessTaskQueue(num_workers=1)
        task_memory.register_memory_handlers(queue)
        try:
            await queue.start()
            task_id = await queue.enqueue(Task(name="memory.bogus", task_type="memory.bogus", payload={}))
            deadline = loop_time() + 5.0
            while loop_time() < deadline:
                task = queue.get_task(task_id)
                if task and task.status.value in ("failed", "completed", "cancelled"):
                    break
                await asyncio.sleep(0.01)
            task = queue.get_task(task_id)
            assert task is not None and task.status.value == "failed"
            assert "No handler registered" in (task.error or "")
        finally:
            await queue.stop()
            task_memory.unregister_memory_handlers()

    async def test_full_consolidate_flow(self, isolated):
        isolated.store("Machine learning learns patterns from data.", "ml", "task")
        isolated.store(
            "Machine learning learns patterns from data very effectively.", "ml", "task",
        )
        queue = InProcessTaskQueue(num_workers=1)
        task_memory.register_memory_handlers(queue)
        try:
            await queue.start()
            task_id = await submit_memory_consolidate(threshold=0.8, queue=queue)
            deadline = loop_time() + 5.0
            while loop_time() < deadline:
                task = queue.get_task(task_id)
                if task and task.status.value in ("completed", "failed"):
                    break
                await asyncio.sleep(0.01)
            task = queue.get_task(task_id)
            assert task is not None and task.status.value == "completed"
            assert task.result["removed"] == 1
            assert task.result["kept"] == 1
            assert len(isolated.list_all(limit=10)) == 1
        finally:
            await queue.stop()
            task_memory.unregister_memory_handlers()


class TestArchive:
    def test_list_archive_newest_first(self, isolated, tmp_path):
        task_memory._append_archive({"ts": 100.0, "task_type": "memory.store", "task_id": "a", "content": "one"})
        task_memory._append_archive({"ts": 200.0, "task_type": "memory.store", "task_id": "b", "content": "two"})
        records = task_memory.list_archive(limit=10)
        assert [r["task_id"] for r in records] == ["b", "a"]

    def test_list_archive_empty_when_missing(self, isolated):
        assert task_memory.list_archive(limit=5) == []

    def test_archive_stats_counts(self, isolated, tmp_path):
        task_memory._append_archive({"ts": 100.0, "task_type": "memory.store", "content": "one"})
        task_memory._append_archive({"ts": 200.0, "task_type": "memory.consolidate", "removed": 1, "kept": 2})
        stats = task_memory.archive_stats()
        assert stats["records"] == 2
        assert stats["task_types"] == {"memory.store": 1, "memory.consolidate": 1}
        assert stats["oldest_ts"] == 100.0
        assert stats["newest_ts"] == 200.0
        assert stats["bytes"] > 0

    def test_archive_stats_empty_when_missing(self, isolated):
        stats = task_memory.archive_stats()
        assert stats["records"] == 0
        assert stats["bytes"] == 0
        assert stats["task_types"] == {}

    def test_prune_removes_old_keeps_recent(self, isolated, tmp_path):
        now = time.time()
        task_memory._append_archive({"ts": now - 60 * 86400, "task_type": "memory.store", "content": "old"})
        task_memory._append_archive({"ts": now - 1 * 86400, "task_type": "memory.store", "content": "recent"})
        removed = task_memory.prune_archive(retain_days=30)
        assert removed == 1
        lines = (tmp_path / "facts.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert "recent" in lines[0]

    def test_prune_keeps_all_within_window(self, isolated, tmp_path):
        now = time.time()
        task_memory._append_archive({"ts": now - 5 * 86400, "task_type": "memory.store"})
        task_memory._append_archive({"ts": now, "task_type": "memory.store"})
        removed = task_memory.prune_archive(retain_days=30)
        assert removed == 0
        assert len((tmp_path / "facts.jsonl").read_text().strip().splitlines()) == 2

    def test_prune_zero_window_removes_all(self, isolated, tmp_path):
        task_memory._append_archive({"ts": time.time() - 1, "task_type": "memory.store"})
        removed = task_memory.prune_archive(retain_days=0)
        assert removed == 1
        assert (tmp_path / "facts.jsonl").read_text().strip() == ""

    def test_prune_no_file_returns_zero(self, isolated):
        assert task_memory.prune_archive(retain_days=30) == 0

    def test_prune_default_uses_config_retention(self, isolated, tmp_path):
        now = time.time()
        task_memory._append_archive({"ts": now - 60 * 86400, "task_type": "memory.store", "content": "old"})
        task_memory._append_archive({"ts": now - 1 * 86400, "task_type": "memory.store", "content": "recent"})
        removed = task_memory.prune_archive()
        assert removed == 1
        assert "recent" in (tmp_path / "facts.jsonl").read_text()

    def test_prune_skips_corrupt_line(self, isolated, tmp_path):
        now = time.time()
        (tmp_path / "facts.jsonl").write_text(
            json.dumps({"ts": now - 60 * 86400, "task_type": "memory.store", "content": "old"}) + "\n"
            + "not-json-line\n",
            encoding="utf-8",
        )
        removed = task_memory.prune_archive(retain_days=30)
        assert removed == 1
        assert (tmp_path / "facts.jsonl").read_text().strip() == ""


def loop_time():
    return asyncio.get_event_loop().time()
