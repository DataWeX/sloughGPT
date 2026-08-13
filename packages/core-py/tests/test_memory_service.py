"""Tests for the auto-memory layer (MemoryService + KnowledgeMemoryProvider)."""

import asyncio

import pytest

from pathlib import Path

from domains.memory.memory_config import MemoryConfig
from domains.memory.memory_provider import KnowledgeMemoryProvider
from domains.memory.memory_service import MemoryService, get_memory_service
from domains.learner.knowledge import KnowledgeMemory


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Keep persistence off the real data dir (repo-root anchored)."""
    from domains.learner import knowledge as K
    monkeypatch.setattr(K, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(K, "FEED_STATE_PATH", tmp_path / "feeds.json")
    monkeypatch.setattr(K, "VISITED_PATH", tmp_path / "visited.json")
    monkeypatch.setattr(K, "ENTRIES_PATH", tmp_path / "entries.json")


@pytest.fixture
def provider():
    store = KnowledgeMemory()
    store.clear_all()
    return KnowledgeMemoryProvider(store=store)


@pytest.fixture
def service(provider):
    return MemoryService(provider=provider, config=MemoryConfig())


class TestRemember:
    def test_returns_true_when_facts_stored(self, service):
        assert service.remember(
            "Tell me about machine learning",
            "Machine learning learns patterns from data. Gradient descent is the optimizer.",
        ) is True

    def test_returns_false_for_empty_turn(self, service):
        assert service.remember("", "") is False
        assert service.remember("hello", "") is False
        assert service.remember("", "world") is False

    def test_returns_false_for_short_turn_below_min_chars(self, service):
        assert service.remember("hi", "yes") is False

    def test_returns_false_when_disabled(self, provider):
        service = MemoryService(
            provider=provider,
            config=MemoryConfig(enabled=False, min_chars=0),
        )
        assert service.remember("question", "factual answer text") is False

    def test_learned_facts_are_retrievable(self, service):
        service.remember(
            "what is the capital",
            "The capital of France is Paris, the largest city in Europe, and a major center for art and culture.",
        )
        results = service.retrieve("Paris France capital", limit=5)
        assert any("Paris" in r["content"] for r in results)


class TestRememberAsync:
    async def test_offloads_to_thread_by_default(self, service, monkeypatch):
        calls = []
        real = asyncio.to_thread

        async def spy(func, *args, **kwargs):
            calls.append((func, args, kwargs))
            return await real(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", spy)
        result = await service.remember_async(
            "Tell me about machine learning",
            "Machine learning learns patterns from data. Gradient descent is the optimizer.",
        )
        assert result is True
        assert calls
        assert calls[0][0].__func__ is MemoryService.remember
        assert calls[0][0].__self__ is service

    async def test_inline_when_sync_remember(self, provider, monkeypatch):
        service = MemoryService(provider=provider, config=MemoryConfig(sync_remember=True))

        async def boom(func, *args, **kwargs):
            raise AssertionError("to_thread must not run when sync_remember=True")

        monkeypatch.setattr(asyncio, "to_thread", boom)
        result = await service.remember_async(
            "Tell me about machine learning",
            "Machine learning learns patterns from data. Gradient descent is the optimizer.",
        )
        assert result is True

    async def test_false_when_disabled(self, provider):
        service = MemoryService(
            provider=provider,
            config=MemoryConfig(enabled=False, min_chars=0),
        )
        assert await service.remember_async("question", "factual answer text") is False


class TestRememberFacts:
    def test_returns_stored_fact_texts(self, service):
        facts = service.remember_facts(
            "Tell me about photosynthesis",
            "Photosynthesis is the process plants use to convert light into chemical energy stored in glucose.",
        )
        assert len(facts) >= 1
        assert any("Photosynthesis" in f for f in facts)

    def test_empty_for_blank_turn(self, service):
        assert service.remember_facts("", "") == []
        assert service.remember_facts("hello", "") == []

    def test_empty_for_short_turn_below_min_chars(self, service):
        assert service.remember_facts("hi", "yes") == []

    def test_empty_when_disabled(self, provider):
        service = MemoryService(
            provider=provider,
            config=MemoryConfig(enabled=False, min_chars=0),
        )
        assert service.remember_facts("question", "factual answer text") == []

    def test_empty_on_duplicate_turn(self, service):
        turn = (
            "Tell me about machine learning",
            "Machine learning learns patterns from data. Gradient descent is the optimizer.",
        )
        assert len(service.remember_facts(*turn)) >= 1
        assert service.remember_facts(*turn) == []


class TestRememberFactsAsync:
    async def test_offloads_to_thread_by_default(self, service, monkeypatch):
        calls = []
        real = asyncio.to_thread

        async def spy(func, *args, **kwargs):
            calls.append((func, args, kwargs))
            return await real(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", spy)
        result = await service.remember_facts_async(
            "Tell me about machine learning",
            "Machine learning learns patterns from data. Gradient descent is the optimizer.",
        )
        assert result
        assert calls
        assert calls[0][0].__func__ is MemoryService.remember_facts
        assert calls[0][0].__self__ is service

    async def test_inline_when_sync_remember(self, provider, monkeypatch):
        service = MemoryService(provider=provider, config=MemoryConfig(sync_remember=True))

        async def boom(func, *args, **kwargs):
            raise AssertionError("to_thread must not run when sync_remember=True")

        monkeypatch.setattr(asyncio, "to_thread", boom)
        result = await service.remember_facts_async(
            "Tell me about machine learning",
            "Machine learning learns patterns from data. Gradient descent is the optimizer.",
        )
        assert result

    async def test_empty_when_disabled(self, provider):
        service = MemoryService(
            provider=provider,
            config=MemoryConfig(enabled=False, min_chars=0),
        )
        assert await service.remember_facts_async("question", "factual answer text") == []


class TestRetrieve:
    def test_empty_when_disabled(self, provider):
        service = MemoryService(
            provider=provider,
            config=MemoryConfig(enabled=False),
        )
        assert service.retrieve("anything", limit=5) == []

    def test_empty_for_blank_query(self, service):
        assert service.retrieve("") == []
        assert service.retrieve("   ") == []

    def test_respects_limit(self, service):
        for i in range(6):
            service.store(
                f"Machine learning topic number {i} about models and training data",
                "ml", "task",
            )
        results = service.retrieve("machine learning models", limit=3)
        assert len(results) <= 3

    def test_returns_fact_dicts(self, service):
        service.store("Cats have retractable claws for climbing", "animals", "task")
        results = service.retrieve("cat claws climbing")
        assert results
        result = results[0]
        assert "content" in result
        assert "topic" in result
        assert "score" in result


class TestStore:
    def test_store_new_fact(self, service):
        assert service.store("The sky appears blue by day", "science", "task") is True

    def test_store_duplicate_returns_false(self, service):
        service.store("Duplicate fact about gravity pulling objects", "science", "task")
        assert service.store("Duplicate fact about gravity pulling objects", "science", "task") is False

    def test_store_empty_content_returns_false(self, service):
        assert service.store("", "science", "task") is False

    def test_store_disabled_returns_false(self, provider):
        service = MemoryService(
            provider=provider,
            config=MemoryConfig(enabled=False),
        )
        assert service.store("some fact content here", "t", "task") is False


class TestStats:
    def test_stats_reports_stored_facts(self, service):
        service.store("First fact about space and planets", "space", "task")
        service.store("Second fact about ocean and tides", "ocean", "task")
        stats = service.stats()
        assert stats.get("total_facts", 0) >= 2


class TestList:
    def test_list_all_returns_stored_items(self, service):
        service.store("First fact about mountain ranges", "geo", "task")
        service.store("Second fact about river deltas", "geo", "task")
        items = service.list_all(limit=10)
        assert len(items) == 2
        contents = [i["content"] for i in items]
        assert "First fact about mountain ranges" in contents
        assert "Second fact about river deltas" in contents

    def test_list_respects_limit(self, service):
        for i in range(5):
            service.store(f"Fact number {i} about lakes and ponds", "geo", "task")
        assert len(service.list_all(limit=2)) == 2

    def test_list_empty_when_disabled(self, provider):
        service = MemoryService(
            provider=provider,
            config=MemoryConfig(enabled=False),
        )
        assert service.list_all() == []


class TestClear:
    def test_clear_removes_all_items(self, service):
        service.store("A fact about glaciers and ice sheets", "geo", "task")
        service.store("Another fact about deserts and dunes", "geo", "task")
        removed = service.clear()
        assert removed >= 2
        assert service.list_all() == []
        assert service.stats().get("total_facts", 0) == 0

    def test_clear_empty_store_returns_zero(self, service):
        assert service.clear() == 0

    def test_clear_disabled_returns_zero(self, provider):
        service = MemoryService(
            provider=provider,
            config=MemoryConfig(enabled=False),
        )
        assert service.clear() == 0


class TestDelete:
    def test_delete_removes_selected_items(self, service):
        service.store("First fact about volcanoes and magma", "geo", "task")
        service.store("Second fact about coral reefs and fish", "geo", "task")
        items = service.list_all(limit=10)
        assert len(items) == 2
        target = next(i for i in items if "First fact" in i["content"])
        removed = service.delete([target["id"]])
        assert removed == 1
        remaining = [i["content"] for i in service.list_all(limit=10)]
        assert "First fact about volcanoes and magma" not in remaining
        assert "Second fact about coral reefs and fish" in remaining

    def test_delete_unknown_id_removes_nothing(self, service):
        service.store("A fact about rainforests and canopies", "nature", "task")
        assert service.delete(["fact_999_deadbeef"]) == 0
        assert len(service.list_all(limit=10)) == 1

    def test_delete_empty_list_returns_zero(self, service):
        service.store("A fact about savannas and grasslands", "nature", "task")
        assert service.delete([]) == 0
        assert len(service.list_all(limit=10)) == 1

    def test_delete_disabled_returns_zero(self, provider):
        service = MemoryService(
            provider=provider,
            config=MemoryConfig(enabled=False),
        )
        assert service.delete(["fact_1_anything"]) == 0


class TestConsolidation:
    """Integration of the planner + delete path the task handler runs."""

    def test_consolidate_removes_near_duplicate_keeps_longest(self, service):
        service.store(
            "Machine learning learns patterns from data.", "ml", "task",
        )
        service.store(
            "Machine learning learns patterns from data very effectively.",
            "ml", "task",
        )
        from domains.memory.consolidation import plan_consolidation

        plan = plan_consolidation(service.list_all(limit=100), threshold=0.80)
        assert plan["removed_count"] == 1
        removed = service.delete(plan["remove_ids"])
        assert removed == 1
        remaining = service.list_all(limit=10)
        assert len(remaining) == 1
        assert "very effectively" in remaining[0]["content"]

    def test_consolidate_distinct_facts_untouched(self, service):
        service.store("Machine learning learns patterns from data.", "ml", "task")
        service.store("The octopus has three hearts and blue blood.", "biology", "task")
        from domains.memory.consolidation import plan_consolidation

        plan = plan_consolidation(service.list_all(limit=100), threshold=0.80)
        assert plan["removed_count"] == 0
        assert service.delete(plan["remove_ids"]) == 0
        assert len(service.list_all(limit=10)) == 2


class TestProvider:
    def test_store_turn_empty_inputs_false(self, provider):
        assert provider.store_turn("", "response") is False
        assert provider.store_turn("message", "") is False

    def test_store_turn_persists_facts(self, provider):
        assert provider.store_turn(
            "explain photosynthesis",
            "Photosynthesis is the process plants use to convert light into chemical energy stored in glucose.",
        ) is True
        assert provider.stats().get("total_facts", 0) >= 1

    def test_retrieve_returns_results(self, provider):
        provider.store("Bees pollinate flowering plants", "nature", "task")
        assert provider.retrieve("bee pollination flowers", limit=5)


class TestGetMemoryService:
    def test_returns_service_singleton(self):
        service = get_memory_service()
        assert isinstance(service, MemoryService)
        assert get_memory_service() is service


class TestSetEnabled:
    def test_toggle_turns_enabled_off_and_on(self, service):
        service.set_enabled(False)
        assert service.enabled is False
        service.set_enabled(True)
        assert service.enabled is True

    def test_disabled_service_skips_remember(self, service):
        turn = ("Tell me about machine learning", "Machine learning learns patterns from data. Gradient descent is the optimizer.")
        service.set_enabled(False)
        assert service.remember(*turn) is False
        service.set_enabled(True)
        assert service.remember(*turn) is True

    def test_disabled_service_skips_store_and_clear(self, provider, service):
        service.set_enabled(False)
        assert service.store("orphan fact content", "test", "api") is False
        assert service.clear() == 0
        assert service.list_all() == []

    def test_config_singleton_state_is_shared(self, provider):
        config = MemoryConfig.get()
        original = config.enabled
        try:
            service = MemoryService(provider=provider, config=config)
            service.set_enabled(False)
            assert MemoryConfig.get().enabled is False
            service.set_enabled(True)
            assert MemoryConfig.get().enabled is True
        finally:
            config.set_enabled(original)


class TestChatWiring:
    """Contract test: the chat post-gen path must call memory.remember().

    Source-level check because the router module requires fastapi (not present
    in every environment). Guards against the wiring being silently removed.
    """

    _ROUTER = Path(__file__).resolve().parents[3] / "apps" / "api" / "server" / "routers" / "inference.py"

    def test_router_imports_memory_service(self):
        src = self._ROUTER.read_text()
        assert "from domains.memory.memory_service import get_memory_service" in src

    def test_router_invokes_remember_facts_in_post_gen(self):
        src = self._ROUTER.read_text()
        assert "get_memory_service().remember_facts_async" in src

    def test_router_emits_memory_event_with_stored_fact(self):
        src = self._ROUTER.read_text()
        assert '"MEMORY"' in src
        assert 'data={"stored": True, "fact": _memory_fact, "facts": _memory_facts}' in src

    def test_router_context_gate_uses_total_facts(self):
        src = self._ROUTER.read_text()
        assert 'get_memory_service().stats().get("total_facts", 0)' in src
        assert "total_items" not in src.split("skip_context = True")[0]

    def test_router_injects_frame_layers_as_knowledge(self):
        src = self._ROUTER.read_text()
        assert "frame_context" in src
        assert 'layer.layer_type in ("memory", "rag")' in src
        assert "knowledge_retrieved + frame_context + (req.knowledge or [])" in src
