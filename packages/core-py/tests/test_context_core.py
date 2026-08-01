"""Tests for context_core — multi-layer context management, memory, RAG, frames."""

import asyncio
from types import SimpleNamespace

import pytest

from domains.infrastructure.context_core import (
    ContextCore,
    ContextFrame,
    ContextLayer,
    get_context_core,
    reset_context_core,
)


# ── Dataclasses ─────────────────────────────────────────────────────────


class TestContextLayer:
    def test_defaults(self):
        layer = ContextLayer("session", "hi", 3, "s", "t")
        assert layer.priority == 1.0

    def test_full_construction(self):
        layer = ContextLayer("rag", "docs", 10, "vs", "t", priority=0.7)
        assert layer.layer_type == "rag"
        assert layer.priority == 0.7


class TestContextFrame:
    def test_to_prompt_sorts_by_priority_desc(self):
        frame = ContextFrame(
            id="f1",
            system_prompt="SYS",
            layers=[
                ContextLayer("session", "low", 1, "s", "t", priority=0.2),
                ContextLayer("system", "high", 1, "s", "t", priority=1.0),
            ],
            total_tokens=10,
            max_tokens=100,
            created_at="t",
        )
        out = frame.to_prompt()
        assert out.startswith("SYS")
        assert "[SYSTEM] high" in out
        assert "[SESSION] low" in out
        assert out.index("[SYSTEM]") < out.index("[SESSION]")

    def test_to_prompt_empty_layers(self):
        frame = ContextFrame("f", "SYS", [], 1, 10, "t")
        assert frame.to_prompt() == "SYS"


# ── Core construction & session ─────────────────────────────────────────


class TestContextCoreInit:
    def test_defaults(self):
        cc = ContextCore()
        assert cc.max_tokens == 2048
        assert cc.system_prompt == ContextCore.DEFAULT_SYSTEM
        assert cc.working_capacity == 7
        assert cc.memory_enabled is True
        assert cc.rag_enabled is True
        assert cc.session_messages == []
        assert cc.working_memory == []
        assert cc.frame_history == []

    def test_custom_max_tokens_and_flags(self):
        cc = ContextCore(max_tokens=128, memory_enabled=False, rag_enabled=False)
        assert cc.max_tokens == 128
        assert cc.memory_enabled is False
        assert cc.rag_enabled is False


class TestSession:
    def test_set_session_id_creates_episodic_slot(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        assert cc.session_id == "s1"
        assert cc.episodic_memory["s1"] == []

    def test_set_session_id_preserves_existing(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.episodic_memory["s1"].append({"x": 1})
        cc.set_session_id("s1")
        assert cc.episodic_memory["s1"] == [{"x": 1}]

    def test_add_message(self):
        cc = ContextCore()
        cc.add_message("user", "hello there")
        assert cc.session_messages[-1] == {"role": "user", "content": "hello there"}
        assert cc.working_memory[-1]["role"] == "user"
        assert len(cc.sensory_buffer) == 1

    def test_add_response(self):
        cc = ContextCore()
        cc.add_response("world", model="gpt2")
        assert cc.session_messages[-1] == {"role": "assistant", "content": "world"}
        assert cc.working_memory[-1]["model"] == "gpt2"

    def test_sensory_buffer_capped(self):
        cc = ContextCore()
        for i in range(120):
            cc._add_sensory(f"item {i}")
        assert len(cc.sensory_buffer) <= 100
        assert cc.sensory_buffer[-1]["data"] == "item 119"


class TestWorkingMemory:
    def test_capacity_evicts_and_consolidates(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.working_capacity = 2
        cc.add_message("user", "one")
        cc.add_message("user", "two")
        cc.add_message("user", "three")
        assert len(cc.working_memory) == 2
        assert cc.working_memory[0]["content"] == "two"
        episodes = cc.episodic_memory["s1"]
        assert any(e["content"]["content"] == "one" for e in episodes)

    def test_eviction_without_session_is_noop(self):
        cc = ContextCore()
        cc.working_capacity = 1
        cc.add_message("user", "a")
        cc.add_message("user", "b")
        assert len(cc.working_memory) == 1
        assert cc.episodic_memory == {}

    def test_capacity_from_memory_manager(self):
        mm = SimpleNamespace(working_capacity=3)
        cc = ContextCore(memory_manager=mm)
        for i in range(5):
            cc.add_message("user", str(i))
        assert len(cc.working_memory) == 3


# ── Semantic memory ─────────────────────────────────────────────────────


class TestSemanticMemory:
    def test_store_fact_new(self):
        cc = ContextCore()
        cc.store_fact("name", "Slough")
        assert cc.semantic_memory["name"]["value"] == "Slough"
        assert cc.semantic_memory["name"]["strength"] == 1.0

    def test_store_fact_increments_strength(self):
        cc = ContextCore()
        cc.store_fact("k", "v")
        cc.store_fact("k", "v")
        assert cc.semantic_memory["k"]["strength"] == 1.1

    def test_recall_fact(self):
        cc = ContextCore()
        cc.store_fact("k", 42)
        assert cc.recall_fact("k") == 42

    def test_recall_missing_returns_none(self):
        cc = ContextCore()
        assert cc.recall_fact("nope") is None

    def test_search_semantic_by_key(self):
        cc = ContextCore()
        cc.store_fact("user_preference_tone", "formal")
        cc.store_fact("unrelated", "x")
        results = cc.search_semantic("preference")
        assert len(results) == 1
        assert results[0]["key"] == "user_preference_tone"

    def test_search_semantic_by_value(self):
        cc = ContextCore()
        cc.store_fact("color", "blue")
        results = cc.search_semantic("blue")
        assert len(results) == 1

    def test_search_semantic_sorted_by_strength(self):
        cc = ContextCore()
        cc.store_fact("a", "shared term")
        cc.store_fact("b", "shared term")
        cc.store_fact("b", "shared term")  # strength 1.1
        results = cc.search_semantic("shared")
        assert results[0]["key"] == "b"

    def test_search_semantic_limit(self):
        cc = ContextCore()
        for i in range(10):
            cc.store_fact(f"common_{i}", "value")
        assert len(cc.search_semantic("common", limit=3)) == 3


# ── Episodic context ────────────────────────────────────────────────────


class TestEpisodicContext:
    def test_disabled_returns_empty(self):
        cc = ContextCore(memory_enabled=False)
        cc.set_session_id("s1")
        assert cc.get_episodic_context() == ""

    def test_no_session_returns_empty(self):
        cc = ContextCore()
        assert cc.get_episodic_context() == ""

    def test_empty_episodes_returns_empty(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        assert cc.get_episodic_context() == ""

    def test_recent_episodes(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.episodic_memory["s1"] = [
            {"content": {"role": "user", "content": "one"}, "timestamp": "t"},
            {"content": {"role": "assistant", "content": "two"}, "timestamp": "t"},
        ]
        out = cc.get_episodic_context()
        assert "[user]: one" in out
        assert "[assistant]: two" in out

    def test_query_scored_orders_results(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.episodic_memory["s1"] = [
            {"content": {"role": "user", "content": "tell me about cats"}, "timestamp": "t"},
            {"content": {"role": "user", "content": "unrelated note"}, "timestamp": "t"},
        ]
        out = cc.get_episodic_context("cats", limit=1)
        assert "cats" in out
        assert "unrelated" not in out

    def test_non_dict_content(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.episodic_memory["s1"] = [{"content": "plain string", "timestamp": "t"}]
        assert "plain string" in cc.get_episodic_context()


# ── RAG ─────────────────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, text, id="doc1", metadata=None, score=0.9):
        self.text = text
        self.id = id
        self.metadata = metadata or {}
        self.score = score


class _FakeStore:
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else []
        self.error = error
        self.queries = []

    async def query(self, vec, top_k=3):
        self.queries.append((vec, top_k))
        if self.error:
            raise self.error
        return self.results[:top_k]


class _EmptyKnowledge:
    def search(self, query, top_k=5):
        return []


def _empty_knowledge(monkeypatch):
    monkeypatch.setattr(
        "domains.learner.knowledge.get_knowledge_memory",
        lambda: _EmptyKnowledge(),
    )


class TestRag:
    def test_disabled_returns_empty(self):
        cc = ContextCore(rag_enabled=False)
        assert asyncio.run(cc.get_rag_context("q")) == ""

    def test_with_vector_store_returns_docs(self):
        cc = ContextCore()
        store = _FakeStore(results=[_FakeResult("alpha"), _FakeResult("beta")])
        cc.set_vector_store(store, embedding_fn=lambda q: [0.1, 0.2])
        out = asyncio.run(cc.get_rag_context("q"))
        assert "[Doc: doc1] alpha" in out
        assert "[Doc: doc1] beta" in out
        assert len(store.queries) == 1
        assert store.queries[0][1] == 3

    def test_empty_results_returns_empty(self):
        cc = ContextCore()
        cc.set_vector_store(_FakeStore(results=[]), embedding_fn=lambda q: [0.1])
        assert asyncio.run(cc.get_rag_context("q")) == ""

    def test_respects_rag_max_chars(self):
        cc = ContextCore()
        cc.set_rag_config(max_chars=10)
        store = _FakeStore(results=[_FakeResult("x" * 100)])
        cc.set_vector_store(store, embedding_fn=lambda q: [0.1])
        out = asyncio.run(cc.get_rag_context("q"))
        assert out == "[Doc: doc1] " + "x" * 10

    def test_store_error_falls_back_to_semantic(self):
        cc = ContextCore()
        cc.store_fact("color", "blue")
        store = _FakeStore(results=[], error=RuntimeError("boom"))
        cc.set_vector_store(store, embedding_fn=lambda q: [0.1])
        out = asyncio.run(cc.get_rag_context("blue"))
        assert "Related: color = blue" in out

    def test_no_store_falls_back_to_semantic(self, monkeypatch):
        cc = ContextCore()
        cc.store_fact("color", "blue")
        monkeypatch.setattr(cc, "_auto_ingest", lambda: None)
        _empty_knowledge(monkeypatch)
        out = asyncio.run(cc.get_rag_context("blue"))
        assert "Related: color = blue" in out

    def test_auto_ingest_triggered_once(self, monkeypatch):
        cc = ContextCore()
        calls = []
        monkeypatch.setattr(cc, "_auto_ingest", lambda: calls.append(1))
        _empty_knowledge(monkeypatch)
        asyncio.run(cc.get_rag_context("nothing matches"))
        asyncio.run(cc.get_rag_context("nothing matches"))
        assert len(calls) == 1

    def test_no_store_uses_knowledge_memory(self, monkeypatch):
        cc = ContextCore()
        fake_kmem = SimpleNamespace(
            search=lambda q, top_k: [{"topic": "t", "content": "c" * 50}]
        )
        monkeypatch.setattr(cc, "_auto_ingest", lambda: None)
        monkeypatch.setattr(
            "domains.learner.knowledge.get_knowledge_memory", lambda: fake_kmem
        )
        out = asyncio.run(cc.get_rag_context("q"))
        assert "[Knowledge: t]" in out


# ── Context frames ──────────────────────────────────────────────────────


class TestContextFrameBuild:
    def test_frame_includes_session_layer(self):
        cc = ContextCore()
        cc.add_message("user", "hello")
        frame = asyncio.run(cc.build_context_frame(query="hello"))
        types = [l.layer_type for l in frame.layers]
        assert "session" in types
        assert frame.system_prompt == cc.system_prompt

    def test_frame_includes_memory_layer(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.episodic_memory["s1"] = [
            {"content": {"role": "user", "content": "remember this"}, "timestamp": "t"}
        ]
        frame = asyncio.run(cc.build_context_frame(query="hello"))
        types = [l.layer_type for l in frame.layers]
        assert "memory" in types

    def test_frame_rag_layer_added(self):
        cc = ContextCore()
        cc.store_fact("color", "blue")
        cc._auto_ingest = lambda: None
        frame = asyncio.run(cc.build_context_frame(query="blue"))
        types = [l.layer_type for l in frame.layers]
        assert "rag" in types

    def test_memory_layer_respects_budget(self):
        cc = ContextCore(max_tokens=20)
        cc.set_session_id("s1")
        cc.episodic_memory["s1"] = [
            {"content": {"role": "user", "content": "x" * 200}, "timestamp": "t"}
        ]
        cc.add_message("user", "hi")
        frame = asyncio.run(cc.build_context_frame(query="q"))
        types = [l.layer_type for l in frame.layers]
        assert "session" in types
        assert "memory" not in types

    def test_include_memory_false(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.episodic_memory["s1"] = [{"content": "m", "timestamp": "t"}]
        frame = asyncio.run(cc.build_context_frame(include_memory=False))
        assert "memory" not in [l.layer_type for l in frame.layers]

    def test_include_rag_false(self):
        cc = ContextCore()
        cc.store_fact("color", "blue")
        cc._auto_ingest = lambda: None
        frame = asyncio.run(cc.build_context_frame(include_rag=False, query="blue"))
        assert "rag" not in [l.layer_type for l in frame.layers]

    def test_frame_history_capped(self):
        cc = ContextCore()
        for _ in range(60):
            asyncio.run(cc.build_context_frame())
        assert len(cc.frame_history) == 50

    def test_manager_system_extra_applied(self):
        personality = SimpleNamespace(apply=lambda sp: "[PERSONALITY TEST]")
        cc = ContextCore(personality_manager=personality)
        frame = asyncio.run(cc.build_context_frame())
        assert "[PERSONALITY TEST]" in frame.system_prompt

    def test_manager_working_capacity_from_task(self):
        task = SimpleNamespace(apply=lambda sp: "")
        memory = SimpleNamespace(working_capacity=4)
        cc = ContextCore(memory_manager=memory, task_manager=task)
        for i in range(6):
            cc.add_message("user", str(i))
        assert len(cc.working_memory) == 4

    def test_estimate_tokens(self):
        cc = ContextCore()
        assert cc._estimate_tokens("abcd") == 1
        assert cc._estimate_tokens("") == 1

    def test_to_prompt_priority_order(self, monkeypatch):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.add_message("user", "hi")
        cc.episodic_memory["s1"] = [{"content": "memo", "timestamp": "t"}]
        cc.store_fact("color", "blue")
        cc._auto_ingest = lambda: None
        _empty_knowledge(monkeypatch)
        frame = asyncio.run(cc.build_context_frame(query="blue"))
        prompt = frame.to_prompt()
        assert "[SESSION]" in prompt
        assert "[MEMORY]" in prompt
        assert "[RAG]" in prompt


# ── Inspector, export/import, reset ─────────────────────────────────────


class TestInspectorAndPersistence:
    def test_get_context_inspector(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.add_message("user", "hi")
        cc.store_fact("k", "v")
        info = cc.get_context_inspector()
        assert info["system_prompt"] == ContextCore.DEFAULT_SYSTEM
        assert info["session_messages"][-1]["content"] == "hi"
        assert info["semantic_keys"] == ["k"]
        assert info["last_frame"] is None

    def test_export_and_import_memory(self):
        cc = ContextCore()
        cc.store_fact("k", "v")
        cc.set_session_id("s1")
        cc.episodic_memory["s1"] = [{"x": 1}]
        cc._add_sensory("event")

        cc2 = ContextCore()
        cc2.import_memory(cc.export_memory())
        assert cc2.recall_fact("k") == "v"
        assert cc2.episodic_memory["s1"] == [{"x": 1}]
        assert cc2.sensory_buffer[-1]["data"] == "event"

    def test_import_memory_partial(self):
        cc = ContextCore()
        cc.import_memory({"semantic": {"a": {"value": 1, "strength": 1.0}}})
        assert cc.recall_fact("a") == 1
        assert cc.episodic_memory == {}

    def test_reset_session_keeps_memory(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.add_message("user", "hi")
        cc.store_fact("k", "v")
        cc.reset_session()
        assert cc.session_messages == []
        assert cc.working_memory == []
        assert cc.session_id is None
        assert cc.recall_fact("k") == "v"

    def test_reset_all(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.add_message("user", "hi")
        cc.store_fact("k", "v")
        cc.reset_all()
        assert cc.session_messages == []
        assert cc.episodic_memory == {}
        assert cc.semantic_memory == {}
        assert cc.frame_history == []

    def test_set_rag_config(self):
        cc = ContextCore()
        cc.set_rag_config(top_k=5, max_chars=100)
        assert cc.rag_top_k == 5
        assert cc.rag_max_chars == 100

    def test_set_managers(self):
        cc = ContextCore()
        personality = SimpleNamespace(apply=lambda sp: "[LATE INJECT]")
        cc.set_managers(personality=personality)
        frame = asyncio.run(cc.build_context_frame())
        assert "[LATE INJECT]" in frame.system_prompt

    def test_set_system_prompt_logs_sensory(self):
        cc = ContextCore()
        cc.set_system_prompt("NEW SYS")
        assert cc.system_prompt == "NEW SYS"
        assert "System prompt updated" in cc.sensory_buffer[-1]["data"]


# ── Global singleton ────────────────────────────────────────────────────


class TestSingleton:
    def test_get_context_core_singleton(self):
        reset_context_core()
        a = get_context_core()
        b = get_context_core()
        assert a is b
        reset_context_core()

    def test_reset_creates_new_instance(self):
        reset_context_core()
        a = get_context_core()
        reset_context_core()
        b = get_context_core()
        assert a is not b
        reset_context_core()

    def test_get_context_core_with_managers(self):
        reset_context_core()
        cc = get_context_core()
        assert cc._personality is not None
        assert cc._memory is not None
        reset_context_core()

    def test_get_context_core_auto_vector_store(self, monkeypatch):
        reset_context_core()
        monkeypatch.setenv("MAN_VECTOR_STORE", "in_memory")
        store = SimpleNamespace(query=lambda *a, **k: [], connect=lambda: None)
        monkeypatch.setattr(
            "domains.inference.vector_store.create_vector_store",
            lambda provider="in_memory", **kw: _async(store),
        )
        cc = get_context_core()
        assert cc._vector_store is not None
        monkeypatch.delenv("MAN_VECTOR_STORE", raising=False)
        reset_context_core()

    def test_get_context_core_auto_config_logs(self, monkeypatch, caplog):
        reset_context_core()
        import logging
        monkeypatch.setenv("MAN_VECTOR_STORE", "in_memory")
        store = SimpleNamespace(query=lambda *a, **k: [], connect=lambda: None)
        monkeypatch.setattr(
            "domains.inference.vector_store.create_vector_store",
            lambda provider="in_memory", **kw: _async(store),
        )
        with caplog.at_level(logging.INFO):
            get_context_core()
        assert any("auto-configured" in r.message for r in caplog.records)
        monkeypatch.delenv("MAN_VECTOR_STORE", raising=False)
        reset_context_core()


def _async(obj):
    async def _get():
        return obj
    return _get()
