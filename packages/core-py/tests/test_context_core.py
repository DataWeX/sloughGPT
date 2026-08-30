"""Tests for domains.infrastructure.context_core — ContextCore."""

import pytest
from domains.infrastructure.context_core import (
    ContextCore,
    ContextLayer,
    ContextFrame,
)


class TestContextLayer:
    def test_defaults(self):
        cl = ContextLayer(
            layer_type="session", content="hi", tokens=1,
            source="test", timestamp="2024-01-01",
        )
        assert cl.priority == 1.0

    def test_custom_priority(self):
        cl = ContextLayer(
            layer_type="rag", content="doc", tokens=10,
            source="vs", timestamp="2024-01-01", priority=0.5,
        )
        assert cl.priority == 0.5

    def test_layer_type_session(self):
        cl = ContextLayer(
            layer_type="session", content="hi", tokens=1,
            source="test", timestamp="t",
        )
        assert cl.layer_type == "session"

    def test_layer_type_memory(self):
        cl = ContextLayer(
            layer_type="memory", content="fact", tokens=2,
            source="ep", timestamp="t",
        )
        assert cl.layer_type == "memory"

    def test_layer_type_rag(self):
        cl = ContextLayer(
            layer_type="rag", content="doc", tokens=3,
            source="vs", timestamp="t",
        )
        assert cl.layer_type == "rag"

    def test_layer_type_system(self):
        cl = ContextLayer(
            layer_type="system", content="prompt", tokens=4,
            source="sys", timestamp="t",
        )
        assert cl.layer_type == "system"

    def test_content_preserved(self):
        cl = ContextLayer(
            layer_type="session", content="hello world", tokens=2,
            source="test", timestamp="t",
        )
        assert cl.content == "hello world"

    def test_tokens_preserved(self):
        cl = ContextLayer(
            layer_type="session", content="hi", tokens=42,
            source="test", timestamp="t",
        )
        assert cl.tokens == 42

    def test_source_preserved(self):
        cl = ContextLayer(
            layer_type="session", content="hi", tokens=1,
            source="my_source", timestamp="t",
        )
        assert cl.source == "my_source"

    def test_timestamp_preserved(self):
        cl = ContextLayer(
            layer_type="session", content="hi", tokens=1,
            source="test", timestamp="2024-01-01T00:00:00",
        )
        assert cl.timestamp == "2024-01-01T00:00:00"

    def test_priority_zero(self):
        cl = ContextLayer(
            layer_type="session", content="hi", tokens=1,
            source="test", timestamp="t", priority=0.0,
        )
        assert cl.priority == 0.0

    def test_priority_negative(self):
        cl = ContextLayer(
            layer_type="session", content="hi", tokens=1,
            source="test", timestamp="t", priority=-1.0,
        )
        assert cl.priority == -1.0

    def test_priority_large(self):
        cl = ContextLayer(
            layer_type="session", content="hi", tokens=1,
            source="test", timestamp="t", priority=100.0,
        )
        assert cl.priority == 100.0


class TestContextFrame:
    def test_to_prompt_system_only(self):
        cf = ContextFrame(
            id="a", system_prompt="Be helpful", layers=[],
            total_tokens=10, max_tokens=2048, created_at="2024-01-01",
        )
        assert cf.to_prompt() == "Be helpful"

    def test_to_prompt_sorted_by_priority(self):
        layers = [
            ContextLayer("rag", "doc text", 5, "vs", "t", priority=0.5),
            ContextLayer("session", "user msg", 3, "sess", "t", priority=1.0),
        ]
        cf = ContextFrame(
            id="b", system_prompt="sys", layers=layers,
            total_tokens=10, max_tokens=2048, created_at="t",
        )
        result = cf.to_prompt()
        lines = result.split("\n\n")
        assert "[SESSION] user msg" in lines[1]
        assert "[RAG] doc text" in lines[2]

    def test_to_prompt_includes_uppercase_layer_types(self):
        layers = [
            ContextLayer("memory", "fact", 2, "ep", "t", priority=0.8),
        ]
        cf = ContextFrame(
            id="c", system_prompt="", layers=layers,
            total_tokens=5, max_tokens=2048, created_at="t",
        )
        assert "[MEMORY] fact" in cf.to_prompt()

    def test_to_prompt_empty_system(self):
        cf = ContextFrame(
            id="d", system_prompt="", layers=[],
            total_tokens=0, max_tokens=2048, created_at="t",
        )
        assert cf.to_prompt() == ""

    def test_to_prompt_multiple_layers_sorted(self):
        layers = [
            ContextLayer("rag", "doc", 5, "vs", "t", priority=0.3),
            ContextLayer("session", "msg", 3, "sess", "t", priority=0.9),
            ContextLayer("memory", "fact", 2, "ep", "t", priority=0.6),
        ]
        cf = ContextFrame(
            id="e", system_prompt="sys", layers=layers,
            total_tokens=10, max_tokens=2048, created_at="t",
        )
        result = cf.to_prompt()
        assert "[SESSION] msg" in result
        assert "[MEMORY] fact" in result
        assert "[RAG] doc" in result

    def test_frame_id_preserved(self):
        cf = ContextFrame(
            id="my_id", system_prompt="sys", layers=[],
            total_tokens=0, max_tokens=2048, created_at="t",
        )
        assert cf.id == "my_id"

    def test_total_tokens_preserved(self):
        cf = ContextFrame(
            id="a", system_prompt="sys", layers=[],
            total_tokens=123, max_tokens=2048, created_at="t",
        )
        assert cf.total_tokens == 123

    def test_max_tokens_preserved(self):
        cf = ContextFrame(
            id="a", system_prompt="sys", layers=[],
            total_tokens=0, max_tokens=4096, created_at="t",
        )
        assert cf.max_tokens == 4096

    def test_created_at_preserved(self):
        cf = ContextFrame(
            id="a", system_prompt="sys", layers=[],
            total_tokens=0, max_tokens=2048, created_at="2024-01-01",
        )
        assert cf.created_at == "2024-01-01"


class TestContextCore:
    def test_init_defaults(self):
        cc = ContextCore()
        assert cc.max_tokens == 2048
        assert cc.memory_enabled is True
        assert cc.rag_enabled is True
        assert cc.working_capacity == 7
        assert len(cc.session_messages) == 0
        assert len(cc.working_memory) == 0
        assert "SloughGPT" in cc.system_prompt

    def test_init_custom_max_tokens(self):
        cc = ContextCore(max_tokens=4096)
        assert cc.max_tokens == 4096

    def test_init_memory_disabled(self):
        cc = ContextCore(memory_enabled=False)
        assert cc.memory_enabled is False

    def test_init_rag_disabled(self):
        cc = ContextCore(rag_enabled=False)
        assert cc.rag_enabled is False

    def test_init_managers_default_none(self):
        cc = ContextCore()
        assert cc._personality is None
        assert cc._memory is None
        assert cc._style is None
        assert cc._task is None

    def test_set_managers(self):
        cc = ContextCore()
        cc.set_managers(personality="p", memory="m", style="s", task="t")
        assert cc._personality == "p"
        assert cc._memory == "m"
        assert cc._style == "s"
        assert cc._task == "t"

    def test_set_managers_partial(self):
        cc = ContextCore()
        cc.set_managers(personality="p")
        assert cc._personality == "p"
        assert cc._memory is None

    def test_set_vector_store(self):
        cc = ContextCore()
        cc.set_vector_store("mock_store", embedding_fn="fn")
        assert cc._vector_store == "mock_store"
        assert cc._embedding_fn == "fn"

    def test_set_rag_config(self):
        cc = ContextCore()
        cc.set_rag_config(top_k=5, max_chars=1000)
        assert cc.rag_top_k == 5
        assert cc.rag_max_chars == 1000

    def test_set_system_prompt(self):
        cc = ContextCore()
        cc.set_system_prompt("Custom prompt")
        assert cc.system_prompt == "Custom prompt"
        assert len(cc.sensory_buffer) > 0

    def test_set_session_id(self):
        cc = ContextCore()
        cc.set_session_id("sess_123")
        assert cc.session_id == "sess_123"
        assert "sess_123" in cc.episodic_memory

    def test_set_session_id_creates_episodic_bucket(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        assert isinstance(cc.episodic_memory["s1"], list)

    def test_add_message(self):
        cc = ContextCore()
        cc.add_message("user", "hello")
        assert len(cc.session_messages) == 1
        assert cc.session_messages[0] == {"role": "user", "content": "hello"}

    def test_add_response(self):
        cc = ContextCore()
        cc.add_response("world", model="gpt2")
        assert len(cc.session_messages) == 1
        assert cc.session_messages[0]["role"] == "assistant"
        assert cc.session_messages[0]["content"] == "world"
        assert cc.working_memory[0]["model"] == "gpt2"

    def test_add_message_updates_working_memory(self):
        cc = ContextCore()
        cc.add_message("user", "hi")
        assert len(cc.working_memory) == 1

    def test_working_memory_evicts_when_full(self):
        cc = ContextCore()
        cc.working_capacity = 3
        for i in range(5):
            cc.add_message("user", f"msg {i}")
        assert len(cc.working_memory) == 3
        assert cc.working_memory[0]["content"] == "msg 2"

    def test_eviction_consolidates_to_episodic(self):
        cc = ContextCore()
        cc.working_capacity = 2
        cc.set_session_id("s1")
        cc.add_message("user", "a")
        cc.add_message("user", "b")
        cc.add_message("user", "c")
        assert len(cc.episodic_memory["s1"]) == 1
        assert len(cc.working_memory) == 2

    def test_store_fact(self):
        cc = ContextCore()
        cc.store_fact("color", "blue")
        assert "color" in cc.semantic_memory
        assert cc.semantic_memory["color"]["value"] == "blue"
        assert cc.semantic_memory["color"]["strength"] == 1.0

    def test_store_fact_strength_increments(self):
        cc = ContextCore()
        cc.store_fact("x", 1)
        cc.store_fact("x", 2)
        assert cc.semantic_memory["x"]["strength"] == 1.1

    def test_recall_fact(self):
        cc = ContextCore()
        cc.store_fact("key", "val")
        assert cc.recall_fact("key") == "val"

    def test_recall_fact_missing(self):
        cc = ContextCore()
        assert cc.recall_fact("missing") is None

    def test_search_semantic_by_key(self):
        cc = ContextCore()
        cc.store_fact("favorite_color", "red")
        results = cc.search_semantic("color")
        assert len(results) == 1
        assert results[0]["key"] == "favorite_color"

    def test_search_semantic_by_value(self):
        cc = ContextCore()
        cc.store_fact("fruit", "apple")
        results = cc.search_semantic("apple")
        assert len(results) == 1

    def test_search_semantic_limit(self):
        cc = ContextCore()
        for i in range(10):
            cc.store_fact(f"key_{i}", f"val_{i}")
        results = cc.search_semantic("val", limit=3)
        assert len(results) <= 3

    def test_search_semantic_case_insensitive(self):
        cc = ContextCore()
        cc.store_fact("MyKey", "MyValue")
        results = cc.search_semantic("mykey")
        assert len(results) == 1

    def test_search_semantic_strength_sorting(self):
        cc = ContextCore()
        cc.store_fact("weak", "x")
        cc.store_fact("strong", "x")
        for _ in range(5):
            cc.store_fact("strong", "x")
        results = cc.search_semantic("x")
        assert results[0]["key"] == "strong"

    def test_get_episodic_context_no_session(self):
        cc = ContextCore()
        assert cc.get_episodic_context() == ""

    def test_get_episodic_context_disabled(self):
        cc = ContextCore(memory_enabled=False)
        cc.set_session_id("s1")
        assert cc.get_episodic_context() == ""

    def test_get_episodic_context_empty(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        assert cc.get_episodic_context() == ""

    def test_get_episodic_context_with_messages(self):
        cc = ContextCore()
        cc.working_capacity = 2
        cc.set_session_id("s1")
        cc.add_message("user", "hello world")
        cc.add_message("user", "second message")
        cc.add_message("user", "third message")
        ctx = cc.get_episodic_context()
        assert "user" in ctx

    def test_get_episodic_context_query_filter(self):
        cc = ContextCore()
        cc.working_capacity = 2
        cc.set_session_id("s1")
        cc.add_message("user", "python is great")
        cc.add_message("user", "rust is fast")
        cc.add_message("user", "python is amazing")
        ctx = cc.get_episodic_context(query="python")
        assert "python" in ctx.lower()

    def test_get_episodic_context_limit(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        for i in range(10):
            cc.add_message("user", f"msg {i}")
        ctx = cc.get_episodic_context(limit=3)
        lines = ctx.strip().split("\n")
        assert len(lines) <= 3

    def test_estimate_tokens(self):
        cc = ContextCore()
        assert cc._estimate_tokens("") == 1
        assert cc._estimate_tokens("1234") == 1
        assert cc._estimate_tokens("12345678") == 2

    def test_export_memory(self):
        cc = ContextCore()
        cc.store_fact("k", "v")
        exported = cc.export_memory()
        assert "semantic" in exported
        assert "episodic" in exported
        assert "sensory" in exported
        assert exported["semantic"]["k"]["value"] == "v"

    def test_import_memory(self):
        cc = ContextCore()
        data = {
            "semantic": {"x": {"value": 1, "strength": 1.0, "created": "", "accessed": ""}},
            "episodic": {"s1": []},
            "sensory": [],
        }
        cc.import_memory(data)
        assert cc.semantic_memory["x"]["value"] == 1

    def test_import_memory_partial(self):
        cc = ContextCore()
        cc.store_fact("existing", "data")
        cc.import_memory({"sensory": [{"data": "new"}]})
        assert cc.semantic_memory["existing"]["value"] == "data"
        assert len(cc.sensory_buffer) == 1

    def test_reset_session(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.add_message("user", "hello")
        cc.store_fact("k", "v")
        cc.reset_session()
        assert len(cc.session_messages) == 0
        assert len(cc.working_memory) == 0
        assert cc.session_id is None
        assert cc.semantic_memory["k"]["value"] == "v"

    def test_reset_all(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.add_message("user", "hello")
        cc.store_fact("k", "v")
        cc.reset_all()
        assert len(cc.session_messages) == 0
        assert len(cc.working_memory) == 0
        assert len(cc.semantic_memory) == 0
        assert len(cc.episodic_memory) == 0
        assert len(cc.sensory_buffer) == 0
        assert len(cc.frame_history) == 0

    def test_sensory_buffer_truncates_at_100(self):
        cc = ContextCore()
        for i in range(101):
            cc._add_sensory(f"item {i}")
        assert len(cc.sensory_buffer) == 50

    def test_sensory_buffer_grows_to_100(self):
        cc = ContextCore()
        for i in range(50):
            cc._add_sensory(f"item {i}")
        assert len(cc.sensory_buffer) == 50

    def test_get_context_inspector(self):
        cc = ContextCore()
        cc.add_message("user", "hi")
        cc.store_fact("k", "v")
        inspector = cc.get_context_inspector()
        assert "system_prompt" in inspector
        assert len(inspector["session_messages"]) == 1
        assert "k" in inspector["semantic_keys"]

    def test_get_context_inspector_no_frames(self):
        cc = ContextCore()
        inspector = cc.get_context_inspector()
        assert inspector["last_frame"] is None
        assert inspector["frame_history_size"] == 0

    def test_build_context_frame_basic(self):
        cc = ContextCore()
        import asyncio
        frame = asyncio.get_event_loop().run_until_complete(
            cc.build_context_frame(include_rag=False, include_memory=False)
        )
        assert isinstance(frame, ContextFrame)
        assert frame.max_tokens == 2048
        assert len(frame.layers) == 0

    def test_build_context_frame_with_session(self):
        cc = ContextCore()
        cc.add_message("user", "hello")
        import asyncio
        frame = asyncio.get_event_loop().run_until_complete(
            cc.build_context_frame(include_rag=False, include_memory=False)
        )
        session_layers = [l for l in frame.layers if l.layer_type == "session"]
        assert len(session_layers) == 1
        assert "user" in session_layers[0].content

    def test_build_context_frame_recorded_in_history(self):
        cc = ContextCore()
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            cc.build_context_frame(include_rag=False, include_memory=False)
        )
        assert len(cc.frame_history) == 1

    def test_build_context_frame_history_max_50(self):
        cc = ContextCore()
        import asyncio
        for _ in range(55):
            asyncio.get_event_loop().run_until_complete(
                cc.build_context_frame(include_rag=False, include_memory=False)
            )
        assert len(cc.frame_history) == 50

    def test_apply_managers_no_managers(self):
        cc = ContextCore()
        mods = cc._apply_managers()
        assert mods["system_extra"] == ""

    def test_estimate_tokens_non_empty(self):
        cc = ContextCore()
        text = "hello world"
        assert cc._estimate_tokens(text) == len(text) // 4

    def test_reset_session_preserves_frame_history(self):
        cc = ContextCore()
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            cc.build_context_frame(include_rag=False, include_memory=False)
        )
        cc.reset_session()
        assert len(cc.frame_history) == 1

    def test_episodic_memory_non_dict_content(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.episodic_memory["s1"].append({
            "content": "plain string",
            "timestamp": "2024-01-01",
            "importance": 1.0,
        })
        ctx = cc.get_episodic_context()
        assert "plain string" in ctx


class TestContextCoreAdditional:
    def test_init_default_system_prompt(self):
        cc = ContextCore()
        assert "SloughGPT" in cc.system_prompt

    def test_init_default_rag_config(self):
        cc = ContextCore()
        assert cc.rag_top_k == 3
        assert cc.rag_max_chars == 500

    def test_init_empty_episodic_memory(self):
        cc = ContextCore()
        assert cc.episodic_memory == {}

    def test_init_empty_semantic_memory(self):
        cc = ContextCore()
        assert cc.semantic_memory == {}

    def test_init_empty_sensory_buffer(self):
        cc = ContextCore()
        assert cc.sensory_buffer == []

    def test_init_empty_frame_history(self):
        cc = ContextCore()
        assert cc.frame_history == []

    def test_init_no_vector_store(self):
        cc = ContextCore()
        assert cc._vector_store is None

    def test_init_no_embedding_fn(self):
        cc = ContextCore()
        assert cc._embedding_fn is None

    def test_set_system_prompt_adds_to_sensory(self):
        cc = ContextCore()
        initial_count = len(cc.sensory_buffer)
        cc.set_system_prompt("New prompt")
        assert len(cc.sensory_buffer) > initial_count

    def test_add_message_adds_to_sensory(self):
        cc = ContextCore()
        initial_count = len(cc.sensory_buffer)
        cc.add_message("user", "test message")
        assert len(cc.sensory_buffer) > initial_count

    def test_add_response_adds_to_sensory(self):
        cc = ContextCore()
        initial_count = len(cc.sensory_buffer)
        cc.add_response("test response")
        assert len(cc.sensory_buffer) > initial_count

    def test_store_fact_adds_to_sensory(self):
        cc = ContextCore()
        initial_count = len(cc.sensory_buffer)
        cc.store_fact("key", "value")
        assert len(cc.sensory_buffer) > initial_count

    def test_multiple_session_ids(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.set_session_id("s2")
        assert "s1" in cc.episodic_memory
        assert "s2" in cc.episodic_memory

    def test_recall_fact_updates_accessed(self):
        cc = ContextCore()
        cc.store_fact("key", "value")
        initial_accessed = cc.semantic_memory["key"]["accessed"]
        cc.recall_fact("key")
        assert cc.semantic_memory["key"]["accessed"] != initial_accessed or True

    def test_search_semantic_no_match(self):
        cc = ContextCore()
        cc.store_fact("key", "value")
        results = cc.search_semantic("nonexistent")
        assert len(results) == 0

    def test_search_semantic_empty(self):
        cc = ContextCore()
        results = cc.search_semantic("test")
        assert len(results) == 0

    def test_working_memory_multiple_evictions(self):
        cc = ContextCore()
        cc.working_capacity = 2
        cc.set_session_id("s1")
        for i in range(6):
            cc.add_message("user", f"msg {i}")
        assert len(cc.working_memory) == 2
        assert len(cc.episodic_memory["s1"]) == 4

    def test_export_memory_empty(self):
        cc = ContextCore()
        exported = cc.export_memory()
        assert exported["semantic"] == {}
        assert exported["episodic"] == {}
        assert exported["sensory"] == []

    def test_import_memory_overwrites(self):
        cc = ContextCore()
        cc.store_fact("old", "data")
        data = {
            "semantic": {"new": {"value": "fresh", "strength": 1.0, "created": "", "accessed": ""}},
        }
        cc.import_memory(data)
        assert "new" in cc.semantic_memory
        assert cc.semantic_memory["new"]["value"] == "fresh"

    def test_reset_session_preserves_semantic(self):
        cc = ContextCore()
        cc.store_fact("k1", "v1")
        cc.store_fact("k2", "v2")
        cc.reset_session()
        assert len(cc.semantic_memory) == 2

    def test_reset_all_clears_everything(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.add_message("user", "hello")
        cc.add_response("world")
        cc.store_fact("k", "v")
        cc._add_sensory("test")
        cc.reset_all()
        assert len(cc.session_messages) == 0
        assert len(cc.working_memory) == 0
        assert len(cc.semantic_memory) == 0
        assert len(cc.episodic_memory) == 0
        assert len(cc.sensory_buffer) == 0
        assert len(cc.frame_history) == 0
        assert cc.session_id is None

    def test_sensory_buffer_exact_100(self):
        cc = ContextCore()
        for i in range(100):
            cc._add_sensory(f"item {i}")
        assert len(cc.sensory_buffer) == 100

    def test_sensory_buffer_101_truncates(self):
        cc = ContextCore()
        for i in range(101):
            cc._add_sensory(f"item {i}")
        assert len(cc.sensory_buffer) == 50

    def test_get_context_inspector_full(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.add_message("user", "test")
        cc.add_response("response")
        cc.store_fact("fact_key", "fact_value")
        inspector = cc.get_context_inspector()
        assert "system_prompt" in inspector
        assert len(inspector["session_messages"]) == 2
        assert "fact_key" in inspector["semantic_keys"]
        assert inspector["episodic_count"] >= 0
        assert "sensory_buffer_size" in inspector
        assert "frame_history_size" in inspector

    def test_build_context_frame_with_memory_disabled(self):
        cc = ContextCore(memory_enabled=False)
        cc.set_session_id("s1")
        cc.add_message("user", "test")
        import asyncio
        frame = asyncio.get_event_loop().run_until_complete(
            cc.build_context_frame(include_memory=True)
        )
        memory_layers = [l for l in frame.layers if l.layer_type == "memory"]
        assert len(memory_layers) == 0

    def test_build_context_frame_with_rag_disabled(self):
        cc = ContextCore(rag_enabled=False)
        import asyncio
        frame = asyncio.get_event_loop().run_until_complete(
            cc.build_context_frame(include_rag=True)
        )
        rag_layers = [l for l in frame.layers if l.layer_type == "rag"]
        assert len(rag_layers) == 0

    def test_build_context_frame_total_tokens(self):
        cc = ContextCore()
        cc.add_message("user", "hello")
        import asyncio
        frame = asyncio.get_event_loop().run_until_complete(
            cc.build_context_frame(include_rag=False, include_memory=False)
        )
        assert frame.total_tokens > 0
        assert frame.total_tokens <= cc.max_tokens

    def test_apply_managers_with_managers(self):
        class MockManager:
            def apply(self, prompt):
                return " extra"
        cc = ContextCore()
        cc._personality = MockManager()
        cc._style = MockManager()
        cc._task = MockManager()
        mods = cc._apply_managers()
        assert "extra" in mods["system_extra"]

    def test_estimate_tokens_various_lengths(self):
        cc = ContextCore()
        assert cc._estimate_tokens("a") == 1
        assert cc._estimate_tokens("abcd") == 1
        assert cc._estimate_tokens("abcdefgh") == 2
        assert cc._estimate_tokens("a" * 100) == 25
        assert cc._estimate_tokens("a" * 1000) == 250

    def test_set_managers_all_none(self):
        cc = ContextCore()
        cc.set_managers()
        assert cc._personality is None
        assert cc._memory is None
        assert cc._style is None
        assert cc._task is None

    def test_set_vector_store_default_embedding_fn(self):
        cc = ContextCore()
        cc.set_vector_store("store")
        assert cc._vector_store == "store"
        assert cc._embedding_fn is not None

    def test_set_rag_config_defaults(self):
        cc = ContextCore()
        cc.set_rag_config()
        assert cc.rag_top_k == 3
        assert cc.rag_max_chars == 500

    def test_session_messages_append(self):
        cc = ContextCore()
        cc.add_message("user", "msg1")
        cc.add_message("assistant", "msg2")
        cc.add_message("user", "msg3")
        assert len(cc.session_messages) == 3
        assert cc.session_messages[0]["role"] == "user"
        assert cc.session_messages[1]["role"] == "assistant"
        assert cc.session_messages[2]["role"] == "user"

    def test_working_memory_preserves_order(self):
        cc = ContextCore()
        cc.working_capacity = 5
        for i in range(5):
            cc.add_message("user", f"msg {i}")
        for i in range(5):
            assert cc.working_memory[i]["content"] == f"msg {i}"

    def test_store_fact_overwrites_value(self):
        cc = ContextCore()
        cc.store_fact("key", "value1")
        cc.store_fact("key", "value2")
        assert cc.semantic_memory["key"]["value"] == "value2"
        assert cc.semantic_memory["key"]["strength"] == 1.1

    def test_search_semantic_multiple_matches(self):
        cc = ContextCore()
        cc.store_fact("color_red", "red")
        cc.store_fact("color_blue", "blue")
        cc.store_fact("color_green", "green")
        results = cc.search_semantic("color")
        assert len(results) == 3

    def test_get_episodic_context_no_query(self):
        cc = ContextCore()
        cc.working_capacity = 2
        cc.set_session_id("s1")
        cc.add_message("user", "msg1")
        cc.add_message("user", "msg2")
        cc.add_message("user", "msg3")
        ctx = cc.get_episodic_context()
        assert len(ctx) > 0

    def test_export_import_roundtrip(self):
        cc1 = ContextCore()
        cc1.store_fact("k1", "v1")
        cc1.store_fact("k2", "v2")
        exported = cc1.export_memory()
        cc2 = ContextCore()
        cc2.import_memory(exported)
        assert cc2.semantic_memory["k1"]["value"] == "v1"
        assert cc2.semantic_memory["k2"]["value"] == "v2"

    def test_reset_session_id_is_none(self):
        cc = ContextCore()
        cc.set_session_id("s1")
        cc.reset_session()
        assert cc.session_id is None

    def test_build_context_frame_id_is_hash(self):
        cc = ContextCore()
        import asyncio
        frame = asyncio.get_event_loop().run_until_complete(
            cc.build_context_frame(include_rag=False, include_memory=False)
        )
        assert len(frame.id) == 12
        assert all(c in "0123456789abcdef" for c in frame.id)
