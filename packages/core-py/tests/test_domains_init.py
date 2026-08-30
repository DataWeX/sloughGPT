"""Tests for domains.__init__ — BaseComponent, Memory, Thought, and exceptions."""

import asyncio
import time
import pytest
from domains import (
    BaseComponent, ComponentException, Memory, Thought, ThoughtType,
    BaseDomain, DomainException,
    ICognitiveProcessor, IMemoryManager, IMetacognitiveMonitor, IReasoningEngine,
)


class TestBaseComponent:
    def setup_method(self):
        self.c = BaseComponent("test")

    def test_name(self):
        assert self.c.component_name == "test"

    def test_not_initialized(self):
        assert self.c.is_initialized is False

    def test_initialize(self):
        asyncio.run(self.c.initialize())
        assert self.c.is_initialized is True

    def test_shutdown(self):
        asyncio.run(self.c.initialize())
        asyncio.run(self.c.shutdown())
        assert self.c.is_initialized is False

    def test_shutdown_without_init(self):
        asyncio.run(self.c.shutdown())
        assert self.c.is_initialized is False

    def test_double_init(self):
        asyncio.run(self.c.initialize())
        asyncio.run(self.c.initialize())
        assert self.c.is_initialized is True

    def test_init_shutdown_cycle(self):
        for _ in range(5):
            asyncio.run(self.c.initialize())
            assert self.c.is_initialized is True
            asyncio.run(self.c.shutdown())
            assert self.c.is_initialized is False

    def test_component_name_type(self):
        c = BaseComponent(123)
        assert c.component_name == 123

    def test_empty_name(self):
        c = BaseComponent("")
        assert c.component_name == ""

    def test_isolation_between_instances(self):
        a = BaseComponent("a")
        b = BaseComponent("b")
        asyncio.run(a.initialize())
        assert a.is_initialized is True
        assert b.is_initialized is False

    def test_shutdown_resets_state(self):
        asyncio.run(self.c.initialize())
        asyncio.run(self.c.shutdown())
        assert self.c.is_initialized is False
        asyncio.run(self.c.initialize())
        assert self.c.is_initialized is True

    def test_multiple_instances_independent(self):
        comps = [BaseComponent(f"c{i}") for i in range(5)]
        asyncio.run(comps[2].initialize())
        assert comps[0].is_initialized is False
        assert comps[2].is_initialized is True
        assert comps[4].is_initialized is False

    def test_component_name_none(self):
        c = BaseComponent(None)
        assert c.component_name is None

    def test_component_name_string_numeric(self):
        c = BaseComponent("42")
        assert c.component_name == "42"
        asyncio.run(c.initialize())
        assert c.is_initialized is True

    def test_shutdown_idempotent(self):
        asyncio.run(self.c.shutdown())
        asyncio.run(self.c.shutdown())
        assert self.c.is_initialized is False

    def test_initialize_returns_none(self):
        result = asyncio.run(self.c.initialize())
        assert result is None

    def test_shutdown_returns_none(self):
        asyncio.run(self.c.initialize())
        result = asyncio.run(self.c.shutdown())
        assert result is None

    def test_component_name_whitespace(self):
        c = BaseComponent("  hello  ")
        assert c.component_name == "  hello  "


class TestComponentException:
    def test_is_exception(self):
        assert issubclass(ComponentException, Exception)

    def test_message(self):
        e = ComponentException("bad config")
        assert str(e) == "bad config"

    def test_empty_message(self):
        e = ComponentException("")
        assert str(e) == ""

    def test_raise_and_catch(self):
        with pytest.raises(ComponentException):
            raise ComponentException("fail")

    def test_chained_exception(self):
        try:
            raise ValueError("root")
        except ValueError as ve:
            try:
                raise ComponentException("wrapped") from ve
            except ComponentException as ce:
                assert isinstance(ce.__cause__, ValueError)

    def test_args(self):
        e = ComponentException("msg", "extra")
        assert e.args == ("msg", "extra")

    def test_is_subclass_of_exception(self):
        assert issubclass(ComponentException, Exception)

    def test_repr(self):
        e = ComponentException("test")
        assert "ComponentException" in repr(e)

    def test_catch_as_generic_exception(self):
        with pytest.raises(Exception):
            raise ComponentException("generic")

    def test_no_args(self):
        e = ComponentException()
        assert str(e) == ""

    def test_numeric_message(self):
        e = ComponentException(42)
        assert str(e) == "42"

    def test_none_message(self):
        e = ComponentException(None)
        assert str(e) == "None"

    def test_catch_specific_then_generic(self):
        caught = False
        try:
            raise ComponentException("x")
        except ComponentException:
            caught = True
        assert caught


class TestMemory:
    def test_defaults(self):
        m = Memory("k", "v")
        assert m.key == "k"
        assert m.value == "v"
        assert m.memory_type == "episodic"
        assert m.importance == 0.5
        assert m.retrieval_count == 0

    def test_custom_type(self):
        m = Memory("k", "v", memory_type="semantic", importance=0.9)
        assert m.memory_type == "semantic"
        assert m.importance == 0.9

    def test_content_alias(self):
        m = Memory("k", "hello")
        assert m.content == "hello"

    def test_metadata(self):
        m = Memory("k", "v")
        assert m.metadata == {}

    def test_metadata_mutation(self):
        m = Memory("k", "v")
        m.metadata["source"] = "test"
        assert m.metadata["source"] == "test"

    def test_last_accessed_is_recent(self):
        before = time.time()
        m = Memory("k", "v")
        after = time.time()
        assert before <= m.last_accessed <= after

    def test_value_types(self):
        m_list = Memory("k", [1, 2, 3])
        assert m_list.value == [1, 2, 3]
        m_dict = Memory("k", {"a": 1})
        assert m_dict.value == {"a": 1}
        m_none = Memory("k", None)
        assert m_none.value is None

    def test_retrieval_count_starts_zero(self):
        m = Memory("k", "v")
        assert m.retrieval_count == 0
        m.retrieval_count += 1
        assert m.retrieval_count == 1

    def test_importance_range(self):
        m_low = Memory("k", "v", importance=0.0)
        assert m_low.importance == 0.0
        m_high = Memory("k", "v", importance=1.0)
        assert m_high.importance == 1.0

    def test_different_memory_types(self):
        for mt in ("episodic", "semantic", "procedural", "working"):
            m = Memory("k", "v", memory_type=mt)
            assert m.memory_type == mt

    def test_empty_key(self):
        m = Memory("", "v")
        assert m.key == ""

    def test_content_and_value_same(self):
        m = Memory("k", "data")
        assert m.content is m.value

    def test_retrieval_count_increment(self):
        m = Memory("k", "v")
        for i in range(10):
            m.retrieval_count += 1
        assert m.retrieval_count == 10

    def test_metadata_multiple_keys(self):
        m = Memory("k", "v")
        m.metadata["a"] = 1
        m.metadata["b"] = 2
        m.metadata["c"] = 3
        assert len(m.metadata) == 3

    def test_last_accessed_monotonic(self):
        m1 = Memory("k1", "v1")
        time.sleep(0.001)
        m2 = Memory("k2", "v2")
        assert m1.last_accessed <= m2.last_accessed

    def test_value_complex_object(self):
        data = {"nested": {"deep": [1, 2, 3]}}
        m = Memory("k", data)
        assert m.value == data
        assert m.content == data

    def test_memory_type_arbitrary(self):
        m = Memory("k", "v", memory_type="custom_type")
        assert m.memory_type == "custom_type"

    def test_importance_out_of_range(self):
        m = Memory("k", "v", importance=2.5)
        assert m.importance == 2.5

    def test_negative_importance(self):
        m = Memory("k", "v", importance=-0.5)
        assert m.importance == -0.5

    def test_key_preserves_type(self):
        m = Memory(123, "v")
        assert m.key == 123

    def test_many_metadata_entries(self):
        m = Memory("k", "v")
        for i in range(100):
            m.metadata[f"key_{i}"] = i
        assert len(m.metadata) == 100


class TestThoughtType:
    def test_values(self):
        assert ThoughtType.PERCEPTION == "perception"
        assert ThoughtType.REASONING == "reasoning"
        assert ThoughtType.CREATIVITY == "creativity"
        assert ThoughtType.REFLECTION == "reflection"
        assert ThoughtType.DECISION == "decision"

    def test_all_values_are_strings(self):
        assert isinstance(ThoughtType.PERCEPTION, str)
        assert isinstance(ThoughtType.REASONING, str)
        assert isinstance(ThoughtType.CREATIVITY, str)
        assert isinstance(ThoughtType.REFLECTION, str)
        assert isinstance(ThoughtType.DECISION, str)

    def test_unique_values(self):
        values = [
            ThoughtType.PERCEPTION, ThoughtType.REASONING,
            ThoughtType.CREATIVITY, ThoughtType.REFLECTION,
            ThoughtType.DECISION,
        ]
        assert len(values) == len(set(values))

    def test_count(self):
        values = [
            ThoughtType.PERCEPTION, ThoughtType.REASONING,
            ThoughtType.CREATIVITY, ThoughtType.REFLECTION,
            ThoughtType.DECISION,
        ]
        assert len(values) == 5

    def test_can_be_used_in_set(self):
        s = {ThoughtType.PERCEPTION, ThoughtType.REASONING}
        assert len(s) == 2

    def test_can_be_compared_directly(self):
        assert ThoughtType.PERCEPTION == "perception"

    def test_can_be_used_as_dict_key(self):
        d = {ThoughtType.PERCEPTION: 1}
        assert d["perception"] == 1


class TestThought:
    def test_defaults(self):
        t = Thought("t1", "hello")
        assert t.thought_id == "t1"
        assert t.content == "hello"
        assert t.thought_type == "reasoning"
        assert t.confidence == 0.5
        assert t.metadata == {}

    def test_custom(self):
        t = Thought("t2", "idea", thought_type="creativity", confidence=0.9)
        assert t.thought_type == "creativity"
        assert t.confidence == 0.9

    def test_metadata_default_empty(self):
        t = Thought("t", "c")
        assert t.metadata == {}

    def test_metadata_custom(self):
        t = Thought("t", "c", metadata={"source": "test"})
        assert t.metadata["source"] == "test"

    def test_metadata_not_shared(self):
        t1 = Thought("t1", "c1")
        t2 = Thought("t2", "c2")
        t1.metadata["key"] = "val"
        assert "key" not in t2.metadata

    def test_confidence_zero(self):
        t = Thought("t", "c", confidence=0.0)
        assert t.confidence == 0.0

    def test_confidence_one(self):
        t = Thought("t", "c", confidence=1.0)
        assert t.confidence == 1.0

    def test_empty_content(self):
        t = Thought("t", "")
        assert t.content == ""

    def test_all_thought_types(self):
        for tt in ("perception", "reasoning", "creativity", "reflection", "decision"):
            t = Thought("t", "c", thought_type=tt)
            assert t.thought_type == tt

    def test_metadata_mutation(self):
        t = Thought("t", "c")
        t.metadata["key"] = "val"
        assert t.metadata["key"] == "val"

    def test_high_confidence(self):
        t = Thought("t", "c", confidence=0.99)
        assert t.confidence == 0.99

    def test_negative_confidence(self):
        t = Thought("t", "c", confidence=-0.5)
        assert t.confidence == -0.5

    def test_over_one_confidence(self):
        t = Thought("t", "c", confidence=1.5)
        assert t.confidence == 1.5

    def test_empty_thought_id(self):
        t = Thought("", "c")
        assert t.thought_id == ""

    def test_long_content(self):
        long = "x" * 10000
        t = Thought("t", long)
        assert len(t.content) == 10000

    def test_none_metadata_uses_empty_dict(self):
        t = Thought("t", "c", metadata=None)
        assert t.metadata == {}

    def test_metadata_shared_reference(self):
        md = {"a": 1}
        t = Thought("t", "c", metadata=md)
        md["b"] = 2
        assert t.metadata["b"] == 2


class TestBaseDomain:
    def test_name(self):
        d = BaseDomain("chat")
        assert d.domain_name == "chat"

    def test_empty_name(self):
        d = BaseDomain("")
        assert d.domain_name == ""

    def test_various_names(self):
        for name in ("chat", "training", "inference", "feedback"):
            d = BaseDomain(name)
            assert d.domain_name == name

    def test_none_name(self):
        d = BaseDomain(None)
        assert d.domain_name is None

    def test_numeric_name(self):
        d = BaseDomain(42)
        assert d.domain_name == 42

    def test_long_name(self):
        d = BaseDomain("x" * 1000)
        assert len(d.domain_name) == 1000

    def test_multiple_instances(self):
        domains = [BaseDomain(f"d{i}") for i in range(10)]
        assert all(d.domain_name == f"d{i}" for i, d in enumerate(domains))

    def test_name_not_shared(self):
        a = BaseDomain("a")
        b = BaseDomain("b")
        assert a.domain_name == "a"
        assert b.domain_name == "b"


class TestDomainException:
    def test_is_exception(self):
        assert issubclass(DomainException, Exception)

    def test_message(self):
        e = DomainException("domain error")
        assert str(e) == "domain error"

    def test_empty_message(self):
        e = DomainException("")
        assert str(e) == ""

    def test_raise_and_catch(self):
        with pytest.raises(DomainException):
            raise DomainException("fail")

    def test_chained_exception(self):
        try:
            raise ValueError("root")
        except ValueError as ve:
            try:
                raise DomainException("wrapped") from ve
            except DomainException as de:
                assert isinstance(de.__cause__, ValueError)

    def test_args(self):
        e = DomainException("msg", "extra")
        assert e.args == ("msg", "extra")

    def test_repr(self):
        e = DomainException("test")
        assert "DomainException" in repr(e)

    def test_catch_as_generic_exception(self):
        with pytest.raises(Exception):
            raise DomainException("generic")

    def test_numeric_message(self):
        e = DomainException(42)
        assert str(e) == "42"

    def test_multiple_causes(self):
        try:
            raise ValueError("inner")
        except ValueError as ve:
            try:
                raise TypeError("mid") from ve
            except TypeError as te:
                try:
                    raise DomainException("outer") from te
                except DomainException as de:
                    assert isinstance(de.__cause__, TypeError)
                    assert isinstance(de.__cause__.__cause__, ValueError)

    def test_catch_inherits_from_exception(self):
        assert issubclass(DomainException, Exception)
        try:
            raise DomainException("x")
        except Exception:
            pass


# ── Protocol tests ────────────────────────────────────────────────────

class TestICognitiveProcessor:
    def test_protocol_check(self):
        class Good:
            async def process(self, input_data):
                return input_data
        assert isinstance(Good(), ICognitiveProcessor)

    def test_protocol_reject_missing(self):
        class Bad:
            pass
        assert not isinstance(Bad(), ICognitiveProcessor)

    def test_protocol_runtime_checkable(self):
        assert isinstance(ICognitiveProcessor, type)

    def test_extra_methods_ok(self):
        class Overloaded:
            async def process(self, input_data):
                return input_data
            async def extra(self):
                pass
        assert isinstance(Overloaded(), ICognitiveProcessor)

    def test_wrong_signature_still_satisfies(self):
        class WrongSig:
            async def process(self):
                return None
        assert isinstance(WrongSig(), ICognitiveProcessor)

    def test_sync_process_satisfies_py312(self):
        """In Python 3.12+, runtime_checkable protocols accept sync methods."""
        class Sync:
            def process(self, input_data):
                return input_data
        # Python 3.12+ relaxed runtime_checkable to accept sync implementations
        assert isinstance(Sync(), ICognitiveProcessor)

    def test_class_not_instance(self):
        # Protocol class itself is not an instance of the protocol
        result = isinstance(ICognitiveProcessor, ICognitiveProcessor)
        # This is implementation-dependent; just verify it doesn't crash
        assert isinstance(result, bool)


class TestIMemoryManager:
    def test_protocol_check(self):
        class Good:
            async def store(self, key, value):
                pass
            async def recall(self, key):
                return None
        assert isinstance(Good(), IMemoryManager)

    def test_protocol_reject_missing(self):
        class Bad:
            async def store(self, key, value):
                pass
        assert not isinstance(Bad(), IMemoryManager)

    def test_extra_methods_ok(self):
        class Overloaded:
            async def store(self, key, value):
                pass
            async def recall(self, key):
                return None
            async def delete(self, key):
                pass
        assert isinstance(Overloaded(), IMemoryManager)

    def test_missing_recall_only(self):
        class NoRecall:
            async def store(self, key, value):
                pass
        assert not isinstance(NoRecall(), IMemoryManager)

    def test_missing_store_only(self):
        class NoStore:
            async def recall(self, key):
                return None
        assert not isinstance(NoStore(), IMemoryManager)

    def test_sync_store_satisfies_py312(self):
        """In Python 3.12+, runtime_checkable protocols accept sync methods."""
        class Sync:
            def store(self, key, value):
                pass
            async def recall(self, key):
                return None
        assert isinstance(Sync(), IMemoryManager)


class TestIMetacognitiveMonitor:
    def test_protocol_check(self):
        class Good:
            async def monitor(self, state):
                pass
        assert isinstance(Good(), IMetacognitiveMonitor)

    def test_protocol_reject_missing(self):
        class Bad:
            pass
        assert not isinstance(Bad(), IMetacognitiveMonitor)

    def test_extra_methods_ok(self):
        class Overloaded:
            async def monitor(self, state):
                pass
            async def extra(self):
                pass
        assert isinstance(Overloaded(), IMetacognitiveMonitor)

    def test_wrong_param_name_still_satisfies(self):
        class DifferentParam:
            async def monitor(self, something_else):
                pass
        assert isinstance(DifferentParam(), IMetacognitiveMonitor)

    def test_sync_monitor_satisfies_py312(self):
        """In Python 3.12+, runtime_checkable protocols accept sync methods."""
        class Sync:
            def monitor(self, state):
                pass
        assert isinstance(Sync(), IMetacognitiveMonitor)

    def test_missing_monitor_wrong_method(self):
        class WrongMethod:
            async def observe(self, state):
                pass
        assert not isinstance(WrongMethod(), IMetacognitiveMonitor)


class TestIReasoningEngine:
    def test_protocol_check(self):
        class Good:
            async def reason(self, context):
                return None
        assert isinstance(Good(), IReasoningEngine)

    def test_protocol_reject_missing(self):
        class Bad:
            pass
        assert not isinstance(Bad(), IReasoningEngine)

    def test_extra_methods_ok(self):
        class Overloaded:
            async def reason(self, context):
                return None
            async def plan(self, context):
                return None
        assert isinstance(Overloaded(), IReasoningEngine)

    def test_sync_reason_satisfies_py312(self):
        """In Python 3.12+, runtime_checkable protocols accept sync methods."""
        class Sync:
            def reason(self, context):
                return None
        assert isinstance(Sync(), IReasoningEngine)

    def test_wrong_return_still_satisfies(self):
        class NoReturn:
            async def reason(self, context):
                pass
        assert isinstance(NoReturn(), IReasoningEngine)

    def test_missing_reason_wrong_name(self):
        class WrongName:
            async def think(self, context):
                return None
        assert not isinstance(WrongName(), IReasoningEngine)

    def test_class_not_instance(self):
        result = isinstance(IReasoningEngine, IReasoningEngine)
        assert isinstance(result, bool)
