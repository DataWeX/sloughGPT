"""Tests for domains.__init__ — BaseComponent, Memory, Thought, and exceptions."""

import time
import pytest
from domains import (
    BaseComponent, ComponentException, Memory, Thought, ThoughtType,
    BaseDomain, DomainException,
)


class TestBaseComponent:
    def setup_method(self):
        self.c = BaseComponent("test")

    def test_name(self):
        assert self.c.component_name == "test"

    def test_not_initialized(self):
        assert self.c.is_initialized is False

    def test_initialize(self):
        import asyncio
        asyncio.run(self.c.initialize())
        assert self.c.is_initialized is True

    def test_shutdown(self):
        import asyncio
        asyncio.run(self.c.initialize())
        asyncio.run(self.c.shutdown())
        assert self.c.is_initialized is False


class TestComponentException:
    def test_is_exception(self):
        assert issubclass(ComponentException, Exception)

    def test_message(self):
        e = ComponentException("bad config")
        assert str(e) == "bad config"


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


class TestThoughtType:
    def test_values(self):
        assert ThoughtType.PERCEPTION == "perception"
        assert ThoughtType.REASONING == "reasoning"
        assert ThoughtType.CREATIVITY == "creativity"
        assert ThoughtType.REFLECTION == "reflection"
        assert ThoughtType.DECISION == "decision"


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


class TestBaseDomain:
    def test_name(self):
        d = BaseDomain("chat")
        assert d.domain_name == "chat"


class TestDomainException:
    def test_is_exception(self):
        assert issubclass(DomainException, Exception)

    def test_message(self):
        e = DomainException("domain error")
        assert str(e) == "domain error"
