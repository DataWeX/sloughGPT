"""Tests for domains/learner/entity_extractor.py."""

import sys

import pytest

from domains.learner.entity_extractor import (
    _is_valid_entity,
    extract_entities,
    extract_facts_from_conversation,
    extract_facts_neural,
    extract_relationships,
    extract_and_store,
)


class TestIsValidEntity:
    def test_valid_word(self):
        assert _is_valid_entity("Python") is True

    def test_stop_word_rejected(self):
        assert _is_valid_entity("the") is False

    def test_short_word_rejected(self):
        assert _is_valid_entity("A") is False

    def test_punctuation_rejected(self):
        assert _is_valid_entity("...") is False


class TestExtractEntities:
    def test_empty_text(self):
        assert extract_entities("") == []

    def test_capitalized_multiple_words(self):
        result = extract_entities("John Smith works at Acme Corp.")
        assert "John Smith" in result
        assert "Acme Corp" in result

    def test_single_capitalized_word(self):
        result = extract_entities("Python is great.")
        assert "Python" in result

    def test_dedup(self):
        result = extract_entities("Maria likes tea. Maria likes tea.")
        assert result.count("Maria") == 1

    def test_false_entities_excluded(self):
        result = extract_entities("Hello there!")
        assert "Hello" not in result


class TestExtractRelationships:
    def test_is_a(self):
        result = extract_relationships("Django is a web framework")
        assert ("Django", "is_a", "web framework") in result

    def test_is_a_removes_article(self):
        result = extract_relationships("Python is a programming language")
        assert ("Python", "is_a", "programming language") in result

    def test_likes(self):
        result = extract_relationships("Alice likes coffee")
        assert ("Alice", "likes", "coffee") in result

    def test_has(self):
        result = extract_relationships("Bob has a cat")
        assert ("Bob", "has", "cat") in result

    def test_possession_apostrophe(self):
        result = extract_relationships("Sarah's laptop")
        assert ("Sarah", "possesses", "laptop") in result

    def test_works_at(self):
        result = extract_relationships("Tom works at Google")
        assert ("Tom", "works_at", "Google") in result

    def test_dedup_relationships(self):
        result = extract_relationships(
            "Django is a web framework. Django is a web framework."
        )
        assert len(result) == 1

    def test_stop_word_subject_rejected(self):
        result = extract_relationships("The dog is a pet")
        assert not any(t[0].lower() == "the" for t in result)


class TestExtractFactsFromConversation:
    def test_relationship_fact(self):
        facts = extract_facts_from_conversation(
            "Django is a web framework.", "Yes, it is."
        )
        assert "Django is a web framework" in facts

    def test_entity_fact(self):
        facts = extract_facts_from_conversation(
            "Do you know Maria?", "Maria loves programming."
        )
        assert any("Entity" in f for f in facts)

    def test_empty_conversation(self):
        assert extract_facts_from_conversation("hi", "hello") == []


class TestExtractFactsNeural:
    @pytest.mark.asyncio
    async def test_empty_when_no_registry(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules,
            "domains.infrastructure.model_registry",
            type(
                "FakeRegistryMod",
                (),
                {"get_model_registry": lambda: None},
            ),
        )
        result = await extract_facts_neural("hello", "world")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_facts_from_model(self, monkeypatch):
        class FakeModel:
            async def generate(self, prompt, **kwargs):
                class R:
                    text = "- User likes coffee\n- Cats are pets"
                return R()

        class FakeRegistry:
            def list_models(self):
                return [1]

            def get_default_model(self):
                return FakeModel()

        monkeypatch.setitem(
            sys.modules,
            "domains.infrastructure.model_registry",
            type("FakeRegistryMod", (), {"get_model_registry": lambda: FakeRegistry()}),
        )
        result = await extract_facts_neural("user", "ai")
        assert "User likes coffee" in result
        assert "Cats are pets" in result


class TestExtractAndStore:
    @pytest.mark.asyncio
    async def test_stores_facts(self, monkeypatch):
        stored = []

        class FakeKnowledgeMemory:
            def add_fact(self, fact):
                stored.append(fact.content)
                return True

        monkeypatch.setattr(
            "domains.learner.knowledge.KnowledgeFact",
            type("KF", (), {"__init__": lambda self, content, topic, source: setattr(
                self, "content", content
            ) or setattr(self, "topic", topic) or setattr(self, "source", source)}),
        )
        count = await extract_and_store(
            "Django is a web framework",
            "Yes indeed it is.",
            FakeKnowledgeMemory(),
        )
        assert count >= 1
        assert any("Django is a web framework" in f for f in stored)

    @pytest.mark.asyncio
    async def test_returns_zero_on_exception(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("bad")

        monkeypatch.setattr(
            "domains.learner.entity_extractor.extract_facts_from_conversation", boom
        )
        count = await extract_and_store("a", "b", None)
        assert count == 0
