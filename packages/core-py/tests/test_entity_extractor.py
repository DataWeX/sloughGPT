"""Tests for domains.learner.entity_extractor: entity/relationship/fact extraction."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from domains.learner import entity_extractor as ee


class TestExtractEntities:
    def test_single_capitalized_words(self):
        entities = ee.extract_entities("Alice works at Google and Microsoft.")
        assert "Alice" in entities
        assert "Google" in entities
        assert "Microsoft" in entities

    def test_multi_word_entity(self):
        entities = ee.extract_entities("I visited New York City last year.")
        assert "New York City" in entities

    def test_multi_word_parts_not_duplicated(self):
        entities = ee.extract_entities("I visited New York City last year.")
        assert "New York City" in entities
        assert "New" not in entities

    def test_common_false_entities_excluded(self):
        assert ee.extract_entities("Hello there, welcome!") == []

    def test_deduplicates(self):
        entities = ee.extract_entities("Alice likes Alice.")
        assert entities.count("Alice") == 1

    def test_empty_text(self):
        assert ee.extract_entities("") == []

    def test_lowercase_words_ignored(self):
        assert ee.extract_entities("the cat sat on the mat.") == []


class TestExtractRelationships:
    def test_likes(self):
        assert ee.extract_relationships("Alice likes pizza.") == [("Alice", "likes", "pizza")]

    def test_is_a(self):
        assert ee.extract_relationships("the dog is an animal.") == [("the dog", "is_a", "animal")]

    def test_has(self):
        assert ee.extract_relationships("Bob has a bicycle.") == [("Bob", "has", "bicycle")]

    def test_wants(self):
        assert ee.extract_relationships("Carol wants coffee.") == [("Carol", "wants", "coffee")]

    def test_works_at(self):
        assert ee.extract_relationships("Dan works at Acme Corp.") == [("Dan", "works_at", "Acme Corp")]

    def test_lives_in(self):
        assert ee.extract_relationships("Eve lives in Berlin.") == [("Eve", "lives_in", "Berlin")]

    def test_possesses(self):
        result = ee.extract_relationships("Frank's favorite food is pasta.")
        assert result[0][0] == "Frank"
        assert result[0][1] == "possesses"
        assert result[0][2].startswith("favorite")

    def test_article_stripped_from_object(self):
        result = ee.extract_relationships("Grace is a teacher.")
        assert result == [("Grace", "is_a", "teacher")]

    def test_duplicate_triples_deduplicated(self):
        result = ee.extract_relationships("Hank likes tea. Hank likes tea.")
        assert result == [("Hank", "likes", "tea")]

    def test_stop_word_subject_excluded(self):
        assert ee.extract_relationships("it likes pizza.") == []

    def test_empty_text(self):
        assert ee.extract_relationships("") == []


class TestExtractFactsFromConversation:
    def test_relationship_facts(self):
        facts = ee.extract_facts_from_conversation("Alice likes pizza.", "That is great!")
        assert "Alice likes pizza" in facts

    def test_lives_in_fact(self):
        facts = ee.extract_facts_from_conversation("Alice lives in Paris.", "Nice!")
        assert "Alice lives_in Paris" in facts

    def test_entity_facts(self):
        facts = ee.extract_facts_from_conversation("", "Alice is a developer.")
        assert "Alice is a developer" in facts

    def test_deduplicated(self):
        facts = ee.extract_facts_from_conversation("Alice likes pizza.", "Alice likes pizza.")
        assert facts.count("Alice likes pizza") == 1

    def test_empty_exchange(self):
        assert ee.extract_facts_from_conversation("", "") == []


class TestExtractFactsNeural:
    def _patch_registry(self, monkeypatch, registry):
        import domains.infrastructure.model_registry as mr

        monkeypatch.setattr(mr, "get_model_registry", lambda: registry)

    @pytest.mark.asyncio
    async def test_no_registry(self, monkeypatch):
        self._patch_registry(monkeypatch, None)
        assert await ee.extract_facts_neural("u", "a") == []

    @pytest.mark.asyncio
    async def test_no_models(self, monkeypatch):
        self._patch_registry(monkeypatch, MagicMock(list_models=lambda: []))
        assert await ee.extract_facts_neural("u", "a") == []

    @pytest.mark.asyncio
    async def test_parses_bullets(self, monkeypatch):
        model = MagicMock()
        model.generate = AsyncMock(return_value=SimpleNamespace(text="\n- Alice likes pizza\n- Bob lives in Berlin\n"))
        registry = MagicMock(list_models=lambda: [model], get_default_model=lambda: model)
        self._patch_registry(monkeypatch, registry)
        facts = await ee.extract_facts_neural("u", "a")
        assert facts == ["Alice likes pizza", "Bob lives in Berlin"]

    @pytest.mark.asyncio
    async def test_empty_output(self, monkeypatch):
        model = MagicMock()
        model.generate = AsyncMock(return_value=SimpleNamespace(text="   "))
        registry = MagicMock(list_models=lambda: [model], get_default_model=lambda: model)
        self._patch_registry(monkeypatch, registry)
        assert await ee.extract_facts_neural("u", "a") == []

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self, monkeypatch):
        model = MagicMock()
        model.generate = AsyncMock(side_effect=RuntimeError("boom"))
        registry = MagicMock(list_models=lambda: [model], get_default_model=lambda: model)
        self._patch_registry(monkeypatch, registry)
        assert await ee.extract_facts_neural("u", "a") == []


class TestExtractAndStore:
    @pytest.mark.asyncio
    async def test_stores_facts(self, monkeypatch):
        knowledge = MagicMock()
        knowledge.add_fact.return_value = True
        monkeypatch.setattr(ee, "extract_facts_neural", AsyncMock(return_value=[]))
        stored = await ee.extract_and_store("my friend likes pizza.", "Nice!", knowledge_memory=knowledge)
        assert stored == 1
        fact = knowledge.add_fact.call_args[0][0]
        assert fact.content == "my friend likes pizza"
        assert fact.source == "auto_extracted"
        assert fact.topic == "chat"

    @pytest.mark.asyncio
    async def test_no_facts_returns_zero(self, monkeypatch, knowledge_memory=None):
        knowledge = MagicMock()
        monkeypatch.setattr(ee, "extract_facts_neural", AsyncMock(return_value=[]))
        stored = await ee.extract_and_store("how are you", "I am fine", knowledge_memory=knowledge)
        assert stored == 0
        knowledge.add_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_fact_false_not_counted(self, monkeypatch):
        knowledge = MagicMock()
        knowledge.add_fact.return_value = False
        monkeypatch.setattr(ee, "extract_facts_neural", AsyncMock(return_value=[]))
        stored = await ee.extract_and_store("Alice likes pizza.", "Nice!", knowledge_memory=knowledge)
        assert stored == 0

    @pytest.mark.asyncio
    async def test_short_exchange_skips_neural(self, monkeypatch):
        knowledge = MagicMock()
        neural = AsyncMock(return_value=["neural fact here"])
        monkeypatch.setattr(ee, "extract_facts_neural", neural)
        await ee.extract_and_store("Alice likes pizza.", "Nice!", knowledge_memory=knowledge)
        neural.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_long_exchange_includes_neural(self, monkeypatch):
        knowledge = MagicMock()
        knowledge.add_fact.return_value = True
        neural = AsyncMock(return_value=["neural fact here"])
        monkeypatch.setattr(ee, "extract_facts_neural", neural)
        long_text = "x" * 150
        await ee.extract_and_store(long_text, long_text, knowledge_memory=knowledge)
        neural.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exception_returns_zero(self, monkeypatch):
        monkeypatch.setattr(ee, "extract_facts_from_conversation", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
        assert await ee.extract_and_store("u", "a", knowledge_memory=MagicMock()) == 0
