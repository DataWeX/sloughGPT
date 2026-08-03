"""Tests for domains/learner/entity_extractor.py."""

import sys
import types

import pytest

from domains.learner.entity_extractor import (
    extract_entities,
    extract_relationships,
    extract_facts_from_conversation,
    extract_facts_neural,
    extract_and_store,
    _is_valid_entity,
)


# ── _is_valid_entity ───────────────────────────────────────────────────


class TestIsValidEntity:
    def test_valid_word(self):
        assert _is_valid_entity("Python") is True

    def test_stop_word_rejected(self):
        assert _is_valid_entity("The") is False

    def test_single_char_rejected(self):
        assert _is_valid_entity("A") is False


# ── extract_entities ──────────────────────────────────────────────────


class TestExtractEntities:
    def test_captures_multi_word(self):
        entities = extract_entities("Alice and Bob visited New York City.")
        assert "New York City" in entities

    def test_captures_single_caps_words(self):
        entities = extract_entities("Alice plays the piano in Paris.")
        assert "Alice" in entities
        assert "Paris" in entities

    def test_multi_word_parts_not_duplicated(self):
        entities = extract_entities("We visited New York City yesterday.")
        # "New", "York", "City" must not appear as standalone entities
        single = [e for e in entities if " " not in e]
        assert "New" not in single
        assert "York" not in single
        assert "City" not in single

    def test_deduplicates_entities(self):
        entities = extract_entities("Paris is nice. Paris is big.")
        assert entities.count("Paris") == 1

    def test_common_false_entities_excluded(self):
        entities = extract_entities("Hello there. Great job.")
        assert "Hello" not in entities
        assert "Great" not in entities

    def test_stop_words_excluded(self):
        entities = extract_entities("The cat sat on the mat.")
        assert "The" not in entities

    def test_empty_text(self):
        assert extract_entities("") == []


# ── extract_relationships ─────────────────────────────────────────────


class TestExtractRelationships:
    def test_is_a(self):
        rels = extract_relationships("Python is a language.")
        assert ("Python", "is_a", "language") in rels

    def test_is_an_article_stripped(self):
        rels = extract_relationships("Alice is an engineer.")
        assert ("Alice", "is_a", "engineer") in rels

    def test_likes(self):
        rels = extract_relationships("Alice likes pizza.")
        assert ("Alice", "likes", "pizza") in rels

    def test_has_article_stripped(self):
        rels = extract_relationships("Bob has a car.")
        assert ("Bob", "has", "car") in rels

    def test_wants(self):
        rels = extract_relationships("Carla wants a laptop.")
        assert ("Carla", "wants", "laptop") in rels

    def test_uses(self):
        rels = extract_relationships("Dave uses Linux.")
        assert ("Dave", "uses", "Linux") in rels

    def test_works_at(self):
        rels = extract_relationships("Eve works at Google.")
        assert ("Eve", "works_at", "Google") in rels

    def test_lives_in(self):
        rels = extract_relationships("Frank lives in Berlin.")
        assert ("Frank", "lives_in", "Berlin") in rels

    def test_created(self):
        rels = extract_relationships("Grace created a framework.")
        assert ("Grace", "created", "framework") in rels

    def test_called(self):
        rels = extract_relationships("Hank called Henry.")
        assert ("Hank", "called", "Henry") in rels

    def test_possessive(self):
        rels = extract_relationships("John's bike is red.")
        assert ("John", "possesses", "bike is red") in rels

    def test_stop_word_subject_filtered(self):
        rels = extract_relationships("I like Python.")
        assert not any(s == "I" for s, _, _ in rels)

    def test_single_char_object_filtered(self):
        rels = extract_relationships("X is a y.")
        assert not any(o == "y" for _, _, o in rels)

    def test_deduplicates_triples(self):
        rels = extract_relationships("Alice likes pizza. Alice likes pizza again.")
        assert rels.count(("Alice", "likes", "pizza")) == 1


# ── extract_facts_from_conversation ───────────────────────────────────


class TestExtractFactsFromConversation:
    def test_is_a_fact_format(self):
        facts = extract_facts_from_conversation("", "Python is a language")
        assert "Python is a language" in facts

    def test_possessive_becomes_has(self):
        facts = extract_facts_from_conversation("", "John's bike is red")
        assert "John has bike is red" in facts

    def test_has_fact_format(self):
        facts = extract_facts_from_conversation("", "Bob has a car")
        assert "Bob has car" in facts

    def test_other_relation_format(self):
        facts = extract_facts_from_conversation("", "Eve works at Google")
        assert "Eve works_at Google" in facts

    def test_entity_facts_added(self):
        facts = extract_facts_from_conversation("", "Alice visited Paris")
        assert any(f.startswith("Entity ") and f.endswith(" exists") for f in facts)

    def test_entity_facts_skip_false_entities(self, monkeypatch):
        monkeypatch.setattr(
            "domains.learner.entity_extractor.extract_entities",
            lambda text: ["Hello", "Alice"],
        )
        facts = extract_facts_from_conversation("", "Alice visited Paris")
        assert "Entity Hello exists" not in facts
        assert "Entity Alice exists" in facts

    def test_short_facts_filtered(self):
        facts = extract_facts_from_conversation("", "hi")
        assert all(len(f) > 5 for f in facts)


# ── extract_facts_neural ──────────────────────────────────────────────


class _FakeResult:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def __init__(self, text, raise_error=False):
        self._text = text
        self._raise = raise_error

    async def generate(self, prompt, max_new_tokens=128, temperature=0.1):
        if self._raise:
            raise RuntimeError("model down")
        return _FakeResult(self._text)


class _FakeRegistry:
    def __init__(self, models=None, default=None):
        self._models = models
        self._default = default

    def list_models(self):
        return self._models or []

    def get_default_model(self):
        return self._default


def _patch_registry(monkeypatch, registry):
    fake_module = types.ModuleType("domains.infrastructure.model_registry")
    fake_module.get_model_registry = lambda: registry
    monkeypatch.setitem(sys.modules, "domains.infrastructure.model_registry", fake_module)


class TestExtractFactsNeural:
    @pytest.mark.asyncio
    async def test_no_registry(self, monkeypatch):
        _patch_registry(monkeypatch, None)
        assert await extract_facts_neural("u", "a") == []

    @pytest.mark.asyncio
    async def test_no_models(self, monkeypatch):
        _patch_registry(monkeypatch, _FakeRegistry(models=[]))
        assert await extract_facts_neural("u", "a") == []

    @pytest.mark.asyncio
    async def test_no_default_model(self, monkeypatch):
        _patch_registry(monkeypatch, _FakeRegistry(models=["m1"], default=None))
        assert await extract_facts_neural("u", "a") == []

    @pytest.mark.asyncio
    async def test_parses_bullets(self, monkeypatch):
        model = _FakeModel("- Alice likes hiking\n- Bob works at Google\n")
        _patch_registry(monkeypatch, _FakeRegistry(models=["m1"], default=model))
        facts = await extract_facts_neural("u", "a")
        assert facts == ["Alice likes hiking", "Bob works at Google"]

    @pytest.mark.asyncio
    async def test_short_facts_filtered(self, monkeypatch):
        model = _FakeModel("- short\n- This is a sufficiently long factual statement here\n")
        _patch_registry(monkeypatch, _FakeRegistry(models=["m1"], default=model))
        facts = await extract_facts_neural("u", "a")
        assert facts == ["This is a sufficiently long factual statement here"]

    @pytest.mark.asyncio
    async def test_empty_text(self, monkeypatch):
        _patch_registry(monkeypatch, _FakeRegistry(models=["m1"], default=_FakeModel("")))
        assert await extract_facts_neural("u", "a") == []

    @pytest.mark.asyncio
    async def test_generation_error(self, monkeypatch):
        _patch_registry(monkeypatch, _FakeRegistry(models=["m1"], default=_FakeModel("", raise_error=True)))
        assert await extract_facts_neural("u", "a") == []


# ── extract_and_store ─────────────────────────────────────────────────


class _FakeMemory:
    def __init__(self, results):
        self._results = dict(results)
        self._added = []

    def add_fact(self, fact):
        key = fact.content
        if self._results.get(key, True) is False:
            return False
        if key in self._added:
            return False
        self._added.append(key)
        return True


class TestExtractAndStore:
    @pytest.mark.asyncio
    async def test_stores_extracted_facts(self, monkeypatch):
        mem = _FakeMemory({})

        async def neural_fake(user_msg, assistant_msg):
            return ["Alice likes hiking"]

        monkeypatch.setattr(
            "domains.learner.entity_extractor.extract_facts_neural", neural_fake
        )
        long_msg = (
            "Alice likes hiking in the mountains near the river and Bob works "
            "at Google in California while Carla studies physics every evening"
        )
        n = await extract_and_store("hi", long_msg, mem)
        assert n >= 1
        assert "Alice likes hiking" in mem._added

    @pytest.mark.asyncio
    async def test_short_conversation_skips_neural(self, monkeypatch):
        mem = _FakeMemory({})
        called = []

        async def neural(user_msg, assistant_msg):
            called.append(True)
            return ["extra fact"]

        monkeypatch.setattr(
            "domains.learner.entity_extractor.extract_facts_neural", neural
        )
        n = await extract_and_store("hi", "Alice likes pizza", mem)
        assert not called
        assert n >= 1

    @pytest.mark.asyncio
    async def test_no_facts_returns_zero(self, monkeypatch):
        mem = _FakeMemory({})
        monkeypatch.setattr(
            "domains.learner.entity_extractor.extract_facts_neural",
            lambda *a, **k: [],
        )
        monkeypatch.setattr(
            "domains.learner.entity_extractor.extract_facts_from_conversation",
            lambda *a, **k: [],
        )
        assert await extract_and_store("hi", "hi", mem) == 0

    @pytest.mark.asyncio
    async def test_defaults_to_singleton_memory(self, monkeypatch):
        mem = _FakeMemory({})
        monkeypatch.setattr(
            "domains.learner.knowledge.get_knowledge_memory", lambda: mem
        )
        monkeypatch.setattr(
            "domains.learner.entity_extractor.extract_facts_neural",
            lambda *a, **k: [],
        )
        n = await extract_and_store("hi", "Alice likes hiking", None)
        assert n >= 1

    @pytest.mark.asyncio
    async def test_add_fact_exception_skipped(self, monkeypatch):
        mem = _FakeMemory({})

        def add_fact(fact):
            raise RuntimeError("boom")

        mem.add_fact = add_fact
        monkeypatch.setattr(
            "domains.learner.entity_extractor.extract_facts_neural",
            lambda *a, **k: [],
        )
        n = await extract_and_store("hi", "Alice likes hiking", mem)
        assert n == 0

    @pytest.mark.asyncio
    async def test_outer_exception_returns_zero(self, monkeypatch):
        mem = _FakeMemory({})

        def boom(user_msg, assistant_msg):
            raise RuntimeError("extraction failed")

        monkeypatch.setattr(
            "domains.learner.entity_extractor.extract_facts_from_conversation", boom
        )
        assert await extract_and_store("hi", "hello there", mem) == 0
