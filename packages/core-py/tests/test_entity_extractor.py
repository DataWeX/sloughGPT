"""Meaningful tests for entity_extractor — entity extraction, relationship extraction, fact extraction from conversations."""

import pytest
from domains.learner.entity_extractor import (
    _is_valid_entity, extract_entities, extract_relationships,
    extract_facts_from_conversation, _STOP_WORDS, _COMMON_FALSE_ENTITIES,
)


class TestIsValidEntity:
    def test_valid(self):
        assert _is_valid_entity("Python") is True
        assert _is_valid_entity("OpenAI") is True

    def test_stop_word(self):
        assert _is_valid_entity("the") is False
        assert _is_valid_entity("is") is False
        assert _is_valid_entity("I") is False

    def test_too_short(self):
        assert _is_valid_entity("A") is False

    def test_no_alpha_start(self):
        assert _is_valid_entity("123") is False

    def test_punctuation(self):
        assert _is_valid_entity("hello!") is False


class TestExtractEntities:
    def test_multi_word_capitalized(self):
        entities = extract_entities("I visited New York City yesterday")
        assert "New York City" in entities

    def test_single_capitalized(self):
        entities = extract_entities("Python is a great language")
        assert "Python" in entities

    def test_false_entities_excluded(self):
        entities = extract_entities("Hello there, how are you?")
        assert "Hello" not in entities

    def test_deduplication(self):
        entities = extract_entities("Python is great. Python is fun.")
        assert entities.count("Python") == 1

    def test_no_match(self):
        entities = extract_entities("hello world how are you")
        assert len(entities) == 0

    def test_multiple_entities(self):
        entities = extract_entities("I use Python and Rust at Google")
        found = [e for e in entities if e in ("Python", "Rust", "Google")]
        assert len(found) >= 2

    def test_possessive_not_entity(self):
        entities = extract_entities("John's car is red")
        assert "John" in entities
        assert "John's" not in entities


class TestExtractRelationships:
    def test_is_a(self):
        rels = extract_relationships("Python is a programming language")
        assert ("Python", "is_a", "programming language") in rels

    def test_likes(self):
        rels = extract_relationships("Alice likes chocolate")
        assert ("Alice", "likes", "chocolate") in rels

    def test_has(self):
        rels = extract_relationships("The car has four wheels")
        assert any(r[1] == "has" for r in rels)

    def test_wants(self):
        rels = extract_relationships("Bob wants a new laptop")
        assert any(r[1] == "wants" for r in rels)

    def test_uses(self):
        rels = extract_relationships("Bob uses Python daily")
        assert any(r[0] == "Bob" and r[1] == "uses" for r in rels)

    def test_works_at(self):
        rels = extract_relationships("Alice works at Google")
        assert ("Alice", "works_at", "Google") in rels

    def test_lives_in(self):
        rels = extract_relationships("Bob lives in New York")
        assert ("Bob", "lives_in", "New York") in rels

    def test_possessive(self):
        rels = extract_relationships("John's car is fast")
        assert any(r[1] == "possesses" for r in rels)

    def test_no_match(self):
        rels = extract_relationships("The weather is nice today")
        assert len(rels) == 0

    def test_deduplication(self):
        rels = extract_relationships("Python is a language. Python is a language.")
        assert len(rels) == 1

    def test_article_stripped(self):
        rels = extract_relationships("Python is a programming language")
        for _, _, obj in rels:
            assert not obj.startswith("a ")
            assert not obj.startswith("an ")


class TestExtractFactsFromConversation:
    def test_is_a_fact(self):
        facts = extract_facts_from_conversation(
            "What is Python?",
            "Python is a programming language"
        )
        assert any("Python" in f and "programming language" in f for f in facts)

    def test_likes_fact(self):
        facts = extract_facts_from_conversation(
            "What does Alice like?",
            "Alice likes hiking"
        )
        assert any("Alice" in f and "hiking" in f for f in facts)

    def test_entity_fact(self):
        facts = extract_facts_from_conversation(
            "Tell me about Tesla Motors",
            "Tesla Motors was founded by Elon Musk"
        )
        assert any("Tesla" in f for f in facts)

    def test_empty_exchange(self):
        facts = extract_facts_from_conversation("hi", "hello")
        assert len(facts) == 0

    def test_fact_min_length(self):
        facts = extract_facts_from_conversation("test", "x is a y")
        for f in facts:
            assert len(f) > 5

    def test_deduplication(self):
        facts = extract_facts_from_conversation(
            "Python is a language",
            "Python is a language and Python is great"
        )
        assert len(facts) == len(set(facts))
