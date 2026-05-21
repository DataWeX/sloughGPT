"""Integration tests for the full learner pipeline: extract → store → augment → ingest → train → deploy.

Tests the chain:
  entity_extractor.extract_facts_from_conversation()
  → KnowledgeMemory.add_fact() / add_article()
  → enrich_with_knowledge()
  → ContinualLearner.ingest_text() / ingest_conversation()
  → ContinualLearner.train_now()
  → ContinualLearner.evaluate()
  → ContinualLearner.deploy()
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from domains.inference.vector_store import InMemoryVectorStore
from domains.learner.entity_extractor import (
    extract_entities,
    extract_relationships,
    extract_facts_from_conversation,
)
from domains.learner.knowledge import (
    KnowledgeMemory,
    KnowledgeFact,
    KnowledgeIngestor,
    _extract_topics,
)
from domains.learner.knowledge_augmenter import enrich_with_knowledge
from domains.learner.continual import ContinualLearner, _tokenize
from domains.learner.data_filter import DataFilter


# ===== Entity Extraction → Storage =====


class TestExtractToStore:
    """Entity extraction produces facts that can be stored in KnowledgeMemory."""

    def test_extract_entities_returns_list(self):
        entities = extract_entities("Alice works at Acme Corp in New York City")
        assert isinstance(entities, list)

    def test_extract_relationships_returns_triples(self):
        rels = extract_relationships("Alice works at Acme Corp")
        assert len(rels) > 0
        subj, rel, obj = rels[0]
        assert subj == "Alice"
        assert rel == "works_at"
        assert obj == "Acme Corp"

    def test_extract_facts_from_conversation_returns_strings(self):
        facts = extract_facts_from_conversation(
            "My name is Bob and I like Python",
            "Hello Bob, Python is great!",
        )
        assert isinstance(facts, list)
        assert all(isinstance(f, str) for f in facts)
        assert len(facts) > 0

    def test_extracted_facts_can_be_stored_in_knowledge_memory(self):
        facts = extract_facts_from_conversation(
            "Bob is a data scientist at Acme Corp",
            "Bob specializes in machine learning",
        )
        memory = KnowledgeMemory(vector_store=InMemoryVectorStore(dimension=384))
        memory._visited.clear()
        stored = 0
        for fact_text in facts:
            fact = KnowledgeFact(
                content=fact_text,
                topic="relationships",
                source="chat",
            )
            if memory.add_fact(fact):
                stored += 1
        assert stored > 0

    def test_multiple_facts_deduplicated(self):
        memory = KnowledgeMemory(vector_store=InMemoryVectorStore(dimension=384))
        memory._visited.clear()
        unique = f"Unique test fact {id(memory)}"
        fact = KnowledgeFact(content=unique, topic="tech", source="test")
        assert memory.add_fact(fact) is True
        assert memory.add_fact(fact) is False  # duplicate should be rejected


# ===== KnowledgeMemory Storage & Search =====


class TestKnowledgeMemory:
    """KnowledgeMemory backed by InMemoryVectorStore works without external services."""

    @pytest.fixture
    def memory(self):
        m = KnowledgeMemory(vector_store=InMemoryVectorStore(dimension=384))
        m._visited.clear()
        return m

    def test_add_and_search_fact(self, memory):
        fact = KnowledgeFact(
            content="The Eiffel Tower is in Paris, France. It was built in 1889.",
            topic="landmarks",
            source="test",
        )
        assert memory.add_fact(fact) is True

        results = memory.search("Eiffel Tower Paris", top_k=5)
        assert len(results) > 0
        assert "Eiffel" in results[0]["content"]

    def test_add_article_facts(self, memory):
        added = memory.add_article(
            url="https://example.com/ai",
            title="Latest AI Breakthroughs",
            content="Deep learning models continue to improve. New architectures achieve better results. "
            "Transformers are the foundation of modern NLP. Reinforcement learning advances robotics.",
            source="test",
        )
        assert added > 0

        results = memory.search("deep learning transformers", top_k=5)
        assert len(results) > 0

    def test_query_by_topic(self, memory):
        memory.add_fact(KnowledgeFact(content="Python is a language", topic="tech", source="test"))
        memory.add_fact(KnowledgeFact(content="Cats are mammals", topic="animals", source="test"))

        results = memory.query("tech", top_k=5)
        assert len(results) > 0
        assert all(r["topic"] == "tech" for r in results)

    def test_search_empty_returns_empty(self, memory):
        results = memory.search("nothing matches this query")
        assert results == []

    def test_stats_reflect_added_facts(self, memory):
        memory.add_fact(KnowledgeFact(content="Fact one", topic="a", source="test"))
        memory.add_fact(KnowledgeFact(content="Fact two", topic="b", source="test"))
        stats = memory.stats()
        assert stats["total_facts"] >= 2


# ===== Knowledge Augmenter Pipeline =====


class TestKnowledgeAugmenter:
    """enrich_with_knowledge retrieves stored facts to augment chat messages."""

    def test_enrich_with_stored_facts(self):
        memory = KnowledgeMemory(vector_store=InMemoryVectorStore(dimension=384))
        memory._visited.clear()
        memory.add_fact(KnowledgeFact(
            content="Python is a high-level programming language created by Guido van Rossum",
            topic="programming",
            source="test",
        ))
        memory.add_fact(KnowledgeFact(
            content="TypeScript adds static typing to JavaScript",
            topic="programming",
            source="test",
        ))

        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=memory):
            with patch("domains.learner.knowledge_augmenter.get_knowledge_ingestor") as mock_ingestor:
                mock_ingestor.return_value.search_and_ingest.return_value = {"new_facts": 0, "rejected": 0}
                result = enrich_with_knowledge("Tell me about Python")
        assert result["source"] in ("memory", "none")
        if result["source"] == "memory":
            assert len(result["facts"]) > 0
            assert any("Python" in f for f in result["facts"])

    def test_enrich_empty_with_no_data(self):
        memory = KnowledgeMemory(vector_store=InMemoryVectorStore(dimension=384))
        memory._visited.clear()

        with patch("domains.learner.knowledge_augmenter.get_knowledge_memory", return_value=memory):
            with patch("domains.learner.knowledge_augmenter.get_knowledge_ingestor") as mock_ingestor:
                mock_ingestor.return_value.search_and_ingest.return_value = {"new_facts": 0, "rejected": 0}
                result = enrich_with_knowledge("Tell me about something unknown", auto_search=False)
        assert result["source"] == "none"
        assert result["facts"] == []


# ===== Continual Learner: Ingest → Train → Evaluate =====


class TestLearnerPipeline:
    """ContinualLearner with mocked background loop for deterministic tests."""

    @pytest.fixture
    def learner(self, tmp_path):
        with (
            patch.object(ContinualLearner, "_background_loop", lambda self: None),
            patch.object(ContinualLearner, "_save_checkpoint", lambda self: None),
            patch("domains.learner.continual.STATE_PATH", tmp_path / "continual.slo"),
        ):
            l = ContinualLearner()
            l._running = False
            yield l
            l.shutdown()

    @pytest.mark.slow
    def test_ingest_and_train(self, learner):
        text = "the cat sat on the mat and the dog played in the yard " * 20
        learner.ingest_text(text)
        assert learner.total_tokens_ingested > 0
        assert len(learner.buffer) > 0

        status = learner.train_now()
        assert status["train_steps_completed"] >= 1
        assert status["current_loss"] > 0

    @pytest.mark.slow
    def test_evaluate_after_training(self, learner):
        text = "the cat sat on the mat and the dog played in the yard " * 20
        learner.ingest_text(text)
        learner.train_now()

        result = learner.evaluate(text="the cat sat on the mat and the dog")
        assert "perplexity" in result
        assert result["perplexity"] > 0

    @pytest.mark.slow
    def test_ingest_conversation_and_train(self, learner):
        learner.ingest_conversation([
            ("hi", "hello there"),
            ("what is python", "python is a programming language"),
        ])
        assert learner.total_tokens_ingested > 0

        status = learner.train_now()
        assert status["train_steps_completed"] >= 1

    @pytest.mark.slow
    def test_status_reflects_state(self, learner):
        learner.ingest_text("some training data")
        status = learner.status()
        assert status["soul_name"] == "continual"
        assert status["buffer_size"] > 0
        assert status["total_tokens_ingested"] > 0
        assert "arch" in status
        assert status["arch"] == "transformer"

    @pytest.mark.slow
    def test_evaluate_without_data_returns_error(self, learner):
        result = learner.evaluate()
        assert "error" in result

    @pytest.mark.slow
    def test_buffer_capacity_respected(self, learner):
        big = "hello world " * 5000
        learner.ingest_text(big)
        assert len(learner.buffer) <= 10000


# ===== Full Pipeline Integration =====


class TestFullPipeline:
    """End-to-end: extract → store → augment → ingest → train → evaluate."""

    def _make_memory_with_facts(self):
        memory = KnowledgeMemory(vector_store=InMemoryVectorStore(dimension=384))
        memory.add_fact(KnowledgeFact(
            content="The Python programming language was created by Guido van Rossum in 1991",
            topic="programming",
            source="test",
        ))
        memory.add_fact(KnowledgeFact(
            content="Transformers are a neural network architecture for sequence processing",
            topic="ai",
            source="test",
        ))
        return memory

    @pytest.mark.slow
    def test_extract_store_and_search(self):
        conversation_facts = extract_facts_from_conversation(
            "Alice is a machine learning engineer at Google",
            "Alice works on transformer models for NLP",
        )
        assert len(conversation_facts) > 0

        memory = self._make_memory_with_facts()
        memory._visited.clear()
        for fact_text in conversation_facts:
            memory.add_fact(KnowledgeFact(
                content=fact_text,
                topic="chat",
                source="chat",
            ))

        results = memory.search("machine learning engineer", top_k=5)
        assert len(results) >= 1

    @pytest.mark.slow
    def test_extract_augment_ingest_chain(self):
        facts = extract_facts_from_conversation(
            "I love programming in Rust",
            "Rust is a systems programming language focused on safety",
        )
        assert len(facts) > 0

        memory = KnowledgeMemory(vector_store=InMemoryVectorStore(dimension=384))
        for f in facts:
            memory.add_fact(KnowledgeFact(content=f, topic="programming", source="chat"))

        augment = enrich_with_knowledge("Tell me about Rust", auto_search=False)
        assert augment["source"] in ("memory", "none")

        with (
            patch.object(ContinualLearner, "_background_loop", lambda self: None),
            patch.object(ContinualLearner, "_save_checkpoint", lambda self: None),
            patch("domains.learner.continual.STATE_PATH", Path(tempfile.mkdtemp()) / "continual.slo"),
        ):
            learner = ContinualLearner()
            try:
                learner._running = False
                for f in facts:
                    learner.ingest_text(f)
                assert learner.total_tokens_ingested > 0

                status = learner.train_now()
                assert status["train_steps_completed"] >= 1
            finally:
                learner.shutdown()

    @pytest.mark.slow
    def test_deploy_creates_file(self):
        with (
            patch.object(ContinualLearner, "_background_loop", lambda self: None),
            patch.object(ContinualLearner, "_save_checkpoint", lambda self: None),
            patch("domains.learner.continual.STATE_PATH", Path(tempfile.mkdtemp()) / "continual.slo"),
        ):
            learner = ContinualLearner()
            try:
                learner._running = False
                learner.ingest_text("training data for deployment " * 20)
                learner.train_now()

                with patch.object(Path, "home", return_value=Path(tempfile.mkdtemp())):
                    result = learner.deploy(name="integration-test-deploy")
                    assert "path" in result
                    assert result["soul_name"] == "continual"
                    assert result["steps"] >= 1
                    assert result["arch"] == "transformer"
            finally:
                learner.shutdown()

    def test_extract_topics_from_text(self):
        topics = _extract_topics("Machine learning and artificial intelligence are transforming technology")
        assert len(topics) > 0
        assert all(isinstance(t, str) for t in topics)


# ===== KnowledgeIngestor Filter Pipeline =====


class TestIngestorFilter:
    """KnowledgeIngestor with DataFilter correctly gates content."""

    def test_filter_article_quality(self):
        filt = DataFilter({"min_quality_score": 0.5, "min_content_length": 100})
        good = "A B C. " * 30 + "Good content with proper writing and structure. " * 10
        passes, _ = filt.filter_article("http://good.com", "Good", good)
        assert passes is True

    def test_filter_rejects_low_quality(self):
        filt = DataFilter({"min_quality_score": 0.8, "min_content_length": 20})
        bad = "x" * 30
        passes, reason = filt.filter_article("http://bad.com", "Bad", bad)
        assert passes is False
        assert "quality" in reason

    def test_filter_rejects_blacklist(self):
        filt = DataFilter({"min_quality_score": 0.1, "min_content_length": 10})
        bad = "This article is about porn and gambling sites " * 10
        passes, reason = filt.filter_article("http://bad.com", "Bad", bad)
        assert passes is False
        assert reason == "blacklisted"

    def test_filter_tracks_stats(self):
        filt = DataFilter({"min_quality_score": 0.9, "min_content_length": 200})
        filt.filter_article("http://a.com", "A", "Short")
        filt.filter_article("http://b.com", "B", "x" * 500)
        stats = filt.get_stats()
        assert stats["total_seen"] >= 2
        assert stats["rejected"] >= 1
