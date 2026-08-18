"""Tests for KG → RAG pipeline, KnowledgeGraph v2, and RAGService integration."""

import json
import threading
import time
from pathlib import Path
from typing import List

import pytest

from domains.cognitive.knowledge_graph_v2 import Entity, Fact, KnowledgeGraph
from domains.cognitive.rag_service import (
    RAGService,
    KGTrainingPipeline,
    get_rag_service,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kg():
    """Create a KG with test entities and triples."""
    g = KnowledgeGraph()
    g.add_entity("paris", "Paris", "city")
    g.add_entity("france", "France", "country")
    g.add_entity("london", "London", "city")
    g.add_entity("uk", "UK", "country")
    g.add_fact("Paris", "capital_of", "France", 0.95, "test")
    g.add_fact("London", "capital_of", "UK", 0.95, "test")
    g.add_fact("Paris", "larger_than", "London", 0.7, "test")
    return g


@pytest.fixture
def rag_svc(tmp_path, monkeypatch):
    """Create a fresh RAGService with temp persistence."""
    from domains.cognitive import rag_service

    monkeypatch.setattr(rag_service, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(rag_service, "_DOCUMENTS_FILE", tmp_path / "docs.jsonl")
    svc = RAGService()
    return svc


@pytest.fixture
def pipeline(rag_svc):
    """Create a KGTrainingPipeline."""
    return KGTrainingPipeline(rag_service=rag_svc)


# ---------------------------------------------------------------------------
# Entity Resolution (KnowledgeGraph v2)
# ---------------------------------------------------------------------------


class TestKGEntityResolution:
    """Case-insensitive entity lookup, dedup, equality."""

    def test_resolve_existing_entity(self):
        g = KnowledgeGraph()
        g.add_entity("paris", "Paris", "city")
        assert g._resolve_entity_id("PARIS") == "paris"
        assert g._resolve_entity_id("Paris") == "paris"
        assert g._resolve_entity_id("paris") == "paris"

    def test_resolve_nonexistent_entity(self):
        g = KnowledgeGraph()
        assert g._resolve_entity_id("London") == "London"

    def test_add_entity_dedup_by_resolved_id(self):
        g = KnowledgeGraph()
        e1 = g.add_entity("paris", "Paris", "city")
        e2 = g.add_entity("PARIS", "Paris", "city")
        assert e1 is e2
        assert len(g.entities) == 1

    def test_add_entity_preserves_aliases(self):
        g = KnowledgeGraph()
        g.add_entity("paris", "Paris", "city", aliases=["city of light"])
        e2 = g.add_entity("Paris", "Paris", "city")
        assert "city of light" in e2.aliases

    def test_add_fact_dedup_same_triple(self):
        g = KnowledgeGraph()
        g.add_entity("paris", "Paris", "city")
        g.add_entity("france", "France", "country")
        f1 = g.add_fact("Paris", "capital_of", "France", 0.95, "test")
        f2 = g.add_fact("paris", "capital_of", "france", 0.8, "other")
        assert f2 is None
        assert len(g.facts) == 1

    def test_add_fact_higher_confidence_wins(self):
        g = KnowledgeGraph()
        g.add_entity("paris", "Paris", "city")
        g.add_entity("france", "France", "country")
        g.add_fact("Paris", "capital_of", "France", 0.7, "old")
        g.add_fact("Paris", "capital_of", "France", 0.99, "new")
        fact = list(g.facts.values())[0]
        assert fact.confidence == 0.99
        assert fact.source == "new"

    def test_add_fact_auto_creates_entities(self):
        g = KnowledgeGraph()
        f = g.add_fact("Tokyo", "capital_of", "Japan", 0.95, "test")
        assert f is not None
        assert "tokyo" in g.entities or "Tokyo" in g.entities
        assert "japan" in g.entities or "Japan" in g.entities

    def test_entity_equality(self):
        a = Entity(id="paris", label="Paris", entity_type="city")
        b = Entity(id="paris", label="Paris", entity_type="city")
        c = Entity(id="london", label="London", entity_type="city")
        assert a == b
        assert a != c
        assert a != "not an entity"

    def test_case_insensitive_query(self):
        g = KnowledgeGraph()
        g.add_entity("paris", "Paris", "city")
        g.add_fact("Paris", "capital_of", "France", 0.95, "test")
        results = g.query(subject="PARIS")
        assert len(results) == 1
        assert results[0].subject == "paris"

    def test_case_insensitive_query_no_preexisting(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France", 0.95, "test")
        results = g.query(subject="Paris")
        assert len(results) == 1
        assert results[0].subject == "Paris"

    def test_case_insensitive_get_outgoing(self):
        g = KnowledgeGraph()
        g.add_entity("paris", "Paris", "city")
        g.add_entity("france", "France", "country")
        g.add_fact("Paris", "capital_of", "France", 0.95, "test")
        out = g.get_outgoing("PARIS")
        assert len(out) == 1
        assert out[0] == ("capital_of", "france")

    def test_case_insensitive_get_incoming(self):
        g = KnowledgeGraph()
        g.add_entity("paris", "Paris", "city")
        g.add_entity("france", "France", "country")
        g.add_fact("Paris", "capital_of", "France", 0.95, "test")
        inc = g.get_incoming("FRANCE")
        assert len(inc) == 1
        assert inc[0] == ("capital_of", "paris")

    def test_get_outgoing_no_entity(self):
        g = KnowledgeGraph()
        assert g.get_outgoing("nonexistent") == []

    def test_get_incoming_no_entity(self):
        g = KnowledgeGraph()
        assert g.get_incoming("nonexistent") == []

    def test_get_outgoing_filtered(self):
        g = KnowledgeGraph()
        g.add_entity("paris", "Paris", "city")
        g.add_entity("france", "France", "country")
        g.add_entity("london", "London", "city")
        g.add_fact("Paris", "capital_of", "France", 0.95, "test")
        g.add_fact("Paris", "larger_than", "London", 0.7, "test")
        out = g.get_outgoing("paris", predicate="capital_of")
        assert len(out) == 1
        assert out[0][0] == "capital_of"


# ---------------------------------------------------------------------------
# Export Triples
# ---------------------------------------------------------------------------


class TestKGExportTriples:
    def test_export_triples_count(self, kg):
        triples = kg.export_triples()
        assert len(triples) == 3

    def test_export_triples_fields(self, kg):
        for t in kg.export_triples():
            assert set(t.keys()) == {"subject", "predicate", "object", "confidence", "source"}

    def test_export_triples_content(self, kg):
        subjects = {t["subject"] for t in kg.export_triples()}
        assert "paris" in subjects
        assert "london" in subjects


# ---------------------------------------------------------------------------
# Triple Validation (KGTrainingPipeline)
# ---------------------------------------------------------------------------


class TestTripleValidation:
    def test_valid_triple(self, pipeline):
        assert pipeline._validate_triple({"subject": "a", "predicate": "b", "object": "c"})

    def test_missing_key(self, pipeline):
        assert not pipeline._validate_triple({"subject": "a", "object": "c"})

    def test_empty_string_value(self, pipeline):
        assert not pipeline._validate_triple({"subject": "a", "predicate": "", "object": "c"})

    def test_whitespace_only_value(self, pipeline):
        assert not pipeline._validate_triple({"subject": "a", "predicate": "  ", "object": "c"})

    def test_non_string_value(self, pipeline):
        assert not pipeline._validate_triple({"subject": "a", "predicate": 123, "object": "c"})

    def test_non_dict_input(self, pipeline):
        assert not pipeline._validate_triple("not a dict")
        assert not pipeline._validate_triple(None)
        assert not pipeline._validate_triple([1, 2, 3])

    def test_extra_keys_ignored(self, pipeline):
        assert pipeline._validate_triple(
            {"subject": "a", "predicate": "b", "object": "c", "extra": True}
        )


# ---------------------------------------------------------------------------
# KGTrainingPipeline
# ---------------------------------------------------------------------------


class TestKGTrainingPipeline:
    def test_submit_triples(self, pipeline, kg):
        triples = kg.export_triples()
        count = pipeline.submit_triples(triples)
        assert count == 3

    def test_submit_empty_list_raises(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.submit_triples([])

    def test_submit_invalid_triple_skipped(self, pipeline):
        result = pipeline.submit_triples([
            {"subject": "a", "predicate": "b", "object": "c"},
            {"bad": "triple"},
            {"subject": "d", "predicate": "e", "object": "f"},
        ])
        assert result == 2

    def test_process_batch(self, pipeline, kg):
        pipeline.submit_triples(kg.export_triples())
        result = pipeline.process_batch(max_tasks=10)
        assert result["processed"] == 3
        assert result["failed"] == 0
        assert result["remaining"] == 0

    def test_process_batch_partial(self, pipeline, kg):
        pipeline.submit_triples(kg.export_triples())
        result = pipeline.process_batch(max_tasks=2)
        assert result["processed"] == 2
        assert result["remaining"] == 1

    def test_process_batch_zero_max(self, pipeline, kg):
        pipeline.submit_triples(kg.export_triples())
        result = pipeline.process_batch(max_tasks=0)
        assert result["processed"] == 0
        assert result["remaining"] == 3

    def test_process_batch_empty_queue(self, pipeline):
        result = pipeline.process_batch()
        assert result["processed"] == 0
        assert result["remaining"] == 0

    def test_sync_kg_to_rag(self, pipeline, kg):
        result = pipeline.sync_kg_to_rag(kg=kg)
        assert result["total_triples"] == 3
        assert result["processed"] == 3
        assert result["failed"] == 0

    def test_sync_empty_kg(self, pipeline):
        result = pipeline.sync_kg_to_rag(kg=KnowledgeGraph())
        assert result["total_triples"] == 0

    def test_sync_no_kg(self, pipeline):
        result = pipeline.sync_kg_to_rag()
        assert result["total_triples"] == 0

    def test_pipeline_stats(self, pipeline, kg):
        pipeline.submit_triples(kg.export_triples())
        stats = pipeline.stats()
        assert stats["pending"] == 3
        assert stats["running"] == 0


# ---------------------------------------------------------------------------
# RAGService Integration
# ---------------------------------------------------------------------------


class TestRAGServiceIntegration:
    def test_add_document(self, rag_svc):
        ids = rag_svc.add_document("Test document content for RAG indexing.")
        assert len(ids) > 0
        assert rag_svc.stats()["total_documents"] == 1

    def test_add_empty_content(self, rag_svc):
        assert rag_svc.add_document("") == []
        assert rag_svc.add_document("   ") == []
        assert rag_svc.add_document(None) == []

    def test_add_document_no_metadata(self, rag_svc):
        rag_svc.add_document("Some content.")
        docs = rag_svc.list_documents()
        assert docs[0]["metadata"] == {"source": "user"}

    def test_list_documents(self, rag_svc):
        rag_svc.add_document("First doc.")
        rag_svc.add_document("Second doc.")
        docs = rag_svc.list_documents()
        assert len(docs) == 2
        assert all("metadata" in d for d in docs)

    def test_clear(self, rag_svc):
        rag_svc.add_document("Some content.")
        count = rag_svc.clear()
        assert count == 1
        assert rag_svc.stats()["total_documents"] == 0

    def test_stats(self, rag_svc):
        s = rag_svc.stats()
        assert "total_documents" in s
        assert "total_chunks" in s
        assert "index_size" in s

    def test_query_returns_results(self, rag_svc):
        rag_svc.add_document("Paris is the capital of France.")
        result = rag_svc.query("capital of France")
        assert "results" in result
        assert len(result["results"]) > 0

    def test_kg_stats_empty(self, rag_svc):
        stats = rag_svc.kg_stats()
        assert stats == {"entities": 0, "facts": 0, "avg_degree": 0.0}

    def test_kg_query_empty(self, rag_svc):
        results = rag_svc.kg_query(subject="anything")
        assert results == []


# ---------------------------------------------------------------------------
# RAG + KG Pipeline Integration
# ---------------------------------------------------------------------------


class TestRAGKGIntegration:
    def test_rag_retrieves_kg_triples(self, pipeline, rag_svc, kg):
        pipeline.sync_kg_to_rag(kg=kg)
        query_result = rag_svc.query("capital of France", top_k=3)
        results = query_result.get("results", [])
        assert len(results) > 0
        assert "capital_of" in results[0]["content"]

    def test_kg_triple_metadata(self, pipeline, rag_svc, kg):
        pipeline.sync_kg_to_rag(kg=kg)
        query_result = rag_svc.query("capital of France", top_k=1)
        results = query_result.get("results", [])
        meta = results[0].get("metadata", {})
        assert meta.get("kg_triple") is True
        assert meta.get("subject") == "paris"
        assert meta.get("predicate") == "capital_of"


# ---------------------------------------------------------------------------
# RAGService Thread Safety
# ---------------------------------------------------------------------------


class TestRAGServiceConcurrency:
    def test_concurrent_add_document(self, rag_svc):
        errors = []

        def add_doc(i):
            try:
                rag_svc.add_document(f"Document {i} with enough content for indexing.")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_doc, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert rag_svc.stats()["total_documents"] == 10

    def test_concurrent_list_documents(self, rag_svc):
        rag_svc.add_document("Base document for concurrent reads.")
        errors = []
        results = []

        def read_docs():
            try:
                docs = rag_svc.list_documents()
                results.append(len(docs))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_docs) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert all(r == 1 for r in results)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestRAGServiceSingleton:
    def test_get_rag_service_returns_same_instance(self, tmp_path, monkeypatch):
        from domains.cognitive import rag_service

        monkeypatch.setattr(rag_service, "_rag_service", None)
        monkeypatch.setattr(rag_service, "_DATA_DIR", tmp_path)
        monkeypatch.setattr(rag_service, "_DOCUMENTS_FILE", tmp_path / "docs.jsonl")
        s1 = get_rag_service()
        s2 = get_rag_service()
        assert s1 is s2
