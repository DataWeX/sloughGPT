"""Tests for KG → RAG training pipeline."""

import pytest
from domains.cognitive.knowledge_graph_v2 import KnowledgeGraph
from domains.cognitive.rag_service import RAGService, KGTrainingPipeline


@pytest.fixture
def kg():
    """Create a KG with test triples."""
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


class TestKGExportTriples:
    def test_export_triples_count(self, kg):
        triples = kg.export_triples()
        assert len(triples) == 3

    def test_export_triples_fields(self, kg):
        triples = kg.export_triples()
        for t in triples:
            assert "subject" in t
            assert "predicate" in t
            assert "object" in t
            assert "confidence" in t
            assert "source" in t

    def test_export_triples_content(self, kg):
        triples = kg.export_triples()
        subjects = {t["subject"] for t in triples}
        # add_fact resolves to existing entity IDs (lowercase from fixture)
        assert "paris" in subjects
        assert "london" in subjects


class TestKGTrainingPipeline:
    def test_submit_triples(self, pipeline, kg):
        triples = kg.export_triples()
        count = pipeline.submit_triples(triples)
        assert count == 3

    def test_process_batch(self, pipeline, kg):
        triples = kg.export_triples()
        pipeline.submit_triples(triples)
        result = pipeline.process_batch(max_tasks=10)
        assert result["processed"] == 3
        assert result["failed"] == 0
        assert result["remaining"] == 0

    def test_sync_kg_to_rag(self, pipeline, kg):
        result = pipeline.sync_kg_to_rag(kg=kg)
        assert result["total_triples"] == 3
        assert result["processed"] == 3
        assert result["failed"] == 0

    def test_sync_empty_kg(self, pipeline):
        empty_kg = KnowledgeGraph()
        result = pipeline.sync_kg_to_rag(kg=empty_kg)
        assert result["total_triples"] == 0

    def test_sync_no_kg(self, pipeline, rag_svc):
        result = pipeline.sync_kg_to_rag()
        assert result["total_triples"] == 0

    def test_rag_retrieves_kg_triples(self, pipeline, rag_svc, kg):
        pipeline.sync_kg_to_rag(kg=kg)
        query_result = rag_svc.query("capital of France", top_k=3)
        results = query_result.get("results", [])
        assert len(results) > 0
        # Top result should be the KG triple (entity IDs are lowercase)
        top = results[0]
        assert "capital_of" in top["content"]
        assert "paris" in top["content"]

    def test_kg_triple_metadata(self, pipeline, rag_svc, kg):
        pipeline.sync_kg_to_rag(kg=kg)
        query_result = rag_svc.query("capital of France", top_k=1)
        results = query_result.get("results", [])
        assert len(results) > 0
        meta = results[0].get("metadata", {})
        assert meta.get("kg_triple") is True
        assert meta.get("subject") == "paris"
        assert meta.get("predicate") == "capital_of"

    def test_pipeline_stats(self, pipeline, kg):
        pipeline.submit_triples(kg.export_triples())
        stats = pipeline.stats()
        assert stats["pending"] == 3
        assert stats["running"] == 0

    def test_process_batch_partial(self, pipeline, kg):
        pipeline.submit_triples(kg.export_triples())
        result = pipeline.process_batch(max_tasks=2)
        assert result["processed"] == 2
        assert result["remaining"] == 1

    def test_process_batch_empty(self, pipeline):
        result = pipeline.process_batch()
        assert result["processed"] == 0
        assert result["remaining"] == 0
