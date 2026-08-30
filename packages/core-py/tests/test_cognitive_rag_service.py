"""Comprehensive tests for domains/cognitive/rag_service.py.

Covers: ProductionRAGWithRealEmbeddings, RAGService (ingestion, query,
persistence, KG integration, list, clear, stats), KGTrainingPipeline
(validation, submit, batch processing, sync), and singleton helpers.
All tests use tmp_path isolation; no external API mocks.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pytest

import domains.cognitive.rag_service as mod
from domains.cognitive.rag_service import (
    KGTrainingPipeline,
    ProductionRAGWithRealEmbeddings,
    RAGService,
    get_rag_service,
    is_rag_service_ready,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_rag_store(tmp_path):
    """Redirect RAG persistence to a temp directory for each test."""
    original_data_dir = mod._DATA_DIR
    original_docs_file = mod._DOCUMENTS_FILE
    mod._DATA_DIR = tmp_path
    mod._DOCUMENTS_FILE = tmp_path / "documents.jsonl"
    yield
    mod._DATA_DIR = original_data_dir
    mod._DOCUMENTS_FILE = original_docs_file


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the RAG service singleton between tests."""
    mod._rag_service = None
    yield
    mod._rag_service = None


@pytest.fixture()
def svc(tmp_path):
    """Create a fresh RAGService for a test."""
    return RAGService()


@pytest.fixture()
def populated_svc(svc):
    """RAGService with two documents already ingested."""
    svc.add_document(
        content="Python is a programming language created by Guido van Rossum in 1991.",
        metadata={"source": "wiki", "topic": "python"},
        chunk_size=200,
    )
    svc.add_document(
        content="SloughGPT is an AI assistant that runs locally using SloNet inference.",
        metadata={"source": "docs", "topic": "sloughgpt"},
        chunk_size=200,
    )
    return svc


# ---------------------------------------------------------------------------
# ProductionRAGWithRealEmbeddings
# ---------------------------------------------------------------------------

class TestProductionRAGWithRealEmbeddings:
    def test_init_sets_real_embedder(self):
        rag = ProductionRAGWithRealEmbeddings()
        assert hasattr(rag.retriever, "_get_embedding")
        assert callable(rag.retriever._get_embedding)

    def test_real_embed_returns_384_dim(self):
        vec = ProductionRAGWithRealEmbeddings._real_embed("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_real_embed_deterministic(self):
        a = ProductionRAGWithRealEmbeddings._real_embed("test input")
        b = ProductionRAGWithRealEmbeddings._real_embed("test input")
        np.testing.assert_array_equal(a, b)

    def test_real_embed_varies_by_input(self):
        a = ProductionRAGWithRealEmbeddings._real_embed("alpha")
        b = ProductionRAGWithRealEmbeddings._real_embed("bravo")
        assert not np.allclose(a, b)

    def test_add_and_query_with_real_embeddings(self):
        rag = ProductionRAGWithRealEmbeddings()
        rag.add_document(
            content="The quick brown fox jumps over the lazy dog.",
            metadata={"source": "test"},
        )
        result = rag.query("quick brown fox", top_k=1)
        assert result["num_results"] >= 1


# ---------------------------------------------------------------------------
# RAGService — __init__ & document loading
# ---------------------------------------------------------------------------

class TestRAGServiceInit:
    def test_creates_data_dir(self, tmp_path):
        mod._DATA_DIR = tmp_path / "sub" / "deep"
        mod._DOCUMENTS_FILE = mod._DATA_DIR / "documents.jsonl"
        RAGService()
        assert mod._DATA_DIR.exists()

    def test_loads_persisted_documents(self, tmp_path):
        docs_file = tmp_path / "documents.jsonl"
        record = {
            "content": "persisted doc content for loading",
            "metadata": {"source": "persist"},
            "chunk_size": 100,
            "overlap": 10,
        }
        docs_file.write_text(json.dumps(record) + "\n")
        mod._DOCUMENTS_FILE = docs_file
        mod._DATA_DIR = tmp_path
        svc = RAGService()
        assert len(svc._documents) == 1
        assert svc._documents[0]["metadata"]["source"] == "persist"

    def test_loads_skips_empty_lines(self, tmp_path):
        docs_file = tmp_path / "documents.jsonl"
        docs_file.write_text("\n\n\n")
        mod._DOCUMENTS_FILE = docs_file
        mod._DATA_DIR = tmp_path
        svc = RAGService()
        assert len(svc._documents) == 0

    def test_loads_skips_docs_without_content(self, tmp_path):
        docs_file = tmp_path / "documents.jsonl"
        record = {"metadata": {"source": "x"}, "chunk_size": 50}
        docs_file.write_text(json.dumps(record) + "\n")
        mod._DOCUMENTS_FILE = docs_file
        mod._DATA_DIR = tmp_path
        svc = RAGService()
        assert len(svc._documents) == 0

    def test_loads_handles_corrupt_json(self, tmp_path):
        docs_file = tmp_path / "documents.jsonl"
        docs_file.write_text("NOT JSON\n{bad\n")
        mod._DOCUMENTS_FILE = docs_file
        mod._DATA_DIR = tmp_path
        svc = RAGService()
        assert len(svc._documents) == 0

    def test_loads_missing_file_is_noop(self, tmp_path):
        mod._DOCUMENTS_FILE = tmp_path / "nonexistent.jsonl"
        mod._DATA_DIR = tmp_path
        svc = RAGService()
        assert len(svc._documents) == 0


# ---------------------------------------------------------------------------
# RAGService — add_document
# ---------------------------------------------------------------------------

class TestRAGServiceAddDocument:
    def test_returns_chunk_ids(self, svc):
        ids = svc.add_document(content="Hello world, this is a test.", metadata={"source": "t"})
        assert isinstance(ids, list)
        assert len(ids) >= 1

    def test_empty_content_returns_empty(self, svc):
        assert svc.add_document(content="") == []
        assert svc.add_document(content="   ") == []
        assert svc.add_document(content=None) == []

    def test_metadata_defaults_to_user(self, svc):
        svc.add_document(content="Some real content for default metadata test.")
        docs = svc.list_documents()
        assert docs[0]["metadata"]["source"] == "user"

    def test_custom_metadata_preserved(self, svc):
        svc.add_document(content="Some real content for custom metadata test.", metadata={"source": "custom"})
        docs = svc.list_documents()
        assert docs[0]["metadata"]["source"] == "custom"

    def test_chunk_size_and_overlap_passed(self, svc):
        ids = svc.add_document(
            content="word " * 100,
            metadata={"source": "x"},
            chunk_size=10,
            overlap=2,
        )
        assert len(ids) >= 1

    def test_document_record_added(self, svc):
        svc.add_document(content="Test doc for record tracking.", metadata={"source": "rec"})
        assert len(svc._documents) == 1
        doc = svc._documents[0]
        assert doc["metadata"]["source"] == "rec"
        assert "chunk_ids" in doc
        assert "added_at" in doc
        assert doc["added_at"] > 0

    def test_persistence_file_written(self, svc, tmp_path):
        svc.add_document(content="Persist me please.", metadata={"source": "file"})
        assert mod._DOCUMENTS_FILE.exists()
        lines = mod._DOCUMENTS_FILE.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["metadata"]["source"] == "file"

    def test_multiple_documents_accumulated(self, svc):
        svc.add_document(content="First document for accumulation test.", metadata={"source": "a"})
        svc.add_document(content="Second document for accumulation test.", metadata={"source": "b"})
        assert len(svc._documents) == 2

    def test_thread_safety_concurrent_adds(self, svc):
        errors = []

        def add_doc(i):
            try:
                svc.add_document(
                    content=f"Concurrent document number {i} with enough content.",
                    metadata={"source": f"t{i}"},
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_doc, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(svc._documents) == 10


# ---------------------------------------------------------------------------
# RAGService — query
# ---------------------------------------------------------------------------

class TestRAGServiceQuery:
    def test_query_returns_expected_keys(self, populated_svc):
        result = populated_svc.query("Python programming")
        assert "context" in result
        assert "results" in result
        assert "num_results" in result

    def test_query_returns_results_for_relevant_question(self, populated_svc):
        result = populated_svc.query("What is SloughGPT?")
        assert result["num_results"] >= 1

    def test_query_context_is_string(self, populated_svc):
        result = populated_svc.query("AI assistant")
        assert isinstance(result["context"], str)

    def test_query_top_k_limits_results(self, populated_svc):
        result = populated_svc.query("programming language", top_k=1)
        assert result["num_results"] <= 1

    def test_query_results_have_metadata(self, populated_svc):
        result = populated_svc.query("Python")
        for r in result["results"]:
            assert "chunk_id" in r
            assert "content" in r
            assert "score" in r
            assert "rank" in r
            assert "metadata" in r

    def test_query_on_empty_index(self, svc):
        result = svc.query("anything at all")
        assert result["num_results"] == 0
        assert result["context"] == ""


# ---------------------------------------------------------------------------
# RAGService — verify_and_ground
# ---------------------------------------------------------------------------

class TestRAGServiceVerifyAndGround:
    def test_returns_expected_keys(self, populated_svc):
        result = populated_svc.verify_and_ground(
            "Python is a programming language.",
            "What is Python?",
        )
        assert "verification" in result
        assert "citations" in result
        assert "confidence" in result
        assert "is_verified" in result

    def test_confidence_is_numeric(self, populated_svc):
        result = populated_svc.verify_and_ground(
            "Python is a programming language.",
            "What is Python?",
        )
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_is_verified_is_bool(self, populated_svc):
        result = populated_svc.verify_and_ground(
            "Python is a programming language.",
            "What is Python?",
        )
        assert isinstance(result["is_verified"], bool)

    def test_verification_has_claims(self, populated_svc):
        result = populated_svc.verify_and_ground(
            "Python is a programming language.",
            "What is Python?",
        )
        v = result["verification"]
        assert "grounded_claims" in v
        assert "hallucinations" in v
        assert "hallucination_rate" in v


# ---------------------------------------------------------------------------
# RAGService — list_documents
# ---------------------------------------------------------------------------

class TestRAGServiceListDocuments:
    def test_empty_when_no_docs(self, svc):
        assert svc.list_documents() == []

    def test_lists_all_documents(self, populated_svc):
        docs = populated_svc.list_documents()
        assert len(docs) == 2

    def test_list_excludes_content(self, populated_svc):
        docs = populated_svc.list_documents()
        for d in docs:
            assert "content" not in d

    def test_list_has_expected_keys(self, populated_svc):
        docs = populated_svc.list_documents()
        for d in docs:
            assert "metadata" in d
            assert "chunk_size" in d
            assert "num_chunks" in d
            assert "added_at" in d

    def test_num_chunks_matches(self, populated_svc):
        docs = populated_svc.list_documents()
        for d in docs:
            assert d["num_chunks"] >= 1


# ---------------------------------------------------------------------------
# RAGService — clear
# ---------------------------------------------------------------------------

class TestRAGServiceClear:
    def test_clear_returns_count(self, populated_svc):
        count = populated_svc.clear()
        assert count == 2

    def test_clear_empties_documents(self, populated_svc):
        populated_svc.clear()
        assert len(populated_svc._documents) == 0
        assert populated_svc.stats()["total_documents"] == 0

    def test_clear_removes_persistence_file(self, populated_svc):
        populated_svc.clear()
        assert not mod._DOCUMENTS_FILE.exists()

    def test_clear_on_empty_returns_zero(self, svc):
        assert svc.clear() == 0

    def test_can_add_after_clear(self, populated_svc):
        populated_svc.clear()
        populated_svc.add_document(content="After clear document for re-add test.", metadata={"source": "x"})
        assert populated_svc.stats()["total_documents"] == 1


# ---------------------------------------------------------------------------
# RAGService — stats
# ---------------------------------------------------------------------------

class TestRAGServiceStats:
    def test_empty_stats(self, svc):
        s = svc.stats()
        assert s["total_documents"] == 0
        assert s["total_chunks"] == 0
        assert s["index_size"] == 0

    def test_stats_after_ingest(self, populated_svc):
        s = populated_svc.stats()
        assert s["total_documents"] == 2
        assert s["total_chunks"] >= 2


# ---------------------------------------------------------------------------
# RAGService — persistence across instances
# ---------------------------------------------------------------------------

class TestRAGServicePersistence:
    def test_documents_survive_recreation(self, svc):
        svc.add_document(content="Persistent content for restart test.", metadata={"source": "persist"})
        # Simulate restart
        svc2 = RAGService()
        assert svc2.stats()["total_documents"] == 1
        docs = svc2.list_documents()
        assert docs[0]["metadata"]["source"] == "persist"

    def test_multiple_documents_persist(self, svc):
        svc.add_document(content="First document for multi-persist test.", metadata={"source": "a"})
        svc.add_document(content="Second document for multi-persist test.", metadata={"source": "b"})
        svc2 = RAGService()
        assert svc2.stats()["total_documents"] == 2

    def test_persistence_file_is_jsonl(self, svc):
        svc.add_document(content="JSONL format verification document.", metadata={"source": "jsonl"})
        content = mod._DOCUMENTS_FILE.read_text()
        lines = content.strip().split("\n")
        for line in lines:
            obj = json.loads(line)
            assert "content" in obj


# ---------------------------------------------------------------------------
# RAGService — KG integration
# ---------------------------------------------------------------------------

class TestRAGServiceKGIntegration:
    def test_kg_stats_empty_when_no_kg(self, svc):
        assert svc._kg is None
        s = svc.kg_stats()
        assert s["entities"] == 0
        assert s["facts"] == 0
        assert s["avg_degree"] == 0.0

    def test_kg_query_empty_when_no_kg(self, svc):
        assert svc.kg_query(subject="anything") == []

    def test_ensure_kg_creates_kg(self, svc):
        svc._ensure_kg()
        assert svc._kg is not None

    def test_kg_stats_after_kg_init(self, svc):
        svc._ensure_kg()
        s = svc.kg_stats()
        assert s["entities"] == 0
        assert s["facts"] == 0

    def test_kg_query_filters_by_subject(self, svc):
        svc._ensure_kg()
        svc._kg.add_fact(subject="Python", predicate="is_a", obj="language")
        svc._kg.add_fact(subject="Java", predicate="is_a", obj="language")
        results = svc.kg_query(subject="Python")
        assert len(results) == 1
        assert results[0]["subject"] == "Python"

    def test_kg_query_filters_by_predicate(self, svc):
        svc._ensure_kg()
        svc._kg.add_fact(subject="Python", predicate="is_a", obj="language")
        svc._kg.add_fact(subject="Python", predicate="created_by", obj="Guido")
        results = svc.kg_query(predicate="created_by")
        assert len(results) == 1
        assert results[0]["predicate"] == "created_by"

    def test_kg_query_filters_by_object(self, svc):
        svc._ensure_kg()
        svc._kg.add_fact(subject="Python", predicate="is_a", obj="language")
        svc._kg.add_fact(subject="Python", predicate="is_a", obj="tool")
        results = svc.kg_query(obj="tool")
        assert len(results) == 1
        assert results[0]["object"] == "tool"

    def test_kg_query_case_insensitive(self, svc):
        svc._ensure_kg()
        svc._kg.add_fact(subject="Python", predicate="is_a", obj="language")
        results = svc.kg_query(subject="python")
        assert len(results) == 1

    def test_kg_query_no_filters_returns_all(self, svc):
        svc._ensure_kg()
        svc._kg.add_fact(subject="A", predicate="p1", obj="B")
        svc._kg.add_fact(subject="C", predicate="p2", obj="D")
        results = svc.kg_query()
        assert len(results) == 2

    def test_extract_kg_claims_with_no_detector(self, svc):
        svc._ensure_kg()
        svc._extract_kg_claims("Python is a language.", {"source": "test"})
        # Should not raise, hallucination detector may not have extract_claims
        # The method catches AttributeError gracefully


# ---------------------------------------------------------------------------
# RAGService — auto_ingest_directory
# ---------------------------------------------------------------------------

class TestRAGServiceAutoIngestDirectory:
    def test_returns_zero_when_repo_scanner_unavailable(self, svc):
        with patch.dict("sys.modules", {"domains.infrastructure.auto_ingest": None}):
            result = svc.auto_ingest_directory("/tmp/nonexistent")
            assert result == 0

    def test_returns_zero_for_empty_directory(self, svc, tmp_path):
        result = svc.auto_ingest_directory(str(tmp_path))
        assert result == 0

    def test_ingests_files_in_directory(self, svc, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    return 'world'\n")
        (tmp_path / "b.txt").write_text("This is a text file with enough content to pass the threshold.")
        result = svc.auto_ingest_directory(str(tmp_path), max_files=10)
        assert result >= 1

    def test_max_files_limit(self, svc, tmp_path):
        for i in range(5):
            (tmp_path / f"doc_{i}.txt").write_text(f"Document {i} with sufficient content for ingestion test.")
        result = svc.auto_ingest_directory(str(tmp_path), max_files=2)
        assert result <= 2

    def test_skips_short_files(self, svc, tmp_path):
        (tmp_path / "short.txt").write_text("hi")
        (tmp_path / "long.txt").write_text("x" * 100)
        result = svc.auto_ingest_directory(str(tmp_path))
        assert result == 1


# ---------------------------------------------------------------------------
# KGTrainingPipeline — _validate_triple
# ---------------------------------------------------------------------------

class TestKGTrainingPipelineValidateTriple:
    def test_valid_triple(self):
        pipe = KGTrainingPipeline(rag_service=RAGService())
        assert pipe._validate_triple({"subject": "A", "predicate": "p", "object": "B"}) is True

    def test_missing_subject(self):
        pipe = KGTrainingPipeline(rag_service=RAGService())
        assert pipe._validate_triple({"predicate": "p", "object": "B"}) is False

    def test_missing_predicate(self):
        pipe = KGTrainingPipeline(rag_service=RAGService())
        assert pipe._validate_triple({"subject": "A", "object": "B"}) is False

    def test_missing_object(self):
        pipe = KGTrainingPipeline(rag_service=RAGService())
        assert pipe._validate_triple({"subject": "A", "predicate": "p"}) is False

    def test_empty_string_subject(self):
        pipe = KGTrainingPipeline(rag_service=RAGService())
        assert pipe._validate_triple({"subject": "", "predicate": "p", "object": "B"}) is False

    def test_whitespace_only_subject(self):
        pipe = KGTrainingPipeline(rag_service=RAGService())
        assert pipe._validate_triple({"subject": "   ", "predicate": "p", "object": "B"}) is False

    def test_non_string_value(self):
        pipe = KGTrainingPipeline(rag_service=RAGService())
        assert pipe._validate_triple({"subject": 123, "predicate": "p", "object": "B"}) is False

    def test_not_a_dict(self):
        pipe = KGTrainingPipeline(rag_service=RAGService())
        assert pipe._validate_triple("not a dict") is False
        assert pipe._validate_triple([1, 2, 3]) is False
        assert pipe._validate_triple(None) is False

    def test_extra_keys_ignored(self):
        pipe = KGTrainingPipeline(rag_service=RAGService())
        assert pipe._validate_triple({
            "subject": "A", "predicate": "p", "object": "B",
            "confidence": 0.9, "source": "test"
        }) is True


# ---------------------------------------------------------------------------
# KGTrainingPipeline — submit_triples
# ---------------------------------------------------------------------------

class TestKGTrainingPipelineSubmitTriples:
    def test_raises_on_empty_list(self):
        pipe = KGTrainingPipeline(rag_service=RAGService())
        with pytest.raises(ValueError, match="must not be empty"):
            pipe.submit_triples([])

    def test_submit_valid_triples(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        triples = [
            {"subject": "Python", "predicate": "is_a", "object": "language"},
            {"subject": "Java", "predicate": "is_a", "object": "language"},
        ]
        count = pipe.submit_triples(triples)
        assert count == 2

    def test_submit_skips_invalid_triples(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        triples = [
            {"subject": "Python", "predicate": "is_a", "object": "language"},
            {"bad": "triple"},
            {"subject": "Java", "predicate": "is_a", "object": "language"},
        ]
        count = pipe.submit_triples(triples)
        assert count == 2

    def test_submit_single_triple(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        count = pipe.submit_triples([{"subject": "A", "predicate": "p", "object": "B"}])
        assert count == 1

    def test_task_data_structure(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        pipe.submit_triples([{
            "subject": "X",
            "predicate": "relates_to",
            "object": "Y",
            "confidence": 0.9,
            "source": "kg",
        }])
        queue = pipe._get_queue()
        # Task should be pending
        assert len(queue._pending) >= 1


# ---------------------------------------------------------------------------
# KGTrainingPipeline — process_batch
# ---------------------------------------------------------------------------

class TestKGTrainingPipelineProcessBatch:
    def test_process_batch_with_no_tasks(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        result = pipe.process_batch(max_tasks=10)
        assert result["processed"] == 0
        assert result["failed"] == 0

    def test_process_batch_zero_max_returns_remaining(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        pipe.submit_triples([{"subject": "A", "predicate": "p", "object": "B"}])
        result = pipe.process_batch(max_tasks=0)
        assert result["processed"] == 0
        assert result["remaining"] >= 1

    def test_process_batch_processes_tasks(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        pipe.submit_triples([
            {"subject": "A", "predicate": "p", "object": "B"},
            {"subject": "C", "predicate": "q", "object": "D"},
        ])
        result = pipe.process_batch(max_tasks=10)
        assert result["processed"] == 2
        assert result["failed"] == 0
        assert result["remaining"] == 0

    def test_process_batch_respects_max_tasks(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        pipe.submit_triples([
            {"subject": "A", "predicate": "p", "object": "B"},
            {"subject": "C", "predicate": "q", "object": "D"},
            {"subject": "E", "predicate": "r", "object": "F"},
        ])
        result = pipe.process_batch(max_tasks=1)
        assert result["processed"] == 1
        assert result["remaining"] == 2

    def test_process_batch_indexes_into_rag(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        pipe.submit_triples([{
            "subject": "Python",
            "predicate": "is_a",
            "object": "language",
        }])
        pipe.process_batch(max_tasks=5)
        assert svc.stats()["total_documents"] >= 1


# ---------------------------------------------------------------------------
# KGTrainingPipeline — sync_kg_to_rag
# ---------------------------------------------------------------------------

class TestKGTrainingPipelineSyncKgToRag:
    def test_sync_with_no_kg(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        result = pipe.sync_kg_to_rag()
        assert result["total_triples"] == 0
        assert result["processed"] == 0

    def test_sync_with_empty_kg(self):
        svc = RAGService()
        svc._ensure_kg()
        pipe = KGTrainingPipeline(rag_service=svc)
        result = pipe.sync_kg_to_rag(kg=svc._kg)
        assert result["total_triples"] == 0
        assert result["processed"] == 0

    def test_sync_with_populated_kg(self):
        svc = RAGService()
        svc._ensure_kg()
        svc._kg.add_fact(subject="Python", predicate="is_a", obj="language")
        svc._kg.add_fact(subject="Java", predicate="is_a", obj="language")
        pipe = KGTrainingPipeline(rag_service=svc)
        result = pipe.sync_kg_to_rag(kg=svc._kg)
        assert result["total_triples"] == 2
        assert result["processed"] == 2
        assert result["failed"] == 0

    def test_sync_uses_rag_internal_kg(self):
        svc = RAGService()
        svc._ensure_kg()
        svc._kg.add_fact(subject="Test", predicate="p", obj="Value")
        pipe = KGTrainingPipeline(rag_service=svc)
        result = pipe.sync_kg_to_rag()
        assert result["total_triples"] == 1


# ---------------------------------------------------------------------------
# KGTrainingPipeline — stats
# ---------------------------------------------------------------------------

class TestKGTrainingPipelineStats:
    def test_stats_empty(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        s = pipe.stats()
        assert s["pending"] == 0
        assert s["running"] == 0
        assert s["completed"] == 0

    def test_stats_after_submit(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        pipe.submit_triples([{"subject": "A", "predicate": "p", "object": "B"}])
        s = pipe.stats()
        assert s["pending"] == 1

    def test_stats_after_process(self):
        svc = RAGService()
        pipe = KGTrainingPipeline(rag_service=svc)
        pipe.submit_triples([{"subject": "A", "predicate": "p", "object": "B"}])
        pipe.process_batch(max_tasks=5)
        s = pipe.stats()
        assert s["pending"] == 0
        assert s["completed"] >= 1


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:
    def test_is_rag_service_ready_false_by_default(self):
        assert is_rag_service_ready() is False

    def test_is_rag_service_ready_true_after_init(self):
        get_rag_service()
        assert is_rag_service_ready() is True

    def test_get_rag_service_returns_same_instance(self):
        a = get_rag_service()
        b = get_rag_service()
        assert a is b

    def test_get_rag_service_thread_safety(self):
        instances = []
        errors = []

        def grab():
            try:
                instances.append(get_rag_service())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=grab) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert all(i is instances[0] for i in instances)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_add_document_with_unicode(self, svc):
        ids = svc.add_document(content="日本語のテストドキュメント。Unicode content.", metadata={"source": "unicode"})
        assert len(ids) >= 1

    def test_add_document_with_newlines(self, svc):
        ids = svc.add_document(
            content="Line one.\nLine two.\nLine three.",
            metadata={"source": "newline"},
        )
        assert len(ids) >= 1

    def test_query_with_special_characters(self, populated_svc):
        result = populated_svc.query("What is @#$%^&*()?")
        assert "num_results" in result

    def test_list_documents_returns_copy(self, svc):
        svc.add_document(content="Original document content for copy test.", metadata={"source": "x"})
        docs1 = svc.list_documents()
        docs1.clear()
        docs2 = svc.list_documents()
        assert len(docs2) == 1

    def test_stats_after_clear_and_readd(self, svc):
        svc.add_document(content="Before clear document.", metadata={"source": "a"})
        svc.clear()
        svc.add_document(content="After clear document.", metadata={"source": "b"})
        s = svc.stats()
        assert s["total_documents"] == 1

    def test_rag_service_with_config(self):
        rag = ProductionRAGWithRealEmbeddings(config={"dense_weight": 0.5, "sparse_weight": 0.5})
        assert rag.retriever.dense_weight == 0.5
        assert rag.retriever.sparse_weight == 0.5

    def test_kg_training_pipeline_default_rag_service(self):
        pipe = KGTrainingPipeline()
        assert pipe._rag is not None
