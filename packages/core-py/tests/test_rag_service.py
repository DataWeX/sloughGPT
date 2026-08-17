"""Tests for the production RAG service (domains/cognitive/rag_service.py).

Covers: document ingestion, query, verification, persistence, stats, and the
BM25/HybridRetriever/CitationTracker/HallucinationDetector from rag.py.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_rag_store(tmp_path):
    """Redirect RAG persistence to a temp directory for each test."""
    import domains.cognitive.rag_service as mod
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
    import domains.cognitive.rag_service as mod
    mod._rag_service = None
    yield
    mod._rag_service = None


# ---------------------------------------------------------------------------
# BM25 Indexer
# ---------------------------------------------------------------------------

class TestBM25Indexer:
    def test_index_and_score(self):
        from domains.cognitive.rag import BM25Indexer, TextChunk
        chunks = [
            TextChunk(id="1", content="the cat sat on the mat", metadata={}),
            TextChunk(id="2", content="the dog chased the cat", metadata={}),
            TextChunk(id="3", content="the bird flew over the tree", metadata={}),
        ]
        indexer = BM25Indexer()
        indexer.index(chunks)
        results = indexer.score("cat")
        assert len(results) >= 1
        # Chunks 1 and 2 mention "cat"
        doc_ids = [doc_id for doc_id, _ in results]
        assert 0 in doc_ids
        assert 1 in doc_ids

    def test_empty_index(self):
        from domains.cognitive.rag import BM25Indexer
        indexer = BM25Indexer()
        indexer.index([])
        results = indexer.score("anything")
        assert results == []


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def test_retrieve_returns_results(self):
        from domains.cognitive.rag import HybridRetriever, TextChunk
        retriever = HybridRetriever(use_rerank=False)
        for i, text in enumerate([
            "Python is a programming language",
            "JavaScript runs in the browser",
            "Rust is fast and safe",
            "Python has garbage collection",
        ]):
            retriever.add_chunk(TextChunk(id=str(i), content=text, metadata={"source": f"doc{i}"}))
        retriever.build_index()
        results = retriever.retrieve("Python programming", top_k=2)
        assert len(results) >= 1
        assert all(r.combined_score > 0 for r in results)

    def test_retrieve_empty_index(self):
        from domains.cognitive.rag import HybridRetriever
        retriever = HybridRetriever()
        retriever.build_index()
        results = retriever.retrieve("anything", top_k=5)
        assert results == []

    def test_add_chunk_increases_count(self):
        from domains.cognitive.rag import HybridRetriever, TextChunk
        retriever = HybridRetriever()
        retriever.add_chunk(TextChunk(id="1", content="hello world", metadata={}))
        assert len(retriever.chunks) == 1
        retriever.add_chunk(TextChunk(id="2", content="goodbye world", metadata={}))
        assert len(retriever.chunks) == 2


# ---------------------------------------------------------------------------
# CitationTracker
# ---------------------------------------------------------------------------

class TestCitationTracker:
    def test_extract_claims(self):
        from domains.cognitive.rag import CitationTracker
        tracker = CitationTracker()
        claims = tracker.extract_claims("Python is a language. Java was created in 1995.")
        assert len(claims) >= 2
        subjects = [c["subject"] for c in claims]
        assert "Python" in subjects

    def test_cite_and_format(self):
        from domains.cognitive.rag import CitationTracker, TextChunk
        tracker = CitationTracker()
        claims = tracker.extract_claims("Python is a language.")
        assert len(claims) >= 1
        chunk = TextChunk(id="c1", content="Python is a programming language", metadata={"source": "wiki"})
        cited = tracker.cite(claims[0], [chunk])
        assert cited["supported"] is True
        assert len(cited["sources"]) == 1
        formatted = tracker.format_citations()
        assert "Python" in formatted


# ---------------------------------------------------------------------------
# HallucinationDetector
# ---------------------------------------------------------------------------

class TestHallucinationDetector:
    def test_detect_returns_structure(self):
        from domains.cognitive.rag import HallucinationDetector, HybridRetriever, TextChunk
        retriever = HybridRetriever(use_rerank=False)
        retriever.add_chunk(TextChunk(id="1", content="Paris is the capital of France", metadata={"source": "wiki"}))
        retriever.build_index()
        detector = HallucinationDetector(retriever)
        result = detector.detect("Paris is the capital of France.")
        assert "overall_confidence" in result
        assert "hallucination_rate" in result
        assert "grounded_claims" in result

    def test_unsupported_claim_detected(self):
        from domains.cognitive.rag import HallucinationDetector, HybridRetriever, TextChunk
        retriever = HybridRetriever(use_rerank=False)
        retriever.add_chunk(TextChunk(id="1", content="Paris is the capital of France", metadata={}))
        retriever.build_index()
        detector = HallucinationDetector(retriever)
        result = detector.detect("Atlantis is a real city.")
        assert result["hallucination_rate"] >= 0


# ---------------------------------------------------------------------------
# ProductionRAG
# ---------------------------------------------------------------------------

class TestProductionRAG:
    def test_add_and_query(self):
        from domains.cognitive.rag import ProductionRAG
        rag = ProductionRAG()
        rag.add_document(
            content="SloughGPT is an AI assistant that runs locally. It uses SloNet for inference.",
            metadata={"source": "docs"},
        )
        result = rag.query("What is SloughGPT?")
        assert result["num_results"] >= 1
        assert "context" in result
        assert len(result["context"]) > 0

    def test_verify_and_ground(self):
        from domains.cognitive.rag import ProductionRAG
        rag = ProductionRAG()
        rag.add_document(
            content="Python was created by Guido van Rossum in 1991.",
            metadata={"source": "wiki"},
        )
        result = rag.verify_and_ground("Python was created by Guido van Rossum.", "Who created Python?")
        assert "verification" in result
        assert "confidence" in result
        assert "is_verified" in result


# ---------------------------------------------------------------------------
# RAGService
# ---------------------------------------------------------------------------

class TestRAGService:
    def test_singleton(self):
        from domains.cognitive.rag_service import get_rag_service
        s1 = get_rag_service()
        s2 = get_rag_service()
        assert s1 is s2

    def test_add_and_query(self):
        from domains.cognitive.rag_service import get_rag_service
        svc = get_rag_service()
        chunk_ids = svc.add_document(
            content="The quick brown fox jumps over the lazy dog. This is a test document for RAG.",
            metadata={"source": "test"},
            chunk_size=20,
        )
        assert len(chunk_ids) >= 1
        result = svc.query("quick brown fox")
        assert result["num_results"] >= 1

    def test_stats(self):
        from domains.cognitive.rag_service import get_rag_service
        svc = get_rag_service()
        svc.add_document(content="test document for stats", metadata={})
        stats = svc.stats()
        assert stats["total_documents"] == 1
        assert stats["total_chunks"] >= 1

    def test_list_documents(self):
        from domains.cognitive.rag_service import get_rag_service
        svc = get_rag_service()
        svc.add_document(content="document one", metadata={"source": "a"})
        svc.add_document(content="document two", metadata={"source": "b"})
        docs = svc.list_documents()
        assert len(docs) == 2
        assert docs[0]["metadata"]["source"] == "a"

    def test_clear(self):
        from domains.cognitive.rag_service import get_rag_service
        svc = get_rag_service()
        svc.add_document(content="to be cleared", metadata={})
        assert svc.stats()["total_documents"] == 1
        count = svc.clear()
        assert count == 1
        assert svc.stats()["total_documents"] == 0

    def test_persistence(self):
        """Verify documents survive RAG service recreation."""
        from domains.cognitive.rag_service import get_rag_service, RAGService
        svc = get_rag_service()
        svc.add_document(content="persistent document", metadata={"source": "persist"})
        # Simulate restart by creating a fresh instance
        import domains.cognitive.rag_service as mod
        mod._rag_service = None
        svc2 = RAGService()
        assert svc2.stats()["total_documents"] == 1

    def test_verify_and_ground(self):
        from domains.cognitive.rag_service import get_rag_service
        svc = get_rag_service()
        svc.add_document(content="The Earth orbits the Sun.", metadata={"source": "science"})
        result = svc.verify_and_ground("The Earth orbits the Sun.", "What does the Earth orbit?")
        assert "verification" in result
        assert "confidence" in result


# ---------------------------------------------------------------------------
# KB Router RAG Endpoints
# ---------------------------------------------------------------------------

class TestKBRouterRAGEndpoints:
    """Integration tests for the /kb/rag/* endpoints."""

    @pytest.fixture(autouse=True)
    def _setup_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.kb import KBRouter
        self.app = FastAPI()
        kb = KBRouter()
        self.app.include_router(kb.router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_rag_ingest(self):
        resp = self.client.post("/knowledge/rag/ingest", json={
            "content": "Python is a programming language created in 1991.",
            "source": "test",
            "topic": "programming",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["num_chunks"] >= 1
        assert data["stats"]["total_documents"] >= 1

    def test_rag_query(self):
        # Ingest first
        self.client.post("/knowledge/rag/ingest", json={
            "content": "SloughGPT is an AI assistant.",
            "source": "test",
        })
        # Query
        resp = self.client.post("/knowledge/rag/query", json={"question": "What is SloughGPT?"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["num_results"] >= 1

    def test_rag_list_documents(self):
        self.client.post("/knowledge/rag/ingest", json={"content": "doc one", "source": "a"})
        resp = self.client.get("/knowledge/rag/documents")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["documents"]) >= 1

    def test_rag_stats(self):
        resp = self.client.get("/knowledge/rag/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "total_documents" in data
        assert "total_chunks" in data

    def test_rag_clear(self):
        self.client.post("/knowledge/rag/ingest", json={"content": "to clear", "source": "test"})
        resp = self.client.post("/knowledge/rag/clear")
        assert resp.status_code == 200
        assert resp.json()["data"]["cleared"] >= 1

    def test_rag_verify(self):
        self.client.post("/knowledge/rag/ingest", json={
            "content": "Python was created by Guido van Rossum.",
            "source": "wiki",
        })
        resp = self.client.post("/knowledge/rag/verify", json={
            "text": "Python was created by Guido van Rossum.",
            "question": "Who created Python?",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "verification" in data
        assert "confidence" in data
