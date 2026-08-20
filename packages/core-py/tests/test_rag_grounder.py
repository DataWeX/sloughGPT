"""Meaningful tests for RAGGrounder — document storage, chunking, retrieval, grounding."""

import pytest
from domains.cognitive.grounding import RAGGrounder, Document


class TestRAGGrounderAddDocument:
    def test_add_document(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="Hello world", source="test")
        rag.add_document(doc)
        assert "d1" in rag.documents

    def test_add_document_chunks(self):
        rag = RAGGrounder()
        words = " ".join([f"word{i}" for i in range(20)])
        doc = Document(id="d1", content=words, source="test")
        rag.add_document(doc, chunk_size=5)
        assert len(rag.chunks) == 4  # 20/5 = 4

    def test_chunk_ids(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="a b c d e f", source="test")
        rag.add_document(doc, chunk_size=2)
        assert rag.chunks[0].id == "d1_chunk_0"
        assert rag.chunks[1].id == "d1_chunk_1"

    def test_chunk_metadata_inherits_parent(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="a b c", source="wiki", metadata={"page": 1})
        rag.add_document(doc, chunk_size=2)
        assert rag.chunks[0].metadata["parent_id"] == "d1"
        assert rag.chunks[0].metadata["page"] == 1

    def test_add_text(self):
        rag = RAGGrounder()
        doc_id = rag.add_text("Hello world", source="user")
        assert doc_id == "doc_0"
        assert len(rag.documents) == 1


class TestRAGGrounderRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_exact_match(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="Python is a programming language", source="wiki")
        rag.add_document(doc)
        results = await rag.retrieve("Python programming", top_k=5, min_relevance=0.3)
        assert len(results) >= 1
        assert "Python" in results[0].content

    @pytest.mark.asyncio
    async def test_retrieve_no_match(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="Python is a programming language", source="wiki")
        rag.add_document(doc)
        results = await rag.retrieve("quantum physics", top_k=5, min_relevance=0.5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_top_k(self):
        rag = RAGGrounder()
        for i in range(10):
            rag.add_text(f"Document {i} about Python programming", source=f"src{i}")
        results = await rag.retrieve("Python", top_k=3, min_relevance=0.3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_min_relevance_filters(self):
        rag = RAGGrounder()
        rag.add_text("alpha beta gamma", source="s1")  # 3 words
        rag.add_text("delta epsilon", source="s2")     # 2 words
        # Query "alpha beta" → overlap with s1 is 2/2=1.0, with s2 is 0
        results = await rag.retrieve("alpha beta", top_k=5, min_relevance=0.5)
        assert len(results) == 1
        assert results[0].content == "alpha beta gamma"


class TestRAGGrounderGround:
    def test_ground_response_no_docs(self):
        rag = RAGGrounder()
        result = rag.ground_response("response", "query")
        assert result["grounded"] is False
        assert result["confidence"] == 0.0

    def test_ground_response_structure(self):
        rag = RAGGrounder()
        result = rag.ground_response("response", "query")
        assert "response" in result
        assert "grounded" in result
        assert "confidence" in result
        assert "supporting_docs" in result
        assert "contradictions" in result
        assert "hallucination_score" in result
