"""Meaningful tests for RelationshipMemory, BM25Indexer, HybridRetriever."""

import time
from domains.soul.cognitive import RelationshipMemory
from domains.cognitive.rag import BM25Indexer, HybridRetriever, TextChunk, RetrievalResult


# ── RelationshipMemory ────────────────────────────────────────────────

class TestRelationshipMemoryProfile:
    def test_get_creates_profile(self):
        rm = RelationshipMemory()
        profile = rm.get_user_profile("u1")
        assert profile["user_id"] == "u1"
        assert profile["total_interactions"] == 0
        assert profile["satisfaction_score"] == 0.5

    def test_get_same_profile(self):
        rm = RelationshipMemory()
        p1 = rm.get_user_profile("u1")
        p2 = rm.get_user_profile("u1")
        assert p1 is p2


class TestRelationshipMemoryUpdate:
    def test_update_increments_interactions(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello there friend", "hi", 0.5, "neutral")
        rm.update_from_interaction("u1", "goodbye now friend", "bye", 0.5, "neutral")
        profile = rm.get_user_profile("u1")
        assert profile["total_interactions"] == 2

    def test_update_tracks_emotions(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello", "hi", 0.8, "happy")
        rm.update_from_interaction("u1", "hello", "hi", 0.8, "happy")
        rm.update_from_interaction("u1", "hello", "hi", -0.5, "sad")
        profile = rm.get_user_profile("u1")
        assert profile["emotional_tendencies"]["happy"] == 2
        assert profile["emotional_tendencies"]["sad"] == 1

    def test_update_tracks_topics(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "quantum computing is fascinating", "tell me more", 0.5, "neutral")
        profile = rm.get_user_profile("u1")
        # "quantum" is 6 chars, should be tracked
        topics = dict(profile["topics_of_interest"])
        assert any("quantum" in t for t in topics)

    def test_update_mood_history_capped(self):
        rm = RelationshipMemory()
        for i in range(60):
            rm.update_from_interaction("u1", "msg", "resp", 0.5, "neutral")
        profile = rm.get_user_profile("u1")
        assert len(profile["mood_history"]) == 50

    def test_update_interaction_history_capped(self):
        rm = RelationshipMemory()
        for i in range(110):
            rm.update_from_interaction("u1", "msg", "resp", 0.5, "neutral")
        assert len(rm.interaction_history["u1"]) == 100

    def test_update_satisfaction_good(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "msg", "resp", 0.5, "neutral", feedback="good")
        profile = rm.get_user_profile("u1")
        assert profile["satisfaction_score"] == 0.6

    def test_update_satisfaction_bad(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "msg", "resp", 0.5, "neutral", feedback="bad")
        profile = rm.get_user_profile("u1")
        assert profile["satisfaction_score"] == 0.4

    def test_satisfaction_clamps(self):
        rm = RelationshipMemory()
        for _ in range(20):
            rm.update_from_interaction("u1", "msg", "resp", 0.5, "neutral", feedback="good")
        profile = rm.get_user_profile("u1")
        assert profile["satisfaction_score"] == 1.0

    def test_satisfaction_clamps_low(self):
        rm = RelationshipMemory()
        for _ in range(20):
            rm.update_from_interaction("u1", "msg", "resp", 0.5, "neutral", feedback="bad")
        profile = rm.get_user_profile("u1")
        assert profile["satisfaction_score"] == 0.0


class TestRelationshipMemorySummary:
    def test_summary(self):
        rm = RelationshipMemory()
        rm.update_from_interaction("u1", "hello world", "hi", 0.8, "happy", feedback="good")
        rm.update_from_interaction("u1", "hello world", "hi", 0.8, "happy")
        summary = rm.get_user_summary("u1")
        assert summary["user_id"] == "u1"
        assert summary["total_interactions"] == 2
        assert summary["dominant_emotion"] == "happy"
        assert summary["satisfaction_score"] == 0.6

    def test_summary_empty(self):
        rm = RelationshipMemory()
        summary = rm.get_user_summary("u1")
        assert summary["dominant_emotion"] == "neutral"


# ── BM25Indexer ───────────────────────────────────────────────────────

class TestBM25Indexer:
    def test_index(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="Python is great", metadata={}, embedding=None),
            TextChunk(id="c2", content="Java is okay", metadata={}, embedding=None),
        ]
        bm25.index(chunks)
        assert bm25.num_docs == 2
        assert bm25.avg_doc_length > 0

    def test_score_matching(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="Python programming language", metadata={}, embedding=None),
            TextChunk(id="c2", content="Java programming language", metadata={}, embedding=None),
        ]
        bm25.index(chunks)
        results = bm25.score("Python")
        assert len(results) >= 1
        assert results[0][0] == 0  # c1 should be highest

    def test_score_no_match(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="Python is great", metadata={}, embedding=None),
        ]
        bm25.index(chunks)
        results = bm25.score("quantum")
        assert len(results) == 0

    def test_score_ranking(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="Python Python Python", metadata={}, embedding=None),
            TextChunk(id="c2", content="Python Java", metadata={}, embedding=None),
        ]
        bm25.index(chunks)
        results = bm25.score("Python")
        # c1 has 3 occurrences, c2 has 1
        assert results[0][0] == 0


# ── HybridRetriever ───────────────────────────────────────────────────

class TestHybridRetriever:
    def test_add_and_retrieve(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="Python is a programming language", metadata={"source": "wiki"}, embedding=None))
        hr.add_chunk(TextChunk(id="c2", content="Java is also a language", metadata={"source": "wiki"}, embedding=None))
        hr.build_index()
        results = hr.retrieve("Python programming", top_k=2)
        assert len(results) >= 1
        assert results[0].chunk.id == "c1"

    def test_retrieve_min_score(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="A B C", metadata={}, embedding=None))
        hr.build_index()
        results = hr.retrieve("A", top_k=5, min_score=999)
        assert len(results) == 0

    def test_retrieve_top_k(self):
        hr = HybridRetriever()
        for i in range(10):
            hr.add_chunk(TextChunk(id=f"c{i}", content=f"Document {i} about Python", metadata={}, embedding=None))
        hr.build_index()
        results = hr.retrieve("Python", top_k=3)
        assert len(results) <= 3

    def test_retrieve_result_has_scores(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="Python is great", metadata={}, embedding=None))
        hr.build_index()
        results = hr.retrieve("Python", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert r.dense_score >= 0
        assert r.sparse_score >= 0
        assert r.combined_score >= 0
        assert r.rank == 1
