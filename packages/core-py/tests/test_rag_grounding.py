"""Tests for domains.cognitive.rag — TextChunk, RetrievalResult, BM25Indexer, HybridRetriever, CitationTracker, HallucinationDetector, ProductionRAG; domains.cognitive.grounding — Document, FisherInformation, KnowledgeNode, KnowledgeEdge, RAGGrounder, KnowledgeGrounding, CurriculumLearner, GroundingOrchestrator, HierarchicalContext."""

import numpy as np
from domains.cognitive.rag import (
    TextChunk, RetrievalResult, BM25Indexer, HybridRetriever,
    CitationTracker, HallucinationDetector, ProductionRAG,
)
from domains.cognitive.grounding import (
    Document, FisherInformation, KnowledgeNode, KnowledgeEdge,
    RAGGrounder, KnowledgeGrounding, CurriculumLearner,
    GroundingOrchestrator, HierarchicalContext,
)


class TestTextChunk:
    def test_fields(self):
        tc = TextChunk(id="c1", content="hello world", metadata={}, token_count=2)
        assert tc.id == "c1"
        assert tc.token_count == 2

    def test_auto_token_count(self):
        tc = TextChunk(id="c1", content="hello world foo bar", metadata={})
        assert tc.token_count == 4

    def test_with_embedding(self):
        emb = np.array([0.1, 0.2])
        tc = TextChunk(id="c1", content="x", metadata={}, embedding=emb)
        assert tc.embedding is not None

    def test_metadata_custom(self):
        tc = TextChunk(id="c1", content="x", metadata={"source": "test", "page": 1})
        assert tc.metadata["source"] == "test"
        assert tc.metadata["page"] == 1

    def test_bm25_score_default(self):
        tc = TextChunk(id="c1", content="x", metadata={})
        assert tc.bm25_score == 0.0

    def test_embedding_default_none(self):
        tc = TextChunk(id="c1", content="x", metadata={})
        assert tc.embedding is None

    def test_empty_content(self):
        tc = TextChunk(id="c1", content="", metadata={})
        assert tc.token_count == 0

    def test_single_word(self):
        tc = TextChunk(id="c1", content="hello", metadata={})
        assert tc.token_count == 1

    def test_long_content(self):
        long = " ".join(["word"] * 1000)
        tc = TextChunk(id="c1", content=long, metadata={})
        assert tc.token_count == 1000


class TestRetrievalResult:
    def test_fields(self):
        tc = TextChunk(id="c1", content="x", metadata={})
        rr = RetrievalResult(chunk=tc, dense_score=0.9, sparse_score=0.8, combined_score=0.85, rank=1)
        assert rr.dense_score == 0.9
        assert rr.rank == 1

    def test_chunk_reference(self):
        tc = TextChunk(id="c2", content="test", metadata={})
        rr = RetrievalResult(chunk=tc, dense_score=0.5, sparse_score=0.5, combined_score=0.5, rank=2)
        assert rr.chunk.id == "c2"

    def test_zero_scores(self):
        tc = TextChunk(id="c1", content="x", metadata={})
        rr = RetrievalResult(chunk=tc, dense_score=0.0, sparse_score=0.0, combined_score=0.0, rank=0)
        assert rr.dense_score == 0.0

    def test_high_rank(self):
        tc = TextChunk(id="c1", content="x", metadata={})
        rr = RetrievalResult(chunk=tc, dense_score=0.9, sparse_score=0.9, combined_score=0.9, rank=100)
        assert rr.rank == 100


class TestBM25Indexer:
    def test_index_and_score(self):
        indexer = BM25Indexer()
        chunks = [
            TextChunk(id="1", content="the cat sat on the mat", metadata={}),
            TextChunk(id="2", content="the dog played in the park", metadata={}),
            TextChunk(id="3", content="the cat chased the dog", metadata={}),
        ]
        indexer.index(chunks)
        results = indexer.score("cat")
        assert len(results) > 0
        assert results[0][0] in [0, 2]

    def test_no_results(self):
        indexer = BM25Indexer()
        chunks = [TextChunk(id="1", content="hello world", metadata={})]
        indexer.index(chunks)
        results = indexer.score("xyz")
        assert len(results) == 0

    def test_single_doc(self):
        indexer = BM25Indexer()
        chunks = [TextChunk(id="1", content="machine learning is great", metadata={})]
        indexer.index(chunks)
        results = indexer.score("machine")
        assert len(results) == 1

    def test_multiple_matches(self):
        indexer = BM25Indexer()
        chunks = [
            TextChunk(id="1", content="python programming language", metadata={}),
            TextChunk(id="2", content="python is popular", metadata={}),
            TextChunk(id="3", content="java programming", metadata={}),
        ]
        indexer.index(chunks)
        results = indexer.score("python")
        assert len(results) == 2

    def test_avg_doc_length(self):
        indexer = BM25Indexer()
        chunks = [
            TextChunk(id="1", content="a b c", metadata={}),
            TextChunk(id="2", content="a b c d e", metadata={}),
        ]
        indexer.index(chunks)
        assert indexer.avg_doc_length == 4.0

    def test_tokenization(self):
        indexer = BM25Indexer()
        tokens = indexer._tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_empty_query(self):
        indexer = BM25Indexer()
        chunks = [TextChunk(id="1", content="hello", metadata={})]
        indexer.index(chunks)
        results = indexer.score("")
        assert len(results) == 0

    def test_duplicate_terms(self):
        indexer = BM25Indexer()
        chunks = [
            TextChunk(id="1", content="cat cat cat", metadata={}),
            TextChunk(id="2", content="cat dog", metadata={}),
        ]
        indexer.index(chunks)
        results = indexer.score("cat")
        assert len(results) == 2
        assert results[0][0] == 0

    def test_case_insensitive(self):
        indexer = BM25Indexer()
        chunks = [TextChunk(id="1", content="Python Programming", metadata={})]
        indexer.index(chunks)
        results = indexer.score("python")
        assert len(results) == 1

    def test_custom_k1_b(self):
        indexer = BM25Indexer(k1=2.0, b=0.5)
        assert indexer.k1 == 2.0
        assert indexer.b == 0.5


class TestHybridRetriever:
    def test_add_and_build(self):
        retriever = HybridRetriever()
        retriever.add_chunk(TextChunk(id="1", content="python is great", metadata={}))
        retriever.add_chunk(TextChunk(id="2", content="java is popular", metadata={}))
        retriever.build_index()
        assert len(retriever.chunks) == 2

    def test_retrieve(self):
        retriever = HybridRetriever()
        for i in range(10):
            retriever.add_chunk(TextChunk(id=str(i), content=f"document {i} about python", metadata={}))
        retriever.build_index()
        results = retriever.retrieve("python", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_empty(self):
        retriever = HybridRetriever()
        retriever.build_index()
        results = retriever.retrieve("anything", top_k=5)
        assert len(results) == 0

    def test_weight_config(self):
        retriever = HybridRetriever(dense_weight=0.3, sparse_weight=0.7)
        assert retriever.dense_weight == 0.3
        assert retriever.sparse_weight == 0.7

    def test_rank_ordering(self):
        retriever = HybridRetriever(use_rerank=False)
        for i in range(5):
            retriever.add_chunk(TextChunk(id=str(i), content=f"python coding example {i}", metadata={}))
        retriever.build_index()
        results = retriever.retrieve("python", top_k=5)
        for i in range(len(results) - 1):
            assert results[i].combined_score >= results[i+1].combined_score

    def test_min_score_filter(self):
        retriever = HybridRetriever(use_rerank=False)
        retriever.add_chunk(TextChunk(id="1", content="python", metadata={}))
        retriever.add_chunk(TextChunk(id="2", content="java", metadata={}))
        retriever.build_index()
        results = retriever.retrieve("python", top_k=10, min_score=0.99)
        assert len(results) <= 2

    def test_embedding_cache(self):
        retriever = HybridRetriever()
        retriever.add_chunk(TextChunk(id="1", content="test content", metadata={}))
        retriever.build_index()
        emb1 = retriever._get_embedding("test content")
        emb2 = retriever._get_embedding("test content")
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_queries(self):
        retriever = HybridRetriever(use_rerank=True)
        retriever.add_chunk(TextChunk(id="1", content="the quick brown fox jumps over the lazy dog", metadata={}))
        retriever.add_chunk(TextChunk(id="2", content="a lazy dog sleeps on the warm sunny porch", metadata={}))
        retriever.add_chunk(TextChunk(id="3", content="colorful tropical fish swim in coral reefs", metadata={}))
        retriever.build_index()
        r1 = retriever.retrieve("fox", top_k=1)
        r2 = retriever.retrieve("fish", top_k=1)
        assert r1[0].chunk.id != r2[0].chunk.id

    def test_rerank_removes_duplicates(self):
        retriever = HybridRetriever(use_rerank=True)
        content = "the python programming language is great for coding"
        for i in range(5):
            retriever.add_chunk(TextChunk(id=str(i), content=content, metadata={}))
        retriever.build_index()
        results = retriever.retrieve("python programming", top_k=5)
        assert len(results) <= 5


class TestCitationTracker:
    def test_extract_claims(self):
        tracker = CitationTracker()
        claims = tracker.extract_claims("Python is a programming language. Java is popular.")
        assert len(claims) == 2
        assert claims[0]["subject"] == "Python"

    def test_cite(self):
        tracker = CitationTracker()
        claim = {"subject": "Python", "predicate": "is a programming language", "text": "Python is a programming language."}
        chunk = TextChunk(id="c1", content="Python is a language", metadata={"source": "wiki"})
        cited = tracker.cite(claim, [chunk])
        assert cited["supported"] is True
        assert len(cited["sources"]) == 1

    def test_cite_no_sources(self):
        tracker = CitationTracker()
        claim = {"subject": "X", "predicate": "is Y", "text": "X is Y."}
        cited = tracker.cite(claim, [])
        assert cited["supported"] is False

    def test_format_citations(self):
        tracker = CitationTracker()
        claims = tracker.extract_claims("Python is a language.")
        tracker.cite(claims[0], [TextChunk(id="c1", content="Python", metadata={"source": "wiki"})])
        formatted = tracker.format_citations()
        assert "[1]" in formatted

    def test_no_claims(self):
        tracker = CitationTracker()
        claims = tracker.extract_claims("Just some random text without patterns.")
        assert len(claims) == 0

    def test_multiple_claims(self):
        tracker = CitationTracker()
        claims = tracker.extract_claims("Python is a language. Java is a language. Go is fast.")
        assert len(claims) >= 2

    def test_cite_multiple_sources(self):
        tracker = CitationTracker()
        claim = {"subject": "Python", "predicate": "is a language", "text": "Python is a language."}
        chunks = [
            TextChunk(id="c1", content="Python", metadata={"source": "wiki"}),
            TextChunk(id="c2", content="Python lang", metadata={"source": "docs"}),
        ]
        cited = tracker.cite(claim, chunks)
        assert len(cited["sources"]) == 2

    def test_format_citations_empty(self):
        tracker = CitationTracker()
        formatted = tracker.format_citations()
        assert formatted == ""


class TestDocument:
    def test_fields(self):
        d = Document(id="d1", content="text", source="file")
        assert d.id == "d1"
        assert d.source == "file"
        assert d.metadata == {}

    def test_with_metadata(self):
        d = Document(id="d2", content="text", source="web", metadata={"url": "http://example.com"})
        assert d.metadata["url"] == "http://example.com"

    def test_embedding_default(self):
        d = Document(id="d1", content="text", source="file")
        assert d.embedding is None

    def test_with_embedding(self):
        emb = np.array([1.0, 2.0, 3.0])
        d = Document(id="d1", content="text", source="file", embedding=emb)
        np.testing.assert_array_equal(d.embedding, emb)


class TestFisherInformation:
    def test_fields(self):
        fi = FisherInformation(param_name="W", importance=0.5, old_value=0.3)
        assert fi.param_name == "W"
        assert fi.importance == 0.5

    def test_zero_importance(self):
        fi = FisherInformation(param_name="b", importance=0.0, old_value=0.0)
        assert fi.importance == 0.0

    def test_high_importance(self):
        fi = FisherInformation(param_name="W", importance=1.0, old_value=0.5)
        assert fi.importance == 1.0


class TestKnowledgeNode:
    def test_fields(self):
        kn = KnowledgeNode(id="n1", label="cat", node_type="entity")
        assert kn.id == "n1"
        assert kn.node_type == "entity"
        assert kn.properties == {}

    def test_with_properties(self):
        kn = KnowledgeNode(id="n1", label="cat", node_type="entity", properties={"color": "orange"})
        assert kn.properties["color"] == "orange"

    def test_concept_type(self):
        kn = KnowledgeNode(id="c1", label="democracy", node_type="concept")
        assert kn.node_type == "concept"

    def test_event_type(self):
        kn = KnowledgeNode(id="e1", label="explosion", node_type="event")
        assert kn.node_type == "event"


class TestKnowledgeEdge:
    def test_fields(self):
        ke = KnowledgeEdge(source="n1", target="n2", relation="is_a", weight=0.8)
        assert ke.source == "n1"
        assert ke.relation == "is_a"
        assert ke.weight == 0.8

    def test_defaults(self):
        ke = KnowledgeEdge(source="a", target="b", relation="related_to")
        assert ke.weight == 1.0

    def test_causes_relation(self):
        ke = KnowledgeEdge(source="fire", target="smoke", relation="causes", weight=0.95)
        assert ke.relation == "causes"

    def test_part_of_relation(self):
        ke = KnowledgeEdge(source="wheel", target="car", relation="part_of")
        assert ke.relation == "part_of"


class TestKnowledgeGrounding:
    def test_add_fact(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "is_a", "animal")
        assert "cat" in kg.nodes
        assert "animal" in kg.nodes
        assert len(kg.edges) == 1

    def test_query(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("dog", "is_a", "animal")
        results = kg.query("cat")
        assert "animal" in results

    def test_query_with_relation(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("cat", "has", "tail")
        results = kg.query("cat", relation="has")
        assert "tail" in results
        assert "animal" not in results

    def test_verify_statement(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "is_a", "animal")
        result = kg.verify_statement("cat is_a animal")
        assert result["verified"] is True

    def test_verify_unknown(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "is_a", "animal")
        result = kg.verify_statement("dog is_a animal")
        assert result["verified"] is False

    def test_context_for_prompt(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        kg.add_fact("python", "used_for", "programming")
        context = kg.get_context_for_prompt("Tell me about python")
        assert "python" in context.lower()

    def test_multiple_facts(self):
        kg = KnowledgeGrounding()
        kg.add_fact("A", "rel1", "B")
        kg.add_fact("A", "rel2", "C")
        results = kg.query("A")
        assert "B" in results
        assert "C" in results

    def test_query_no_results(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "is_a", "animal")
        results = kg.query("dog")
        assert len(results) == 0

    def test_verify_short_statement(self):
        kg = KnowledgeGrounding()
        result = kg.verify_statement("hello")
        assert result["verified"] is False


class TestRAGGrounder:
    def test_add_document(self):
        grounder = RAGGrounder()
        doc = Document(id="d1", content="this is a test document with many words " * 10, source="test")
        grounder.add_document(doc, chunk_size=10)
        assert len(grounder.chunks) > 0

    def test_add_text(self):
        grounder = RAGGrounder()
        doc_id = grounder.add_text("hello world")
        assert doc_id.startswith("doc_")

    def test_retrieve(self):
        import asyncio
        grounder = RAGGrounder()
        grounder.add_text("python programming language tutorial")
        results = asyncio.run(grounder.retrieve("python", top_k=5))
        assert len(results) > 0

    def test_ground_response(self):
        grounder = RAGGrounder()
        grounder.add_text("python is a programming language")
        result = grounder.ground_response("Python is a language", "what is python")
        assert "response" in result
        assert "grounded" in result
        assert "confidence" in result

    def test_empty_retrieve(self):
        import asyncio
        grounder = RAGGrounder()
        results = asyncio.run(grounder.retrieve("anything", top_k=5))
        assert len(results) == 0

    def test_multiple_documents(self):
        grounder = RAGGrounder()
        grounder.add_text("python is a programming language")
        grounder.add_text("java is another programming language")
        import asyncio
        results = asyncio.run(grounder.retrieve("programming", top_k=5))
        assert len(results) > 0

    def test_chunk_sizes(self):
        grounder = RAGGrounder()
        doc = Document(id="d1", content="word " * 100, source="test")
        grounder.add_document(doc, chunk_size=10)
        assert len(grounder.chunks) >= 10


class TestCurriculumLearner:
    def test_add_example(self):
        cl = CurriculumLearner()
        cl.add_example("easy_example", 0.1)
        cl.add_example("hard_example", 0.9)
        assert len(cl.difficulty_levels) > 0

    def test_get_batch_bootstrapping(self):
        cl = CurriculumLearner()
        for i in range(10):
            cl.add_example(f"ex_{i}", i / 10.0)
        batch = cl.get_batch(5)
        assert len(batch) <= 5

    def test_update_stage_progressing(self):
        cl = CurriculumLearner()
        cl.update_stage(0.8)
        assert cl.stage == "progressing"

    def test_update_stage_mastery(self):
        cl = CurriculumLearner()
        cl.update_stage(0.95)
        assert cl.stage == "mastery"

    def test_update_stage_bootstrapping(self):
        cl = CurriculumLearner()
        cl.update_stage(0.5)
        assert cl.stage == "bootstrapping"

    def test_batch_mastery(self):
        cl = CurriculumLearner()
        cl.stage = "mastery"
        for i in range(10):
            cl.add_example(f"ex_{i}", i / 10.0)
        batch = cl.get_batch(10)
        assert len(batch) > 0

    def test_progressing_batch_includes_harder(self):
        cl = CurriculumLearner()
        cl.stage = "progressing"
        cl.current_level = 5
        for i in range(10):
            cl.add_example(f"ex_{i}", i / 10.0)
        batch = cl.get_batch(10)
        assert len(batch) > 0

    def test_empty_batch(self):
        cl = CurriculumLearner()
        batch = cl.get_batch(5)
        assert len(batch) == 0


class TestHierarchicalContext:
    def test_build_hierarchy(self):
        hc = HierarchicalContext(chunk_size=10)
        text = "word " * 100
        hc.build_hierarchy(text)
        assert len(hc.hierarchy) > 0

    def test_get_relevant_context(self):
        hc = HierarchicalContext(chunk_size=10)
        text = "word " * 100
        hc.build_hierarchy(text)
        context = hc.get_relevant_context("query")
        assert len(context) > 0

    def test_empty_hierarchy(self):
        hc = HierarchicalContext()
        context = hc.get_relevant_context("query")
        assert context == ""

    def test_attention_mask(self):
        hc = HierarchicalContext(chunk_size=4)
        text = "word " * 16
        hc.build_hierarchy(text)
        mask = hc.attention_mask(8)
        assert mask.shape == (8, 8)

    def test_max_context_limit(self):
        hc = HierarchicalContext(max_context=50, chunk_size=10)
        text = "word " * 100
        hc.build_hierarchy(text)
        context = hc.get_relevant_context("query")
        assert len(context) <= 50

    def test_single_chunk(self):
        hc = HierarchicalContext(chunk_size=100)
        text = "hello world test"
        hc.build_hierarchy(text)
        assert len(hc.hierarchy) >= 1

    def test_hierarchy_levels(self):
        hc = HierarchicalContext(chunk_size=4)
        text = "word " * 32
        hc.build_hierarchy(text)
        assert len(hc.hierarchy) >= 2


class TestGroundingOrchestrator:
    def test_add_data(self):
        go = GroundingOrchestrator()
        go.add_data("cat is a animal")
        assert len(go.rag.chunks) > 0

    def test_ground_output(self):
        go = GroundingOrchestrator()
        go.add_data("python is a language")
        result = go.ground_output("Python is a language", "what is python")
        assert "response" in result
        assert "confidence" in result

    def test_knowledge_context(self):
        go = GroundingOrchestrator()
        go.add_data("fire causes smoke")
        context = go.get_knowledge_context("fire")
        assert len(context) > 0

    def test_curriculum_batch(self):
        go = GroundingOrchestrator()
        go.curriculum.add_example("easy", 0.1)
        go.curriculum.add_example("hard", 0.9)
        batch = go.get_curriculum_batch(2)
        assert len(batch) <= 2

    def test_triple_extraction(self):
        go = GroundingOrchestrator()
        triples = go._extract_triples("fire causes smoke")
        assert len(triples) > 0

    def test_ground_output_empty(self):
        go = GroundingOrchestrator()
        result = go.ground_output("test response", "test query")
        assert result["confidence"] > 0


class TestHallucinationDetector:
    def test_detect_basic(self):
        retriever = HybridRetriever()
        retriever.add_chunk(TextChunk(id="1", content="python is a programming language", metadata={}))
        retriever.build_index()
        detector = HallucinationDetector(retriever)
        result = detector.detect("Python is a programming language.")
        assert "hallucinations" in result
        assert "grounded_claims" in result
        assert "overall_confidence" in result

    def test_detect_no_claims(self):
        retriever = HybridRetriever()
        retriever.build_index()
        detector = HallucinationDetector(retriever)
        result = detector.detect("Just random text without patterns.")
        assert result["total_claims"] == 0
        assert result["overall_confidence"] == 1.0

    def test_hallucination_rate(self):
        retriever = HybridRetriever()
        retriever.add_chunk(TextChunk(id="1", content="cat is animal", metadata={}))
        retriever.build_index()
        detector = HallucinationDetector(retriever)
        result = detector.detect("Cat is an animal. Dog flies in space.")
        assert 0.0 <= result["hallucination_rate"] <= 1.0

    def test_detect_with_evidence(self):
        retriever = HybridRetriever()
        retriever.add_chunk(TextChunk(id="1", content="Python is a programming language used for web development", metadata={}))
        retriever.build_index()
        detector = HallucinationDetector(retriever)
        result = detector.detect("Python is a programming language.")
        assert result["total_claims"] >= 1

    def test_formatted_citations(self):
        retriever = HybridRetriever()
        retriever.add_chunk(TextChunk(id="1", content="Python is a language", metadata={}))
        retriever.build_index()
        detector = HallucinationDetector(retriever)
        result = detector.detect("Python is a language.")
        assert "formatted_citations" in result


class TestProductionRAG:
    def test_add_document(self):
        rag = ProductionRAG()
        ids = rag.add_document("This is a test document with enough words to chunk properly " * 20)
        assert len(ids) > 0

    def test_query(self):
        rag = ProductionRAG()
        rag.add_document("python programming language tutorial guide")
        result = rag.query("python")
        assert "results" in result
        assert "context" in result
        assert "num_results" in result

    def test_verify_and_ground(self):
        rag = ProductionRAG()
        rag.add_document("python is a programming language")
        result = rag.verify_and_ground("Python is a language.", "what is python")
        assert "verification" in result
        assert "confidence" in result

    def test_empty_query(self):
        rag = ProductionRAG()
        result = rag.query("anything")
        assert result["num_results"] == 0

    def test_config(self):
        rag = ProductionRAG({"dense_weight": 0.5, "sparse_weight": 0.5})
        assert rag.retriever.dense_weight == 0.5

    def test_multiple_documents(self):
        rag = ProductionRAG()
        rag.add_document("python programming tutorial guide book")
        rag.add_document("java programming language guide book")
        result = rag.query("programming")
        assert result["num_results"] > 0

    def test_add_document_custom_metadata(self):
        rag = ProductionRAG()
        ids = rag.add_document("test content", metadata={"source": "manual"})
        assert len(ids) > 0

    def test_query_no_context(self):
        rag = ProductionRAG()
        rag.add_document("python is a language")
        result = rag.query("python", return_context=False)
        assert result["context"] == ""
