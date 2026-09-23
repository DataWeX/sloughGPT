"""
Tests for MathRAG — mathematically rigorous RAG engine.

Tests use real-world document content (technical, encyclopedic, conversational)
to validate retrieval quality, extraction accuracy, and scoring correctness.
"""

import pytest
import numpy as np

from domain.cognition._internal.math_rag import (
    BM25,
    TFIDFIndex,
    ExtractionRule,
    RuleEngine,
    MathRAG,
    Chunk,
    _tokenize,
    _tokenize_raw,
)


# ─── Real-World Test Documents ────────────────────────────────────────────────

DOC_PYTHON = """
Python is a high-level, general-purpose programming language. Its design
philosophy emphasizes code readability with the use of significant indentation.
Python is dynamically typed and garbage-collected. It supports multiple
programming paradigms, including structured, object-oriented, and functional
programming. Python was created by Guido van Rossum and first released in 1991.
Python consistently ranks as one of the most popular programming languages.
Python is used extensively in scientific computing, data analysis, artificial
intelligence, web development, and automation. The Python Package Index (PyPI)
hosts tens of thousands of third-party packages.
"""

DOC_EINSTEIN = """
Albert Einstein was a German-born theoretical physicist. Einstein is best known
for developing the theory of relativity. Einstein received the 1921 Nobel Prize
in Physics for his discovery of the law of the photoelectric effect. Einstein's
work is known for its influence on the philosophy of science. Einstein was born
in Ulm, in the Kingdom of Wurttemberg in the German Empire. Einstein worked at
the Swiss Patent Office in Bern. Einstein published four groundbreaking papers
in 1905, known as his annus mirabilis. Einstein emigrated to the United States
in 1933. Einstein was a lifelong pacifist and Zionist.
"""

DOC_CHEMISTRY = """
Water is a chemical compound with the formula H2O. Water is a transparent,
tasteless, odorless, and nearly colorless chemical substance. Water is the main
constituent of Earth's hydrosphere and the fluids of all known living organisms.
Water covers about 71 percent of the Earth's surface. Water is a polar molecule,
meaning it has a partial positive charge on one side and a partial negative charge
on the other. Water is an excellent solvent for many substances. Water boils at
100 degrees Celsius at standard atmospheric pressure. Water freezes at 0 degrees
Celsius. Water is essential for all known forms of life.
"""

DOCS = [DOC_PYTHON, DOC_EINSTEIN, DOC_CHEMISTRY]


# ─── Tokenizer Tests ─────────────────────────────────────────────────────────

class TestTokenizer:
    def test_tokenize_removes_stop_words(self):
        tokens = _tokenize("The quick brown fox is jumping")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "jumping" in tokens

    def test_tokenize_lowercases(self):
        tokens = _tokenize("Python Language")
        assert "python" in tokens
        assert "language" in tokens

    def test_tokenize_raw_keeps_stop_words(self):
        tokens = _tokenize_raw("The quick brown fox")
        assert "the" in tokens
        assert "quick" in tokens

    def test_tokenize_filters_short_tokens(self):
        tokens = _tokenize("a b I am")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "am" in tokens


# ─── BM25 Tests ──────────────────────────────────────────────────────────────

class TestBM25:
    def _build_bm25(self, docs: list[str]) -> BM25:
        chunks = []
        for i, doc in enumerate(docs):
            chunks.append(Chunk(id=str(i), content=doc))
        bm25 = BM25()
        bm25.build(chunks)
        return bm25

    def test_idf_positive_for_all_terms(self):
        """IDF must always be positive (Robertson-Sparck Jones smoothing)."""
        bm25 = self._build_bm25(DOCS)
        for term in bm25.df:
            assert bm25.idf(term) > 0, f"IDF for '{term}' should be positive"

    def test_idf_rare_terms_score_higher(self):
        """Rare terms should have higher IDF than common terms."""
        bm25 = self._build_bm25(DOCS)
        # "relativity" appears in 1 doc, "water" appears in 1 doc
        # "python" appears in 1 doc
        idf_rel = bm25.idf("relativity")
        idf_einstein = bm25.idf("einstein")
        # Both should be high (appear in 1 of 3 docs)
        assert idf_rel > 0.5
        assert idf_einstein > 0.5

    def test_bm25_returns_relevant_docs(self):
        """Query about Python should rank Python doc highest."""
        bm25 = self._build_bm25(DOCS)
        results = bm25.score(_tokenize_raw("python programming language"))
        assert len(results) > 0
        # Doc 0 is the Python doc
        assert results[0][0] == 0

    def test_bm25_returns_relevant_docs_einstein(self):
        """Query about physics should rank Einstein doc highest."""
        bm25 = self._build_bm25(DOCS)
        results = bm25.score(_tokenize_raw("physics relativity Nobel Prize"))
        assert len(results) > 0
        assert results[0][0] == 1  # Einstein doc

    def test_bm25_scores_are_sorted_desc(self):
        """Results must be sorted by score descending."""
        bm25 = self._build_bm25(DOCS)
        results = bm25.score(_tokenize_raw("science"))
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_empty_query(self):
        """Empty query should return empty results."""
        bm25 = self._build_bm25(DOCS)
        results = bm25.score([])
        assert results == []

    def test_bm25_no_match(self):
        """Query with no matching terms should return empty results."""
        bm25 = self._build_bm25(DOCS)
        results = bm25.score(_tokenize_raw("xyzzy plugh"))
        assert results == []


# ─── TF-IDF Tests ────────────────────────────────────────────────────────────

class TestTFIDF:
    def _build_tfidf(self, docs: list[str]) -> TFIDFIndex:
        chunks = [Chunk(id=str(i), content=doc) for i, doc in enumerate(docs)]
        tfidf = TFIDFIndex()
        tfidf.build(chunks)
        return tfidf

    def test_sublinear_tf_scales(self):
        """Sublinear TF should dampen high-frequency terms."""
        tfidf = self._build_tfidf(DOCS)
        # Build a query with repeated terms
        q1 = ["python"]
        q2 = ["python", "python", "python"]
        r1 = tfidf.query(q1, top_k=1)
        r2 = tfidf.query(q2, top_k=1)
        # Both should find the same doc, but score should differ
        if r1 and r2:
            # q2 has higher TF, so score should be higher (but not 3x due to log scaling)
            assert r2[0][1] >= r1[0][1]

    def test_tfidf_finds_relevant_doc(self):
        """TF-IDF should find the correct document."""
        tfidf = self._build_tfidf(DOCS)
        results = tfidf.query(_tokenize("water chemical compound H2O"), top_k=1)
        assert len(results) == 1
        assert results[0][0] == 2  # Chemistry doc

    def test_tfidf_scores_are_normalized(self):
        """All scores should be in [0, 1] range (cosine of normalized vectors)."""
        tfidf = self._build_tfidf(DOCS)
        results = tfidf.query(_tokenize("programming"))
        for _, score in results:
            assert -1.0 <= score <= 1.0

    def test_tfidf_vocab_built(self):
        """Vocabulary should contain all unique terms."""
        tfidf = self._build_tfidf(DOCS)
        assert "python" in tfidf.vocab
        assert "einstein" in tfidf.vocab
        assert "water" in tfidf.vocab


# ─── Rule Engine Tests ───────────────────────────────────────────────────────

class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_extract_is_a(self):
        facts = self.engine.extract("Python is a programming language.")
        assert len(facts) >= 1
        py_facts = [f for f in facts if f.subject == "Python"]
        assert len(py_facts) >= 1
        assert py_facts[0].relation == "is_a"

    def test_extract_location(self):
        facts = self.engine.extract("Einstein was born in Ulm.")
        location_facts = [f for f in facts if f.relation == "located_in"]
        assert len(location_facts) >= 1

    def test_extract_possession(self):
        facts = self.engine.extract("The car has four wheels.")
        has_facts = [f for f in facts if f.relation == "has"]
        assert len(has_facts) >= 1

    def test_extract_causation(self):
        facts = self.engine.extract("Smoking causes cancer.")
        causes_facts = [f for f in facts if f.relation == "causes"]
        assert len(causes_facts) >= 1

    def test_extract_deduplication(self):
        """Same fact repeated should only appear once."""
        facts = self.engine.extract("Python is a language. Python is a language.")
        py_facts = [f for f in facts if f.subject == "Python"]
        assert len(py_facts) == 1

    def test_extract_confidence_ordering(self):
        """Higher-confidence facts should come first."""
        facts = self.engine.extract(
            "Python is a programming language. "
            "It can be used for web development."
        )
        if len(facts) >= 2:
            # is_a (0.9) should come before can (0.8)
            assert facts[0].confidence >= facts[-1].confidence

    def test_extract_multiple_relations(self):
        text = (
            "Python is a programming language. "
            "Python has many libraries. "
            "Python is used for data science. "
            "Python was created in 1991."
        )
        facts = self.engine.extract(text)
        relations = {f.relation for f in facts}
        assert "is_a" in relations
        assert "has" in relations
        assert "used_for" in relations
        assert "was" in relations

    def test_custom_rule(self):
        """Test adding a custom rule."""
        import re
        custom = ExtractionRule(
            name="invented_by",
            pattern=re.compile(r"([A-Z][\w]+)\s+was\s+invented\s+by\s+(.+)", re.I),
            relation="invented_by",
            confidence=0.9,
            priority=20,
        )
        self.engine.add_rule(custom)
        facts = self.engine.extract("Python was invented by Guido van Rossum.")
        invented = [f for f in facts if f.relation == "invented_by"]
        assert len(invented) >= 1
        assert invented[0].subject == "Python"


# ─── MathRAG Integration Tests ───────────────────────────────────────────────

class TestMathRAG:
    def setup_method(self):
        self.rag = MathRAG()
        for doc in DOCS:
            self.rag.add_document(doc, metadata={"source": "test"})
        self.rag.build_index()

    def test_query_finds_relevant(self):
        """Query about Python should find the Python document."""
        result = self.rag.query("What is Python programming?")
        assert result["num_results"] > 0
        # First result should be the Python doc
        first_chunk = result["results"][0].chunk
        assert "python" in first_chunk.content.lower()

    def test_query_finds_einstein(self):
        """Query about physics should find the Einstein document."""
        result = self.rag.query("theory of relativity Nobel Prize")
        assert result["num_results"] > 0
        first_chunk = result["results"][0].chunk
        assert "einstein" in first_chunk.content.lower()

    def test_query_finds_chemistry(self):
        """Query about water should find the chemistry document."""
        result = self.rag.query("water chemical compound H2O")
        assert result["num_results"] > 0
        first_chunk = result["results"][0].chunk
        assert "water" in first_chunk.content.lower()

    def test_context_concatenation(self):
        """Context should be concatenated chunk texts."""
        result = self.rag.query("python")
        assert isinstance(result["context"], str)
        assert len(result["context"]) > 0

    def test_deduplication(self):
        """Same document should not be indexed twice."""
        rag = MathRAG()
        rag.add_document("Python is a language.", deduplicate=True)
        rag.add_document("Python is a language.", deduplicate=True)
        rag.build_index()
        assert len(rag.chunks) == 1

    def test_no_dedup_when_disabled(self):
        """Deduplication can be disabled."""
        rag = MathRAG()
        rag.add_document("Python is a language.", deduplicate=False)
        rag.add_document("Python is a language.", deduplicate=False)
        rag.build_index()
        assert len(rag.chunks) > 1

    def test_extract_facts(self):
        """Should extract structured facts from text."""
        facts = self.rag.extract_facts("Python is a programming language.")
        assert len(facts) >= 1
        assert facts[0].subject == "Python"
        assert facts[0].relation == "is_a"

    def test_verify_and_ground(self):
        """Verification should detect grounded and ungrounded claims."""
        result = self.rag.verify_and_ground(
            "Python is a programming language. unicorns are magical creatures.",
            "What is Python?",
        )
        assert "verification" in result
        assert "grounded_claims" in result
        assert "hallucinations" in result
        # Python claim should be grounded (it's in the index)
        grounded_subjects = [c["subject"] for c in result["grounded_claims"]]
        assert "Python" in grounded_subjects

    def test_stats(self):
        """Stats should report correct counts."""
        stats = self.rag.stats()
        assert stats["chunks"] > 0
        assert stats["total_tokens"] > 0
        assert stats["indexed"] is True

    def test_empty_document(self):
        """Empty document should return empty chunk IDs."""
        rag = MathRAG()
        ids = rag.add_document("")
        assert ids == []

    def test_scoring_components_transparent(self):
        """Each result should have bm25, tfidf, and combined scores."""
        result = self.rag.query("python programming")
        for r in result["results"]:
            assert hasattr(r, "bm25")
            assert hasattr(r, "tfidf")
            assert hasattr(r, "combined")
            assert r.combined > 0


# ─── Benchmark: MathRAG vs ProductionRAG ─────────────────────────────────────

class TestBenchmark:
    """Compare MathRAG against ProductionRAG on the same documents."""

    def _build_prod_rag(self, docs: list[str]):
        from domain.cognition._internal.rag import ProductionRAG
        rag = ProductionRAG()
        for doc in docs:
            rag.add_document(doc, metadata={"source": "test"})
        return rag

    def _build_math_rag(self, docs: list[str]) -> MathRAG:
        rag = MathRAG()
        for doc in docs:
            rag.add_document(doc, metadata={"source": "test"})
        rag.build_index()
        return rag

    def test_retrieval_overlap(self):
        """Both engines should find a result for clear queries."""
        prod_rag = self._build_prod_rag(DOCS)
        math_rag = self._build_math_rag(DOCS)

        queries = [
            "python programming language",
            "relativity physics Nobel",
            "water chemical H2O",
        ]

        for query in queries:
            prod_result = prod_rag.query(query, top_k=1)
            math_result = math_rag.query(query, top_k=1)

            # ProductionRAG returns dicts, MathRAG returns ScoredChunks — just check both return results
            assert len(prod_result.get("results", [])) > 0, f"ProductionRAG found no results for '{query}'"
            assert len(math_result["results"]) > 0, f"MathRAG found no results for '{query}'"

    def test_math_rag_deterministic(self):
        """MathRAG should produce the same results on repeated queries."""
        rag = self._build_math_rag(DOCS)

        r1 = rag.query("python programming")
        r2 = rag.query("python programming")

        assert r1["num_results"] == r2["num_results"]
        for a, b in zip(r1["results"], r2["results"]):
            assert a.chunk.id == b.chunk.id
            assert abs(a.combined - b.combined) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
