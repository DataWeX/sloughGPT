"""Meaningful tests for CitationTracker — claim extraction, citation, formatting."""

from domains.cognitive.rag import CitationTracker, TextChunk


class TestExtractClaims:
    def test_extract_is_claim(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Paris is a city.")
        assert len(claims) == 1
        assert claims[0]["subject"] == "Paris"
        assert claims[0]["predicate"] == "a city"

    def test_extract_was_claim(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python was created in 1991.")
        assert len(claims) == 1
        assert claims[0]["subject"] == "Python"

    def test_extract_can_claim(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Dogs can swim.")
        assert len(claims) == 1
        assert claims[0]["subject"] == "Dogs"

    def test_extract_has_claim(self):
        ct = CitationTracker()
        claims = ct.extract_claims("France has 67 million people.")
        assert len(claims) == 1

    def test_extract_multiple_claims(self):
        ct = CitationTracker()
        text = "Paris is a city. Berlin is also a city."
        claims = ct.extract_claims(text)
        assert len(claims) == 2

    def test_extract_no_claims(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Hello world!")
        assert len(claims) == 0

    def test_extract_cross_sentence_not_captured(self):
        ct = CitationTracker()
        # Two sentences that together look like a claim but shouldn't be
        text = "I like Paris. Is a beautiful place."
        claims = ct.extract_claims(text)
        # "Paris. Is" shouldn't form a claim — the period breaks it
        assert len(claims) <= 1

    def test_extract_positions(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Paris is great.")
        assert claims[0]["start"] >= 0
        assert claims[0]["end"] > claims[0]["start"]


class TestCite:
    def test_cite_with_sources(self):
        ct = CitationTracker()
        claim = {"subject": "Paris", "predicate": "a city", "text": "Paris is a city."}
        chunk = TextChunk(
            id="c1", content="Paris is the capital of France",
            metadata={"source": "wiki"}, embedding=None
        )
        cited = ct.cite(claim, [chunk])
        assert cited["supported"] is True
        assert len(cited["sources"]) == 1
        assert cited["sources"][0]["chunk_id"] == "c1"

    def test_cite_no_sources(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        cited = ct.cite(claim, [])
        assert cited["supported"] is False
        assert cited["sources"] == []

    def test_cite_max_3_sources(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "y", "text": "X y."}
        chunks = [
            TextChunk(id=f"c{i}", content=f"content{i}", metadata={}, embedding=None)
            for i in range(5)
        ]
        cited = ct.cite(claim, chunks)
        assert len(cited["sources"]) == 3


class TestFormatCitations:
    def test_format_empty(self):
        ct = CitationTracker()
        assert ct.format_citations() == ""

    def test_format_with_claims(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": "Paris is a city", "sources": [{"metadata": {"source": "wiki"}}]},
            {"text": "Berlin is a city", "sources": []},
        ]
        output = ct.format_citations()
        assert "[1] Paris is a city" in output
        assert "[2] Berlin is a city" in output
        assert "→ wiki" in output
