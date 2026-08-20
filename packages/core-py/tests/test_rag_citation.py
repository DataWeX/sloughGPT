"""Tests for domains.cognitive.rag — CitationTracker."""

from domains.cognitive.rag import CitationTracker, TextChunk


class TestCitationTracker:
    def test_extract_claims(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a programming language.")
        assert len(claims) >= 1
        assert claims[0]["subject"] == "Python"
        assert "programming language" in claims[0]["predicate"]

    def test_extract_multiple(self):
        ct = CitationTracker()
        text = "Python is great. Java is popular."
        claims = ct.extract_claims(text)
        assert len(claims) == 2

    def test_extract_no_claims(self):
        ct = CitationTracker()
        claims = ct.extract_claims("hello world")
        assert len(claims) == 0

    def test_cite(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language.")
        chunk = TextChunk(id="c1", content="Python source", metadata={"source": "doc"})
        cited = ct.cite(claims[0], [chunk])
        assert cited["supported"] is True
        assert len(cited["sources"]) == 1

    def test_format_citations(self):
        ct = CitationTracker()
        ct.extract_claims("Python is a language.")
        output = ct.format_citations()
        assert "[1]" in output
        assert "Python" in output
