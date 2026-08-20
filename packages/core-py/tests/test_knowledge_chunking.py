"""Tests for domains.learner.knowledge - chunking functions and _topic_slug."""

from domains.learner.knowledge import (
    _topic_slug, chunk_by_fixed_size, chunk_by_paragraph,
    chunk_by_heading, chunk_by_semantic, chunk_text,
)


class TestTopicSlug:
    def test_basic(self):
        assert _topic_slug("Machine Learning") == "machine_learning"

    def test_special_chars(self):
        slug = _topic_slug("AI/ML & Deep Learning")
        assert slug == "ai_ml_deep_learning"

    def test_truncate(self):
        slug = _topic_slug("a" * 100)
        assert len(slug) == 64

    def test_empty(self):
        assert _topic_slug("") == ""


class TestChunkByFixedSize:
    def test_empty(self):
        assert chunk_by_fixed_size("") == []
        assert chunk_by_fixed_size("  ") == []

    def test_short_text(self):
        assert chunk_by_fixed_size("hello", chunk_size=100) == ["hello"]

    def test_long_text(self):
        text = "a" * 200
        chunks = chunk_by_fixed_size(text, chunk_size=100, overlap=0)
        assert len(chunks) == 2

    def test_overlap(self):
        text = "abcdefghij" * 10
        chunks = chunk_by_fixed_size(text, chunk_size=30, overlap=10)
        assert len(chunks) > 1


class TestChunkByParagraph:
    def test_empty(self):
        assert chunk_by_paragraph("") == []
        assert chunk_by_paragraph("  ") == []

    def test_single_para(self):
        assert len(chunk_by_paragraph("hello world")) == 1

    def test_para_splitting(self):
        text = ("paragraph one " * 20).strip() + "\n\n" + ("paragraph two " * 20).strip()
        chunks = chunk_by_paragraph(text, max_chunk_size=200)
        assert len(chunks) >= 2


class TestChunkByHeading:
    def test_empty(self):
        assert chunk_by_heading("") == []

    def test_no_headings(self):
        text = "plain text without headings at all"
        chunks = chunk_by_heading(text)
        assert len(chunks) >= 1

    def test_long_headed(self):
        h1 = ("Content under the intro section. " * 30).strip()
        h2 = ("Content under the details section. " * 30).strip()
        text = f"# Intro\n{h1}\n\n## Details\n{h2}"
        chunks = chunk_by_heading(text)
        assert len(chunks) >= 1
        assert any("# Intro" in c for c in chunks)


class TestChunkBySemantic:
    def test_empty(self):
        assert chunk_by_semantic("") == []

    def test_short(self):
        chunks = chunk_by_semantic("hello")
        assert len(chunks) >= 1

    def test_long(self):
        text = "This is a sentence about AI. " * 50
        chunks = chunk_by_semantic(text)
        assert len(chunks) >= 1


class TestChunkText:
    def test_auto(self):
        text = "Hello world. " * 100
        chunks = chunk_text(text, strategy="auto")
        assert len(chunks) >= 1

    def test_fixed(self):
        text = "a" * 200
        chunks = chunk_text(text, strategy="fixed", chunk_size=100)
        assert len(chunks) >= 2

    def test_paragraph(self):
        text = "para one\n\npara two"
        chunks = chunk_text(text, strategy="paragraph")
        assert len(chunks) >= 1

    def test_heading(self):
        text = "# Title\nBody"
        chunks = chunk_text(text, strategy="heading")
        assert len(chunks) >= 1
