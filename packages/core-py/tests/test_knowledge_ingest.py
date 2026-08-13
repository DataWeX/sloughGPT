"""Tests for knowledge.py ingestion pipeline — chunking, scraping, ingestor, async branches."""

import asyncio
import json
import sys
import time
import types

import pytest

from domains import learner
from domains.learner import knowledge as K
from domains.inference.vector_store import VectorEntry

from domains.learner.knowledge import (
    KnowledgeFact,
    KnowledgeMemory,
    KnowledgeIngestor,
    FeedSubscription,
    _topic_slug,
    chunk_by_fixed_size,
    chunk_by_paragraph,
    chunk_by_heading,
    chunk_by_semantic,
    chunk_text,
    _extract_topics,
    _extract_facts_from_text,
    _scrape_article,
    _search_ddg,
    get_knowledge_memory,
    get_knowledge_ingestor,
)


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(K, "FEED_STATE_PATH", tmp_path / "feeds.json")
    monkeypatch.setattr(K, "VISITED_PATH", tmp_path / "visited.json")
    monkeypatch.setattr(K, "ENTRIES_PATH", tmp_path / "entries.json")
    from domains.learner import data_filter as df
    monkeypatch.setattr(df, "FILTER_CONFIG_PATH", tmp_path / "filter_config.json")
    monkeypatch.setattr(K, "_knowledge_memory", None)
    monkeypatch.setattr(K, "_knowledge_ingestor", None)


# ── helpers ────────────────────────────────────────────────────────────


class TestTopicSlug:
    def test_normalizes(self):
        assert _topic_slug("AI & Machine Learning!") == "ai_machine_learning_"

    def test_lowercases_and_trims(self):
        assert _topic_slug("  Python  ") == "python"

    def test_capped_at_64(self):
        assert len(_topic_slug("x" * 200)) == 64


# ── chunking ───────────────────────────────────────────────────────────


class TestChunkFixed:
    def test_empty(self):
        assert chunk_by_fixed_size("") == []
        assert chunk_by_fixed_size("   ") == []

    def test_short_returns_whole(self):
        assert chunk_by_fixed_size("short text") == ["short text"]

    def test_splits_with_overlap(self):
        text = "word " * 300
        chunks = chunk_by_fixed_size(text, chunk_size=500, overlap=50)
        assert len(chunks) > 1
        assert all(0 < len(c) <= 500 for c in chunks)

    def test_overlap_gte_chunk_size(self):
        text = "word " * 300
        chunks = chunk_by_fixed_size(text, chunk_size=500, overlap=600)
        assert len(chunks) >= 2


class TestChunkParagraph:
    def test_empty(self):
        assert chunk_by_paragraph("") == []

    def test_merges_short_paragraphs(self):
        text = "Para one is short.\n\nPara two is also short."
        chunks = chunk_by_paragraph(text)
        assert len(chunks) == 1
        assert "Para one" in chunks[0] and "Para two" in chunks[0]

    def test_splits_large_paragraph(self):
        text = "\n\n".join("A" * 600 for _ in range(4))
        chunks = chunk_by_paragraph(text, max_chunk_size=1000)
        assert len(chunks) >= 2

    def test_whitespace_only_paragraphs_ignored(self):
        text = "First real paragraph.\n\n   \n\nSecond real paragraph."
        chunks = chunk_by_paragraph(text)
        assert len(chunks) == 1


class TestChunkHeading:
    def test_empty(self):
        assert chunk_by_heading("") == []

    def test_splits_by_heading(self):
        text = "# Intro\nSome intro content here.\n## Details\nMore details here."
        chunks = chunk_by_heading(text)
        assert any("# Intro" in c for c in chunks)
        assert any("## Details" in c for c in chunks)

    def test_plain_text_single_section(self):
        text = "no headings here just plain content " * 40
        chunks = chunk_by_heading(text, max_chunk_size=500)
        assert len(chunks) == 1

    def test_heading_only_text_falls_back_to_fixed(self):
        assert chunk_by_heading("# A\n## B\n### C") == ["# A\n## B\n### C"]

    def test_merges_small_sections(self):
        text = "# A\nshort\n# B\nbrief"
        chunks = chunk_by_heading(text)
        assert any("# A" in c and "# B" in c for c in chunks)

    def test_splits_oversized_merged_sections(self):
        text = "# A\n" + "x" * 200 + "\n# B\n" + "y" * 200
        chunks = chunk_by_heading(text, max_chunk_size=300)
        assert len(chunks) == 2
        assert any("# A" in c for c in chunks)
        assert any("# B" in c for c in chunks)


class TestChunkSemantic:
    def test_empty(self):
        assert chunk_by_semantic("") == []

    def test_few_sentences_kept_together(self):
        text = "First sentence here. Second sentence here."
        assert chunk_by_semantic(text) == [text.strip()]

    def test_splits_long_chunks(self):
        sentences = ["The quick brown fox jumps over the lazy dog near the river bank."] * 30
        text = " ".join(sentences)
        chunks = chunk_by_semantic(text, max_chunk_size=300, min_chunk_size=50)
        assert len(chunks) > 1

    def test_short_sentence_natural_break(self):
        text = (
            "The quick brown fox jumps over the lazy dog near the river. "
            "Short. "
            "Another reasonably long sentence about the topic at hand here."
        )
        chunks = chunk_by_semantic(text, min_chunk_size=20)
        assert len(chunks) >= 2

    def test_low_similarity_break(self):
        text = (
            "Quantum entanglement causes particles to correlate instantly. "
            "Baking soda volcanoes erupt with colorful foam. "
            "Corals build reefs in warm tropical waters."
        )
        chunks = chunk_by_semantic(text, min_chunk_size=20)
        assert len(chunks) >= 2

    def test_force_split_with_small_current(self):
        text = "Short. " + "Very long declarative sentence about substantial topics " * 5
        text += ". " + "Another very long declarative sentence about separate topics " * 5
        chunks = chunk_by_semantic(text, max_chunk_size=150, min_chunk_size=100)
        assert len(chunks) >= 2


class TestChunkText:
    def test_auto_heading(self):
        out = chunk_text("# H\ncontent here", strategy="auto")
        assert any("# H" in c for c in out)

    def test_auto_paragraph(self):
        text = "p1\n\np2\n\np3\n\np4"
        out = chunk_text(text, strategy="auto")
        assert len(out) >= 1

    def test_auto_long_semantic(self):
        text = ("A reasonably long declarative sentence about the subject." * 60)
        out = chunk_text(text, strategy="auto")
        assert len(out) >= 1

    def test_auto_short_fixed(self):
        out = chunk_text("short plain content")
        assert out == ["short plain content"]

    def test_explicit_strategies(self):
        for strategy in ("fixed", "paragraph", "heading", "semantic"):
            out = chunk_text("Some content here for strategy.", strategy=strategy)
            assert out

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            chunk_text("x", strategy="bogus")


class TestExtractTopics:
    def test_extracts_keywords(self):
        topics = _extract_topics("Python programming language for web development")
        assert topics
        assert "python" in topics

    def test_stopwords_excluded(self):
        topics = _extract_topics("the and for they that this")
        assert topics == []

    def test_respects_max(self):
        topics = _extract_topics("alpha beta gamma delta epsilon zeta", max_topics=3)
        assert len(topics) <= 3


class TestExtractFacts:
    def test_imperative_skipped(self):
        assert _extract_facts_from_text("Try running this command now.") == []

    def test_exclamatory_skipped(self):
        assert _extract_facts_from_text("This is an absolutely incredible discovery!") == []

    def test_has_pattern(self):
        facts = _extract_facts_from_text("This system has multiple components inside.")
        assert any("This system has multiple components inside." == f for f in facts)

    def test_can_pattern(self):
        facts = _extract_facts_from_text("Python can run on any platform today.")
        assert any("Python can run on any platform today." == f for f in facts)

    def test_short_sentence_skipped(self):
        facts = _extract_facts_from_text("It is fine.")
        assert facts == []

    def test_wh_question_skipped(self):
        assert _extract_facts_from_text("Where does this happen?") == []

    def test_long_wh_question_skipped(self):
        text = "What is the capital of France and why is it important?"
        assert _extract_facts_from_text(text) == []

    def test_long_imperative_skipped(self):
        text = "Try running this command right now and observe what happens."
        assert _extract_facts_from_text(text) == []


# ── scraping / search ──────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class _FakeP:
    def __init__(self, text):
        self._text = text

    def get_text(self, strip=False):
        return self._text


class _FakeSoup:
    def __init__(self, html, parser=None):
        self._html = html
        self.decomposed = 0

    def __call__(self, tags):
        return [types.SimpleNamespace(decompose=self._decompose)]

    def _decompose(self):
        self.decomposed += 1

    def find_all(self, name):
        return [_FakeP("Some sufficiently long paragraph text about the article contents here.")]

    def decompose(self):
        pass


def _patch_httpx(monkeypatch, response=None, error=None):
    import httpx

    def fake_get(*args, **kwargs):
        if error:
            raise error
        return response

    monkeypatch.setattr(httpx, "get", fake_get)


def _install_fake(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


class TestScrapeArticle:
    def test_uses_trafilatura(self, monkeypatch):
        _patch_httpx(monkeypatch, _FakeResp("<html>page</html>"))
        _install_fake(
            monkeypatch, "trafilatura",
            extract=lambda html, include_tables=False, include_images=False,
            no_fallback=False: "Extracted article text with more than fifty characters in total.",
        )
        out = _scrape_article("https://example.com")
        assert out.startswith("Extracted article")

    def test_bs4_fallback(self, monkeypatch):
        _patch_httpx(monkeypatch, _FakeResp("<html><p>para</p></html>"))
        _install_fake(
            monkeypatch, "trafilatura",
            extract=lambda *a, **k: None,
        )
        _install_fake(monkeypatch, "bs4", BeautifulSoup=_FakeSoup)
        out = _scrape_article("https://example.com")
        assert "paragraph text" in out

    def test_error_returns_empty(self, monkeypatch, caplog):
        _patch_httpx(monkeypatch, error=RuntimeError("network down"))
        _install_fake(
            monkeypatch, "trafilatura",
            extract=lambda *a, **k: "",
        )
        assert _scrape_article("https://example.com") == ""
        assert any("Article scrape failed" in r.message for r in caplog.records)


_SEARCH_HTML = """
<a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=abc">Example Title</a>
<div class="result__snippet">A snippet of example content.</div>
<a class="result__a" href="https://other.com/article">Other Article</a>
<div class="result__snippet">Another snippet here.</div>
"""


class TestSearchDDG:
    def test_parses_results_with_uddg(self, monkeypatch):
        _patch_httpx(monkeypatch, _FakeResp(_SEARCH_HTML))
        results = _search_ddg("test query", max_results=5)
        assert any(r["url"] == "https://example.com/page" for r in results)
        assert any(r["url"] == "https://other.com/article" for r in results)
        assert all(r["title"] for r in results)

    def test_respects_max_results(self, monkeypatch):
        _patch_httpx(monkeypatch, _FakeResp(_SEARCH_HTML))
        assert len(_search_ddg("q", max_results=1)) == 1

    def test_error_returns_empty(self, monkeypatch, caplog):
        _patch_httpx(monkeypatch, error=RuntimeError("timeout"))
        assert _search_ddg("q") == []
        assert any("DDG search failed" in r.message for r in caplog.records)


# ── KnowledgeMemory async/sync branches ────────────────────────────────


class _AsyncResult:
    def __init__(self, eid, text, vector, metadata, score=0.9):
        self.id = eid
        self.text = text
        self.vector = vector
        self.metadata = metadata
        self.score = score


class FakeAsyncStore:
    """Vector store exposing only async methods."""

    def __init__(self, dimension=8):
        self.dimension = dimension
        self._entries = {}

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def upsert(self, entries):
        for e in entries:
            self._entries[e.id] = e

    async def query(self, vector=None, top_k=10, filter_metadata=None):
        results = []
        for e in self._entries.values():
            if filter_metadata and e.metadata.get("topic") != filter_metadata.get("topic"):
                continue
            results.append(_AsyncResult(e.id, e.text, e.vector, e.metadata))
        return results[:top_k]

    async def count(self):
        return len(self._entries)

    async def delete(self, ids):
        for i in ids:
            self._entries.pop(i, None)
        return True


class _BrokenStore:
    dimension = 8

    def __init__(self, async_raise=False):
        self._async_raise = async_raise
        self._entries = {}

    async def connect(self):
        raise RuntimeError("connect failed")

    async def upsert(self, entries):
        raise RuntimeError("upsert failed")

    async def query(self, *a, **k):
        raise RuntimeError("query failed")

    async def count(self):
        raise RuntimeError("count failed")

    async def delete(self, ids):
        raise RuntimeError("delete failed")


@pytest.fixture
def amem():
    store = FakeAsyncStore()
    mem = KnowledgeMemory(vector_store=store, load_persisted=False)
    return mem, store


class TestKnowledgeMemoryAsync:
    def test_async_add_and_list(self, amem):
        mem, store = amem
        assert mem.add_fact(KnowledgeFact(content="Alpha fact", topic="t1")) is True
        items = mem.list_all()
        assert len(items) == 1
        assert items[0]["content"] == "Alpha fact"

    def test_async_search(self, amem):
        mem, store = amem
        mem._embed_fn = lambda text: [0.1] * 8
        mem.add_fact(KnowledgeFact(content="Beta fact content", topic="t1"))
        results = mem.search("beta")
        assert any("Beta fact content" == r["content"] for r in results)

    def test_async_query_filter_topic(self, amem):
        mem, store = amem
        mem.add_fact(KnowledgeFact(content="One", topic="a"))
        mem.add_fact(KnowledgeFact(content="Two", topic="b"))
        results = mem.query("a", top_k=10)
        assert len(results) == 1
        assert results[0]["content"] == "One"

    def test_async_stats(self, amem):
        mem, store = amem
        mem.add_fact(KnowledgeFact(content="Stat fact", topic="a"))
        assert mem.stats()["total_facts"] == 1

    def test_async_delete_and_clear(self, amem):
        mem, store = amem
        mem.add_fact(KnowledgeFact(content="Delete target", topic="a"))
        item_id = mem.list_all()[0]["id"]
        assert mem.delete_by_id(item_id) is True
        assert mem.list_all() == []
        mem.add_fact(KnowledgeFact(content="Clear target", topic="a"))
        assert mem.clear_all() == 1
        assert mem.list_all() == []

    def test_all_topics_and_get_topic_facts(self, amem):
        mem, store = amem
        assert mem.all_topics() == ["general"]
        assert mem.get_topic_facts("anything") == []

    def test_context_string_empty(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        assert mem.get_context_string() == ""


class TestKnowledgeMemoryErrorPaths:
    def test_connect_failure_ignored(self):
        mem = KnowledgeMemory(vector_store=_BrokenStore(), load_persisted=False)

    def test_upsert_failure_returns_true(self, caplog):
        store = _BrokenStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        assert mem.add_fact(KnowledgeFact(content="will fail upsert")) is True
        assert any("Vector store upsert failed" in r.message for r in caplog.records)

    def test_query_failure_returns_empty(self, caplog):
        store = _BrokenStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        assert mem.query("t") == []
        assert mem.search("q") == []
        assert mem.list_all() == []
        assert any("Vector store query failed" in r.message for r in caplog.records)

    def test_stats_failure_returns_zero(self):
        store = _BrokenStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        assert mem.stats()["total_facts"] == 0

    def test_delete_failure_returns_false(self, caplog):
        store = _BrokenStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        assert mem.delete_by_id("x") is False
        assert mem.clear_all() == 0
        assert any("delete_by_id failed" in r.message for r in caplog.records)


class TestAddArticle:
    def test_adds_chunks_across_topics(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        content = "Quantum computing uses superposition for fast calculations. " * 5
        added = mem.add_article("https://a.com", "Quantum Computing Advances", content, source="article")
        assert added > 0
        assert len(mem.list_all()) == added

    def test_empty_topics_falls_back_to_general(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        added = mem.add_article("https://a.com", "", "the and for they that this", source="article")
        assert added == 1
        assert mem.list_all()[0]["topic"] == "general"

    def test_whitespace_chunks_skipped(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        content = "A" * 500 + " " * 500
        added = mem.add_article("https://a.com", "Title", content, source="article")
        assert added == 1

    def test_chunk_filter_rejects_all(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        content = "Quantum computing uses superposition for fast calculations. " * 5
        added = mem.add_article(
            "https://a.com", "Title", content, source="article",
            chunk_filter=lambda t, topic: False,
        )
        assert added == 0


class TestAutoIngestChat:
    def test_short_fragments_skipped(self, monkeypatch):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        monkeypatch.setattr(K, "_extract_facts_from_text", lambda text: ["short"])
        assert mem.auto_ingest_from_chat("user", "assistant") == 0

    def test_ingests_facts_with_topic(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        response = ("Python is a high level programming language. "
                    "NumPy provides fast array operations for science.")
        added = mem.auto_ingest_from_chat("tell me about python and numpy", response)
        assert added >= 1
        topics = {f["topic"] for f in mem.list_all()}
        assert topics  # non-empty topic set inferred from user message

    def test_ingest_from_chat_returns_stored_texts(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        response = ("Python is a high level programming language. "
                    "NumPy provides fast array operations for science.")
        stored = mem.ingest_from_chat("tell me about python and numpy", response)
        assert len(stored) >= 1
        assert any("Python is a high level" in f for f in stored)

    def test_ingest_from_chat_empty_for_short_fragments(self, monkeypatch):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        monkeypatch.setattr(K, "_extract_facts_from_text", lambda text: ["short"])
        assert mem.ingest_from_chat("user", "assistant") == []

    def test_ingest_from_chat_empty_for_duplicate(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        response = ("Photosynthesis is the process plants use to convert light "
                    "into chemical energy stored in glucose.")
        assert len(mem.ingest_from_chat("explain photosynthesis", response)) >= 1
        assert mem.ingest_from_chat("explain photosynthesis", response) == []

    def test_ingest_from_chat_respects_max_facts(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        response = ("Fact alpha is the first declarative statement here. "
                    "Fact beta is the second declarative statement here. "
                    "Fact gamma is the third declarative statement here. "
                    "Fact delta is the fourth declarative statement here.")
        stored = mem.ingest_from_chat("list facts", response, max_facts=2)
        assert len(stored) == 2


class TestRunAsync:
    def test_without_running_loop(self):
        async def coro():
            return 42

        assert KnowledgeMemory._run_async(coro()) == 42

    @pytest.mark.asyncio
    async def test_inside_running_loop(self):
        async def coro():
            return "threaded"

        result = KnowledgeMemory._run_async(coro())
        assert result == "threaded"

    @pytest.mark.asyncio
    async def test_inside_running_loop_propagates_error(self):
        async def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            KnowledgeMemory._run_async(bad())


class TestEmbedFn:
    def test_custom_embed_fn_used(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        mem._embed_fn = lambda text: [0.5] * 8
        mem.add_fact(KnowledgeFact(content="Embedded via custom fn", topic="t"))
        assert mem._get_embedding("anything") == [0.5] * 8


class TestPersistence:
    def test_load_visited_corrupt_file(self):
        K.VISITED_PATH.write_text("{bad json")
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=True)
        assert mem._visited == set()

    def test_load_entries_wrong_dimension_skipped(self, caplog):
        store = FakeAsyncStore(dimension=8)
        data = [
            {"id": "a", "vector": [0.0] * 8, "text": "ok", "metadata": {}},
            {"id": "b", "vector": [0.0] * 3, "text": "wrong dim", "metadata": {}},
        ]
        K.ENTRIES_PATH.write_text(json.dumps(data))
        mem = KnowledgeMemory(vector_store=store, load_persisted=True)
        items = mem.list_all()
        assert len(items) == 1
        assert items[0]["content"] == "ok"

    def test_load_entries_missing_file_returns(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=True)
        assert mem.list_all() == []

    def test_migrate_from_json_topics(self):
        topics_dir = K.KNOWLEDGE_DIR / "topics"
        topics_dir.mkdir()
        (topics_dir / "science.json").write_text(json.dumps([
            {"content": "Water boils at 100 degrees Celsius.", "topic": "science"},
            {"content": "Gravity keeps planets in orbit.", "topic": "science"},
        ]))
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=True)
        items = mem.list_all()
        assert len(items) == 2
        assert (K.KNOWLEDGE_DIR / ".migrated_to_vector").exists()

    def test_migrate_skipped_when_marker_exists(self):
        topics_dir = K.KNOWLEDGE_DIR / "topics"
        topics_dir.mkdir(parents=True)
        (K.KNOWLEDGE_DIR / ".migrated_to_vector").write_text("{}")
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=True)
        assert mem.list_all() == []

    def test_migrate_no_dir_returns(self):
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=True)
        assert mem.list_all() == []

    def test_migrate_corrupt_file_logs_and_continues(self, caplog):
        topics_dir = K.KNOWLEDGE_DIR / "topics"
        topics_dir.mkdir(parents=True)
        (topics_dir / "broken.json").write_text("{not json")
        (topics_dir / "good.json").write_text(json.dumps([
            {"content": "Corals form reefs in warm oceans.", "topic": "ocean"},
        ]))
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=True)
        assert len(mem.list_all()) == 1
        assert (K.KNOWLEDGE_DIR / ".migrated_to_vector").exists()

    def test_load_entries_empty_data_returns(self):
        K.ENTRIES_PATH.write_text("[]")
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=True)
        assert mem.list_all() == []

    def test_load_entries_corrupt_logs_warning(self, caplog):
        K.ENTRIES_PATH.write_text("{not json")
        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=True)
        assert mem.list_all() == []
        assert any("Failed to load entries" in r.message for r in caplog.records)

    def test_save_entries_failure_logs_warning(self, caplog):
        class _BadPath:
            def write_text(self, *a, **k):
                raise OSError("disk full")

        store = FakeAsyncStore()
        mem = KnowledgeMemory(vector_store=store, load_persisted=False)
        original = K.ENTRIES_PATH
        try:
            K.ENTRIES_PATH = _BadPath()
            mem.add_fact(KnowledgeFact(content="won't persist", topic="t"))
        finally:
            K.ENTRIES_PATH = original
        assert any("Failed to save entries" in r.message for r in caplog.records)

    def test_default_store_connect_failure_ignored(self, monkeypatch):
        class _ConnFailStore:
            dimension = 8

            def __init__(self, dimension):
                self.dimension = dimension

            async def connect(self):
                raise RuntimeError("no connect")

            async def disconnect(self):
                pass

        monkeypatch.setattr(
            "domains.inference.vector_store.InMemoryVectorStore", _ConnFailStore
        )
        mem = KnowledgeMemory(vector_store=None, load_persisted=False)
        assert mem._vector_store is not None


# ── KnowledgeIngestor ──────────────────────────────────────────────────


class _FakeMem:
    def __init__(self):
        self._visited = set()
        self.added = []

    def search(self, text, top_k=5):
        return []

    def add_article(self, url, title, content, source, chunk_filter=None):
        self.added.append((url, title))
        return 1


class _FakeFilter:
    def __init__(self, passes=True):
        self._passes = passes

    def filter_article(self, url, title, content, existing_facts=None):
        return (self._passes, "" if self._passes else "low_quality")

    def filter_chunk(self, text, topic):
        return True

    def get_stats(self):
        return {"total_seen": 1}


class _FakeFeedEntry:
    def __init__(self, link, title, summary):
        self._link = link
        self._title = title
        self._summary = summary

    def get(self, key, default=""):
        return {"link": self._link, "title": self._title,
                "summary": self._summary, "description": self._summary}.get(key, default)


@pytest.fixture
def ingestor():
    mem = _FakeMem()
    filt = _FakeFilter()
    inst = KnowledgeIngestor(memory=mem, filter_instance=filt)
    return inst, mem, filt


class TestIngestorFeeds:
    def test_load_feeds_from_file(self, ingestor, monkeypatch):
        data = [{"url": "https://feed.example/rss", "title": "Feed", "poll_interval": 3600}]
        K.FEED_STATE_PATH.write_text(json.dumps(data))
        inst, mem, filt = ingestor
        inst._load_feeds()
        inst._feeds = inst._load_feeds()
        assert len(inst._feeds) == 1
        assert inst._feeds[0].url == "https://feed.example/rss"

    def test_load_feeds_corrupt(self, ingestor, caplog):
        K.FEED_STATE_PATH.write_text("{bad")
        inst, mem, filt = ingestor
        assert inst._load_feeds() == []

    def test_subscribe_and_list(self, ingestor):
        inst, mem, filt = ingestor
        assert inst.subscribe_feed("https://feed.example/rss") is True
        assert inst.subscribe_feed("https://feed.example/rss") is False
        assert len(inst.list_feeds()) == 1

    def test_unsubscribe(self, ingestor):
        inst, mem, filt = ingestor
        inst.subscribe_feed("https://feed.example/rss")
        assert inst.unsubscribe_feed("https://feed.example/rss") is True
        assert inst.list_feeds() == []

    def test_fetch_feed_uses_feedparser(self, ingestor, monkeypatch):
        fake_parsed = types.SimpleNamespace(entries=[
            _FakeFeedEntry("https://e.com/1", "Title 1", "<b>Summary</b> 1"),
        ])

        def fake_parse(url):
            return fake_parsed

        mod = types.ModuleType("feedparser")
        mod.parse = fake_parse
        monkeypatch.setitem(sys.modules, "feedparser", mod)
        inst, mem, filt = ingestor
        articles = inst._fetch_feed(FeedSubscription(url="https://feed.example/rss"))
        assert len(articles) == 1
        assert articles[0]["url"] == "https://e.com/1"
        assert "<b>" not in articles[0]["summary"]

    def test_fetch_feed_error(self, ingestor, monkeypatch, caplog):
        def fake_parse(url):
            raise RuntimeError("parse error")

        mod = types.ModuleType("feedparser")
        mod.parse = fake_parse
        monkeypatch.setitem(sys.modules, "feedparser", mod)
        inst, mem, filt = ingestor
        assert inst._fetch_feed(FeedSubscription(url="x")) == []


class TestIngestorPollFeeds:
    def test_disabled_feed_skipped(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        inst._feeds = [FeedSubscription(url="x", enabled=False)]
        monkeypatch.setattr(inst, "_fetch_feed", lambda f: [{"url": "u", "title": "t", "summary": "s"}])
        result = inst.poll_feeds()
        assert result["new_articles"] == 0

    def test_not_due_skipped(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        inst._feeds = [FeedSubscription(url="x", last_fetched=time.time(), poll_interval=3600)]
        monkeypatch.setattr(inst, "_fetch_feed", lambda f: [{"url": "u", "title": "t", "summary": "s"}])
        assert inst.poll_feeds()["new_articles"] == 0

    def test_ingests_due_articles(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        inst._feeds = [FeedSubscription(url="x", last_fetched=0, poll_interval=3600)]
        monkeypatch.setattr(
            inst, "_fetch_feed",
            lambda f: [{"url": "https://e.com/1", "title": "T1", "summary": "s1"}],
        )
        monkeypatch.setattr(K, "_scrape_article", lambda url: "Good article content with plenty of words.")
        result = inst.poll_feeds()
        assert result["new_articles"] == 1
        assert mem.added

    def test_rejected_article_marked_visited(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        filt._passes = False
        inst._feeds = [FeedSubscription(url="x", last_fetched=0, poll_interval=3600)]
        monkeypatch.setattr(
            inst, "_fetch_feed",
            lambda f: [{"url": "https://e.com/2", "title": "T2", "summary": "s2"}],
        )
        monkeypatch.setattr(K, "_scrape_article", lambda url: "Good content here.")
        result = inst.poll_feeds()
        assert result["rejected"] == 1
        assert inst._is_visited("https://e.com/2")

    def test_visited_article_skipped(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        inst._mark_visited("https://e.com/3")
        inst._feeds = [FeedSubscription(url="x", last_fetched=0, poll_interval=3600)]
        monkeypatch.setattr(
            inst, "_fetch_feed",
            lambda f: [{"url": "https://e.com/3", "title": "T3", "summary": "s3"}],
        )
        monkeypatch.setattr(K, "_scrape_article", lambda url: "content")
        result = inst.poll_feeds()
        assert result["new_articles"] == 0
        assert not mem.added

    def test_empty_scrape_uses_summary(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        inst._feeds = [FeedSubscription(url="x", last_fetched=0, poll_interval=3600)]
        monkeypatch.setattr(
            inst, "_fetch_feed",
            lambda f: [{"url": "https://e.com/4", "title": "T4", "summary": "Summary fallback text."}],
        )
        monkeypatch.setattr(K, "_scrape_article", lambda url: "")
        result = inst.poll_feeds()
        assert result["new_articles"] == 1
        assert mem.added


class TestIngestorSearch:
    def test_search_and_ingest(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        monkeypatch.setattr(
            K, "_search_ddg",
            lambda q, max_results=5: [{"url": "https://s.com/1", "title": "S1", "snippet": "snip"}],
        )
        monkeypatch.setattr(K, "_scrape_article", lambda url: "Search article content here.")
        result = inst.search_and_ingest("python")
        assert result["new_facts"] >= 1
        assert inst._is_visited("https://s.com/1")

    def test_search_skips_visited(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        inst._mark_visited("https://s.com/2")
        monkeypatch.setattr(
            K, "_search_ddg",
            lambda q, max_results=5: [{"url": "https://s.com/2", "title": "S2", "snippet": "snip"}],
        )
        monkeypatch.setattr(K, "_scrape_article", lambda url: "content")
        result = inst.search_and_ingest("q")
        assert result["new_facts"] == 0

    def test_search_empty_scrape_uses_snippet(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        monkeypatch.setattr(
            K, "_search_ddg",
            lambda q, max_results=5: [{"url": "https://s.com/3", "title": "S3", "snippet": "Snippet fallback text."}],
        )
        monkeypatch.setattr(K, "_scrape_article", lambda url: "")
        result = inst.search_and_ingest("q")
        assert result["new_facts"] >= 1
        assert inst._is_visited("https://s.com/3")

    def test_search_rejected_article(self, ingestor, monkeypatch, caplog):
        inst, mem, filt = ingestor
        filt._passes = False
        monkeypatch.setattr(
            K, "_search_ddg",
            lambda q, max_results=5: [{"url": "https://s.com/4", "title": "S4", "snippet": "snip"}],
        )
        monkeypatch.setattr(K, "_scrape_article", lambda url: "content")
        result = inst.search_and_ingest("q")
        assert result["new_facts"] == 0
        assert result["rejected"] == 1


class TestIngestorUrl:
    def test_ingest_url_ok(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        monkeypatch.setattr(K, "_scrape_article", lambda url: "Title line\n" + "body " * 60)
        result = inst.ingest_url("https://d.com/1")
        assert result["status"] == "ok"
        assert result["new_facts"] >= 1
        assert inst._is_visited("https://d.com/1")

    def test_ingest_url_no_content(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        monkeypatch.setattr(K, "_scrape_article", lambda url: "")
        result = inst.ingest_url("https://d.com/2")
        assert result["status"] == "no_content"

    def test_ingest_url_short_content(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        monkeypatch.setattr(K, "_scrape_article", lambda url: "tiny")
        assert inst.ingest_url("https://d.com/3")["status"] == "no_content"

    def test_ingest_url_rejected(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        filt._passes = False
        monkeypatch.setattr(K, "_scrape_article", lambda url: "Title line\n" + "body " * 60)
        result = inst.ingest_url("https://d.com/4")
        assert result["status"] == "rejected"
        assert inst._is_visited("https://d.com/4")

    def test_ingest_url_visited_requery(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        inst._mark_visited("https://d.com/5")
        monkeypatch.setattr(K, "_scrape_article", lambda url: "Title line\n" + "body " * 60)
        result = inst.ingest_url("https://d.com/5")
        assert result["status"] == "ok"

    def test_ingest_url_visited_no_content(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        inst._mark_visited("https://d.com/6")
        monkeypatch.setattr(K, "_scrape_article", lambda url: "")
        result = inst.ingest_url("https://d.com/6")
        assert result["status"] == "already_visited"


class TestIngestorPolling:
    def test_start_guards_against_reentry(self, ingestor):
        inst, mem, filt = ingestor
        inst._running = True
        inst.start_background_polling()
        assert inst._feed_thread is None

    def test_start_stop_cycle(self, ingestor):
        inst, mem, filt = ingestor
        inst.start_background_polling(interval=1)
        assert inst._running is True
        assert inst._feed_thread is not None
        inst.stop_background_polling()
        assert inst._running is False

    def test_poll_loop_ingests(self, ingestor, monkeypatch):
        inst, mem, filt = ingestor
        inst._running = True
        state = {"n": 0}

        def fake_poll(max_articles=5):
            state["n"] += 1
            inst._running = False
            return {"new_articles": 1}

        monkeypatch.setattr(inst, "poll_feeds", fake_poll)
        monkeypatch.setattr(K.time, "sleep", lambda s: None)
        inst._poll_loop(0)
        assert state["n"] == 1

    def test_poll_loop_error_handled(self, ingestor, monkeypatch, caplog):
        inst, mem, filt = ingestor
        inst._running = True

        def fake_poll(max_articles=5):
            inst._running = False
            raise RuntimeError("poll boom")

        monkeypatch.setattr(inst, "poll_feeds", fake_poll)
        monkeypatch.setattr(K.time, "sleep", lambda s: None)
        inst._poll_loop(0)
        assert any("Background poll error" in r.message for r in caplog.records)


class TestSingleton:
    def test_get_knowledge_memory_creates_once(self):
        a = get_knowledge_memory()
        b = get_knowledge_memory()
        assert a is b
        assert isinstance(a, KnowledgeMemory)

    def test_get_knowledge_ingestor_creates_once(self):
        a = get_knowledge_ingestor()
        b = get_knowledge_ingestor()
        assert a is b
        assert isinstance(a, KnowledgeIngestor)
