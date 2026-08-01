"""Tests for knowledge.py chunkers, topic/fact extraction, and KnowledgeIngestor."""

import sys
import types

import pytest

import domains.learner.knowledge as knowledge
from domains.learner.knowledge import (
    DEFAULT_FEED_POLL_INTERVAL,
    FeedSubscription,
    KnowledgeIngestor,
    KnowledgeMemory,
    KnowledgeFact,
    _extract_facts_from_text,
    _extract_topics,
    _topic_slug,
    chunk_by_fixed_size,
    chunk_by_heading,
    chunk_by_paragraph,
    chunk_by_semantic,
    chunk_text,
)

ARTICLE_TEXT = (
    "Climate science has made major progress over the past decade. "
    "Researchers have published thousands of studies on warming patterns. "
    "The global temperature continues to rise each year. "
    "Scientists have confirmed rising sea levels across the world."
)


class TestTopicSlug:
    def test_lowercases_and_strips(self):
        assert _topic_slug("  Machine Learning  ") == "machine_learning"

    def test_replaces_special_chars(self):
        assert _topic_slug("What's New?") == "what_s_new_"

    def test_caps_at_64_chars(self):
        assert len(_topic_slug("a" * 100)) == 64

    def test_keeps_numbers(self):
        assert _topic_slug("Sector 7") == "sector_7"


class TestChunkFixedSize:
    def test_empty_returns_empty(self):
        assert chunk_by_fixed_size("") == []
        assert chunk_by_fixed_size("   ") == []

    def test_short_text_returns_single(self):
        text = "Hello world"
        assert chunk_by_fixed_size(text) == ["Hello world"]

    def test_splits_into_chunks(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_by_fixed_size(text, chunk_size=10, overlap=0)
        assert len(chunks) == 3
        assert "".join(chunks) == text

    def test_overlap_applied(self):
        chunks = chunk_by_fixed_size("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=5)
        assert len(chunks) == 6
        assert chunks[1] == "fghijklmno"

    def test_overlap_larger_than_chunk_no_loop(self):
        chunks = chunk_by_fixed_size("abcdefghij", chunk_size=5, overlap=8)
        assert chunks


class TestChunkParagraph:
    def test_empty_returns_empty(self):
        assert chunk_by_paragraph("") == []
        assert chunk_by_paragraph("\n\n  \n") == []

    def test_single_paragraph(self):
        assert chunk_by_paragraph("Just one paragraph.") == ["Just one paragraph."]

    def test_merges_short_paragraphs(self):
        chunks = chunk_by_paragraph("Alpha.\n\nBeta.\n\nGamma.")
        assert len(chunks) == 1
        assert "\n\n" in chunks[0]

    def test_splits_when_over_max(self):
        p1 = "This is the first long paragraph."
        p2 = "This is the second paragraph."
        chunks = chunk_by_paragraph(f"{p1}\n\n{p2}", max_chunk_size=30)
        assert chunks == [p1, p2]


class TestChunkHeading:
    def test_empty_returns_empty(self):
        assert chunk_by_heading("") == []

    def test_no_headings_keeps_text(self):
        assert chunk_by_heading("Plain text only here.") == ["Plain text only here."]

    def test_heading_only_falls_back_to_fixed(self):
        chunks = chunk_by_heading("# Intro")
        assert chunks == chunk_by_fixed_size("# Intro", 1500)

    def test_splits_on_headings(self):
        text = "# Intro\n\nSome intro text.\n\n## Details\n\nMore detailed content here."
        chunks = chunk_by_heading(text, max_chunk_size=50)
        assert len(chunks) == 2
        assert chunks[0].startswith("# Intro")
        assert chunks[1].startswith("## Details")

    def test_heading_within_max_keeps_heading(self):
        text = "# Title\n\nBody text under the title."
        chunks = chunk_by_heading(text, max_chunk_size=1500)
        assert len(chunks) == 1
        assert "# Title" in chunks[0]


class TestChunkSemantic:
    def test_empty_returns_empty(self):
        assert chunk_by_semantic("") == []

    def test_few_sentences_returns_single(self):
        text = "First sentence. Second sentence."
        assert chunk_by_semantic(text) == [text.strip()]

    def test_force_split_long_sentence(self):
        s0 = "A" * 80
        s1 = "B" * 30
        s2 = "C" * 5
        chunks = chunk_by_semantic(f"{s0}. {s1}. {s2}.", max_chunk_size=100, min_chunk_size=50)
        assert len(chunks) == 2
        assert chunks[0] == s0 + "."
        assert s2 in chunks[1]

    def test_splits_on_short_transition(self):
        s0 = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."
        s2 = "More alpha beta gamma delta content."
        chunks = chunk_by_semantic(f"{s0} Short. {s2}", max_chunk_size=1000, min_chunk_size=40)
        assert len(chunks) == 2
        assert chunks[0] == s0

    def test_splits_on_topic_shift(self):
        s0 = "Dogs chase cats around the yard."
        s1 = "Subatomic particle interactions have baffled physicists for generations."
        s2 = "The kennel keeps warm blankets inside."
        chunks = chunk_by_semantic(f"{s0} {s1} {s2}", max_chunk_size=300, min_chunk_size=20)
        assert len(chunks) == 3
        assert chunks[0] == s0
        assert chunks[1] == s1


class TestChunkText:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_text("hello", strategy="bogus")

    def test_dispatches_fixed(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        assert chunk_text(text, strategy="fixed", chunk_size=10, overlap=0) == chunk_by_fixed_size(text, 10, 0)

    def test_dispatches_paragraph(self):
        text = "One.\n\nTwo.\n\nThree."
        assert chunk_text(text, strategy="paragraph") == chunk_by_paragraph(text)

    def test_dispatches_heading(self):
        text = "# H\n\nBody."
        assert chunk_text(text, strategy="heading") == chunk_by_heading(text)

    def test_dispatches_semantic(self):
        text = "First sentence here. Second sentence there."
        assert chunk_text(text, strategy="semantic") == chunk_by_semantic(text)

    def test_auto_picks_heading(self):
        text = "# Intro\n\nSome intro text."
        assert chunk_text(text, strategy="auto") == chunk_by_heading(text)

    def test_auto_picks_paragraph(self):
        text = "One.\n\nTwo.\n\nThree.\n\nFour."
        assert chunk_text(text, strategy="auto") == chunk_by_paragraph(text)

    def test_auto_picks_semantic_for_long_text(self):
        sentence = "The atmosphere contains several key gases that affect global temperatures. "
        text = sentence * 60
        assert len(text) > 2000
        assert chunk_text(text, strategy="auto") == chunk_by_semantic(text)

    def test_auto_picks_fixed_for_short_text(self):
        text = "Just a short piece of text without structure."
        assert chunk_text(text, strategy="auto") == chunk_by_fixed_size(text)


class TestExtractFacts:
    def test_empty_or_short_returns_empty(self):
        assert _extract_facts_from_text("") == []
        assert _extract_facts_from_text("Too short.") == []

    def test_is_are_pattern(self):
        facts = _extract_facts_from_text("The capital of France is Paris.")
        assert facts == ["The capital of France is Paris."]

    def test_has_have_pattern(self):
        facts = _extract_facts_from_text("The company has opened offices in Berlin.")
        assert facts

    def test_numbers_pattern(self):
        facts = _extract_facts_from_text("The population of Tokyo reached 37 million in 2020.")
        assert facts

    def test_modal_pattern(self):
        facts = _extract_facts_from_text("People should recycle plastic containers.")
        assert facts

    def test_skips_questions(self):
        facts = _extract_facts_from_text("What is the capital of France?")
        assert facts == []

    def test_skips_imperatives(self):
        facts = _extract_facts_from_text("Try to focus on the main points of the talk.")
        assert facts == []

    def test_skips_short_sentences(self):
        facts = _extract_facts_from_text("It rains a lot. The soil is wet and cold.")
        assert len(facts) == 1


class TestExtractTopics:
    def test_strips_stopwords(self):
        topics = _extract_topics("the and for are not the puppies and kittens")
        assert "the" not in topics
        assert "puppies" in topics
        assert "kittens" in topics

    def test_respects_max_topics(self):
        topics = _extract_topics("apple banana cherry date elderberry fig grape", max_topics=3)
        assert len(topics) <= 3

    def test_returns_lowercase(self):
        topics = _extract_topics("Artificial Intelligence Advances")
        assert all(t == t.lower() for t in topics)

    def test_weights_longer_words(self):
        topics = _extract_topics("aeroplane flight")
        assert topics[0] == "aeroplane"

    def test_empty_returns_empty(self):
        assert _extract_topics("the and for") == []


class TestFeedSubscription:
    def test_defaults(self):
        f = FeedSubscription(url="https://example.com/feed")
        assert f.url == "https://example.com/feed"
        assert f.title == ""
        assert f.last_fetched == 0.0
        assert f.poll_interval == DEFAULT_FEED_POLL_INTERVAL
        assert f.enabled is True


class _FakeFilter:
    def __init__(self, allow=True):
        self.allow = allow

    def filter_article(self, url, title, content, existing_facts=None):
        return (self.allow, "ok" if self.allow else "blocked")

    def filter_chunk(self, text, topic):
        return self.allow

    def get_stats(self):
        return {"filtered": 0, "total": 0}


class TestKnowledgeIngestor:
    @pytest.fixture
    def ingestor(self, tmp_path, monkeypatch):
        monkeypatch.setattr(knowledge, "FEED_STATE_PATH", tmp_path / "feeds.json")
        monkeypatch.setattr(knowledge, "VISITED_PATH", tmp_path / "visited.json")
        monkeypatch.setattr(knowledge, "ENTRIES_PATH", tmp_path / "entries.json")
        mem = KnowledgeMemory(load_persisted=False)
        mem.clear_all()
        ing = KnowledgeIngestor(memory=mem, filter_instance=_FakeFilter(allow=True))
        yield ing
        ing.stop_background_polling()

    @pytest.fixture
    def feedparser(self, monkeypatch):
        fake = types.SimpleNamespace(
            parse=lambda url: types.SimpleNamespace(entries=[
                {"link": "https://example.com/1", "title": "First", "summary": "Summary text"},
                {"link": "https://example.com/2", "title": "Second", "summary": ""},
            ])
        )
        monkeypatch.setitem(sys.modules, "feedparser", fake)
        return fake

    def test_subscribe_feed(self, ingestor):
        assert ingestor.subscribe_feed("https://example.com/rss") is True
        feeds = ingestor.list_feeds()
        assert len(feeds) == 1
        assert feeds[0]["url"] == "https://example.com/rss"

    def test_subscribe_duplicate_returns_false(self, ingestor):
        assert ingestor.subscribe_feed("https://example.com/rss") is True
        assert ingestor.subscribe_feed("https://example.com/rss") is False

    def test_unsubscribe_feed(self, ingestor):
        ingestor.subscribe_feed("https://example.com/rss")
        assert ingestor.unsubscribe_feed("https://example.com/rss") is True
        assert ingestor.list_feeds() == []

    def test_feeds_persist_across_instances(self, ingestor):
        ingestor.subscribe_feed("https://example.com/rss", poll_interval=1234)
        ing2 = KnowledgeIngestor(
            memory=KnowledgeMemory(load_persisted=False),
            filter_instance=_FakeFilter(allow=True),
        )
        feeds = ing2.list_feeds()
        assert len(feeds) == 1
        assert feeds[0]["url"] == "https://example.com/rss"
        assert feeds[0]["poll_interval"] == 1234

    def test_fetch_feed_parses_entries(self, ingestor, monkeypatch):
        fake = types.SimpleNamespace(
            entries=[
                {
                    "link": "https://example.com/1",
                    "title": "  First  ",
                    "summary": "<p>Summary <b>text</b></p>",
                },
                {"link": "https://example.com/2", "title": "Second", "summary": ""},
            ]
        )
        monkeypatch.setitem(
            sys.modules, "feedparser", types.SimpleNamespace(parse=lambda url: fake)
        )
        articles = ingestor._fetch_feed(FeedSubscription(url="https://example.com/rss"))
        assert len(articles) == 2
        assert articles[0]["title"] == "First"
        assert "Summary" in articles[0]["summary"]
        assert "<" not in articles[0]["summary"]
        assert articles[1]["title"] == "Second"

    def test_fetch_feed_caps_at_20(self, ingestor, monkeypatch):
        entries = [{"link": f"https://example.com/{i}", "title": f"T{i}", "summary": ""} for i in range(25)]
        monkeypatch.setitem(
            sys.modules, "feedparser", types.SimpleNamespace(parse=lambda url: types.SimpleNamespace(entries=entries))
        )
        articles = ingestor._fetch_feed(FeedSubscription(url="https://example.com/rss"))
        assert len(articles) == 20

    def test_fetch_feed_skips_missing_title(self, ingestor, monkeypatch):
        fake = types.SimpleNamespace(entries=[{"link": "https://example.com/1", "title": "", "summary": ""}])
        monkeypatch.setitem(sys.modules, "feedparser", types.SimpleNamespace(parse=lambda url: fake))
        assert ingestor._fetch_feed(FeedSubscription(url="https://example.com/rss")) == []

    def test_poll_feeds_ingests_new_articles(self, ingestor, monkeypatch, feedparser):
        ingestor.subscribe_feed("https://example.com/rss")
        monkeypatch.setattr(
            knowledge, "_scrape_article",
            lambda url, timeout=15: ARTICLE_TEXT + f" Unique facts about {url}.",
        )
        result = ingestor.poll_feeds(max_articles=10)
        assert result["new_articles"] == 2
        assert result["rejected"] == 0
        assert ingestor._is_visited("https://example.com/1")

    def test_poll_feeds_rejects_blocked(self, ingestor, monkeypatch, feedparser):
        ingestor.filter = _FakeFilter(allow=False)
        ingestor.subscribe_feed("https://example.com/rss")
        monkeypatch.setattr(knowledge, "_scrape_article", lambda url, timeout=15: ARTICLE_TEXT)
        result = ingestor.poll_feeds(max_articles=10)
        assert result["rejected"] == 2
        assert result["new_articles"] == 0

    def test_poll_feeds_respects_poll_interval(self, ingestor, monkeypatch):
        ingestor.subscribe_feed("https://example.com/rss")
        feed = ingestor._feeds[0]
        feed.last_fetched = 10 ** 15
        calls = []
        monkeypatch.setattr(
            KnowledgeIngestor, "_fetch_feed",
            lambda self, f: calls.append(f.url) or [],
        )
        result = ingestor.poll_feeds()
        assert calls == []
        assert result["new_articles"] == 0

    def test_poll_feeds_skips_visited(self, ingestor, monkeypatch):
        ingestor.subscribe_feed("https://example.com/rss")
        url = "https://example.com/1"
        ingestor._mark_visited(url)
        fake = types.SimpleNamespace(
            parse=lambda u: types.SimpleNamespace(entries=[
                {"link": url, "title": "First", "summary": "Summary text"},
            ])
        )
        monkeypatch.setitem(sys.modules, "feedparser", fake)
        monkeypatch.setattr(knowledge, "_scrape_article", lambda url, timeout=15: ARTICLE_TEXT)
        result = ingestor.poll_feeds(max_articles=10)
        assert result["new_articles"] == 0

    def test_search_and_ingest(self, ingestor, monkeypatch):
        monkeypatch.setattr(
            knowledge,
            "_search_ddg",
            lambda query, max_results=5: [
                {"title": "Climate report", "url": "https://example.com/a", "snippet": "Findings on warming."}
            ],
        )
        monkeypatch.setattr(knowledge, "_scrape_article", lambda url, timeout=15: ARTICLE_TEXT)
        result = ingestor.search_and_ingest("climate research")
        assert result["new_facts"] >= 1
        assert result["rejected"] == 0
        assert ingestor._is_visited("https://example.com/a")

    def test_search_and_ingest_rejects(self, ingestor, monkeypatch):
        ingestor.filter = _FakeFilter(allow=False)
        monkeypatch.setattr(
            knowledge,
            "_search_ddg",
            lambda query, max_results=5: [
                {"title": "Climate report", "url": "https://example.com/a", "snippet": "Findings on warming."}
            ],
        )
        monkeypatch.setattr(knowledge, "_scrape_article", lambda url, timeout=15: ARTICLE_TEXT)
        result = ingestor.search_and_ingest("climate research")
        assert result["new_facts"] == 0
        assert result["rejected"] == 1

    def test_search_and_ingest_skips_visited(self, ingestor, monkeypatch):
        ingestor._mark_visited("https://example.com/a")
        monkeypatch.setattr(
            knowledge,
            "_search_ddg",
            lambda query, max_results=5: [
                {"title": "Climate report", "url": "https://example.com/a", "snippet": "Findings on warming."}
            ],
        )
        result = ingestor.search_and_ingest("climate research")
        assert result["new_facts"] == 0

    def test_ingest_url_ok(self, ingestor, monkeypatch):
        monkeypatch.setattr(knowledge, "_scrape_article", lambda url, timeout=15: ARTICLE_TEXT)
        result = ingestor.ingest_url("https://example.com/page")
        assert result["status"] == "ok"
        assert result["new_facts"] >= 1
        assert ingestor._is_visited("https://example.com/page")

    def test_ingest_url_no_content(self, ingestor, monkeypatch):
        monkeypatch.setattr(knowledge, "_scrape_article", lambda url, timeout=15: "")
        result = ingestor.ingest_url("https://example.com/empty")
        assert result["status"] == "no_content"
        assert result["new_facts"] == 0

    def test_ingest_url_short_content(self, ingestor, monkeypatch):
        monkeypatch.setattr(knowledge, "_scrape_article", lambda url, timeout=15: "tiny")
        result = ingestor.ingest_url("https://example.com/tiny")
        assert result["status"] == "no_content"

    def test_ingest_url_rejected(self, ingestor, monkeypatch):
        ingestor.filter = _FakeFilter(allow=False)
        monkeypatch.setattr(knowledge, "_scrape_article", lambda url, timeout=15: ARTICLE_TEXT)
        result = ingestor.ingest_url("https://example.com/blocked")
        assert result["status"] == "rejected"
        assert result["rejected"] is True
        assert ingestor._is_visited("https://example.com/blocked")

    def test_ingest_url_already_visited_rechecks(self, ingestor, monkeypatch):
        ingestor._mark_visited("https://example.com/page")
        monkeypatch.setattr(knowledge, "_scrape_article", lambda url, timeout=15: ARTICLE_TEXT)
        result = ingestor.ingest_url("https://example.com/page")
        assert result["status"] == "ok"

    def test_visited_tracking(self, ingestor):
        url = "https://example.com/track"
        assert not ingestor._is_visited(url)
        ingestor._mark_visited(url)
        assert ingestor._is_visited(url)

    def test_start_stop_background_polling(self, ingestor):
        assert not ingestor._running
        ingestor.start_background_polling(interval=1)
        assert ingestor._running
        ingestor.start_background_polling(interval=1)
        assert ingestor._running
        ingestor.stop_background_polling()
        assert not ingestor._running


class TestKnowledgeIngestorHelpers:
    def test_get_knowledge_ingestor_singleton(self, monkeypatch):
        saved_ing = knowledge._knowledge_ingestor
        saved_mem = knowledge._knowledge_memory
        knowledge._knowledge_ingestor = None
        knowledge._knowledge_memory = None
        try:
            a = knowledge.get_knowledge_ingestor()
            b = knowledge.get_knowledge_ingestor()
            assert a is b
            assert isinstance(a, KnowledgeIngestor)
            assert a.memory is knowledge.get_knowledge_memory()
        finally:
            knowledge._knowledge_ingestor = saved_ing
            knowledge._knowledge_memory = saved_mem


class TestKnowledgeMemoryAddArticle:
    @pytest.fixture
    def km(self, tmp_path, monkeypatch):
        monkeypatch.setattr(knowledge, "VISITED_PATH", tmp_path / "visited.json")
        monkeypatch.setattr(knowledge, "ENTRIES_PATH", tmp_path / "entries.json")
        mem = KnowledgeMemory(load_persisted=False)
        mem.clear_all()
        return mem

    def test_add_article_counts_new_facts(self, km):
        added = km.add_article("https://example.com/a", "Climate report", ARTICLE_TEXT, source="rss")
        assert added >= 1

    def test_add_article_respects_chunk_filter(self, km):
        added = km.add_article(
            "https://example.com/a", "Climate report", ARTICLE_TEXT,
            chunk_filter=lambda text, topic: False,
        )
        assert added == 0

    def test_add_article_dup_url_returns_zero(self, km):
        km.add_article("https://example.com/a", "Climate report", ARTICLE_TEXT, source="rss")
        added = km.add_article("https://example.com/a", "Climate report", ARTICLE_TEXT, source="rss")
        assert added == 0

    def test_add_fact_missing_content_hash_dup(self, km):
        assert km.add_fact(KnowledgeFact(content=ARTICLE_TEXT, topic="climate"))
        assert not km.add_fact(KnowledgeFact(content=ARTICLE_TEXT, topic="climate"))


class _FakeResp:
    def __init__(self, text="", raise_error=False):
        self.text = text
        self._raise = raise_error

    def raise_for_status(self):
        if self._raise:
            raise RuntimeError("http error")


class _FakeHttp:
    def __init__(self, resp):
        self.resp = resp

    def get(self, *args, **kwargs):
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


class _FakeTag:
    def __init__(self, text):
        self._text = text

    def get_text(self, strip=False):
        return self._text.strip() if strip else self._text

    def decompose(self):
        pass


class _FakeSoup:
    def __init__(self, paragraphs):
        self._paragraphs = paragraphs

    def __call__(self, selectors):
        return [_FakeTag("") for _ in selectors]

    def find_all(self, name):
        return self._paragraphs


class _FakeBs4:
    def __init__(self, paragraphs):
        self._paragraphs = paragraphs

    def BeautifulSoup(self, text, parser):
        return _FakeSoup(self._paragraphs)


class _FakeTrafilatura:
    def __init__(self, result):
        self._result = result

    def extract(self, *args, **kwargs):
        return self._result


LONG_TEXT = "This is a long enough article text that easily exceeds the fifty character threshold for extraction."
SHORT_TEXT = "Too short."

DDG_HTML = """
<div class="results">
  <div class="result">
    <a class="result__a" href="https://example.com/a">Title <b>One</b></a>
    <div class="result__snippet">Snippet <b>alpha</b> text.</div>
  </div>
  <div class="result">
    <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fb&amp;rut=abc">Title Two</a>
    <a class="result__snippet">Nested snippet here.</a>
  </div>
  <div class="result">
    <a class="result__a" href="https://example.com/c">Title Three</a>
    <div class="result__snippet">Third snippet.</div>
  </div>
</div>
"""


class TestScrapeArticle:
    @pytest.fixture
    def fake_httpx(self, monkeypatch, resp=None):
        fake = _FakeHttp(resp if resp is not None else _FakeResp(LONG_TEXT))
        monkeypatch.setitem(sys.modules, "httpx", fake)
        return fake

    def test_uses_trafilatura_when_long(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttp(_FakeResp("<html>raw</html>")))
        monkeypatch.setitem(
            sys.modules, "trafilatura", _FakeTrafilatura(LONG_TEXT)
        )
        assert knowledge._scrape_article("https://example.com/a") == LONG_TEXT

    def test_trafilatura_falls_back_to_bs4(self, monkeypatch):
        paragraphs = [
            _FakeTag("A short paragraph."),
            _FakeTag("This is a reasonably long paragraph extracted from the page body."),
        ]
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttp(_FakeResp("<html>raw</html>")))
        monkeypatch.setitem(sys.modules, "trafilatura", _FakeTrafilatura(None))
        monkeypatch.setitem(sys.modules, "bs4", _FakeBs4(paragraphs))
        result = knowledge._scrape_article("https://example.com/a")
        assert "reasonably long paragraph" in result
        assert "A short paragraph." not in result

    def test_returns_fallback_text_when_no_paragraphs(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttp(_FakeResp("<html>raw</html>")))
        monkeypatch.setitem(sys.modules, "trafilatura", _FakeTrafilatura(None))
        monkeypatch.setitem(sys.modules, "bs4", _FakeBs4([]))
        result = knowledge._scrape_article("https://example.com/a")
        assert "<html>raw</html>" in result

    def test_returns_empty_on_http_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttp(_FakeResp(raise_error=True)))
        assert knowledge._scrape_article("https://example.com/a") == ""

    def test_returns_empty_on_exception(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttp(ValueError("boom")))
        assert knowledge._scrape_article("https://example.com/a") == ""


class TestSearchDdg:
    @pytest.fixture
    def fake_httpx(self, monkeypatch):
        fake = _FakeHttp(_FakeResp(DDG_HTML))
        monkeypatch.setitem(sys.modules, "httpx", fake)
        return fake

    def test_parses_results(self, fake_httpx):
        results = knowledge._search_ddg("climate research")
        assert len(results) == 3
        assert results[0]["title"] == "Title One"
        assert results[0]["snippet"] == "Snippet alpha text."

    def test_decodes_uddg_url(self, fake_httpx):
        results = knowledge._search_ddg("climate research")
        assert any(r["url"] == "https://example.com/b" for r in results)

    def test_excludes_duckduckgo_links(self, fake_httpx):
        for r in knowledge._search_ddg("climate research"):
            assert "duckduckgo.com" not in r["url"]

    def test_respects_max_results(self, fake_httpx):
        results = knowledge._search_ddg("climate research", max_results=1)
        assert len(results) == 1

    def test_returns_empty_on_exception(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttp(ValueError("boom")))
        assert knowledge._search_ddg("climate research") == []
