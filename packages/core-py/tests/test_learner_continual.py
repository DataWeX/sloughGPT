"""Tests for domains/learner/continual.py — ContinualLearner."""

import time
from pathlib import Path

import numpy as np
import pytest

from domains.learner import continual
from domains.learner.continual import (
    CHAR_SET,
    STOI,
    ITOS,
    UNK,
    VOCAB,
    TRAIN_SEQ_LEN,
    BUFFER_CAPACITY,
    ContinualLearner,
    _tokenize,
    _detokenize,
    _build_transformer,
)
from domains.training.slonet import SloTransformer
from domains.learner.knowledge import KnowledgeFact


@pytest.fixture
def state_paths(tmp_path, monkeypatch):
    state_path = tmp_path / "learner" / "continual.soul"
    monkeypatch.setattr(continual, "STATE_PATH", state_path)
    monkeypatch.setattr(continual, "LEARNER_STATE_DIR", state_path.parent)
    return state_path


@pytest.fixture
def learner(state_paths, tmp_path, monkeypatch):
    inst = ContinualLearner(n_embed=32, n_layer=1, n_head=1, lr=1e-4)
    yield inst
    inst._running = False
    inst.ingestor._running = False
    if inst.ingestor._feed_thread and inst.ingestor._feed_thread.is_alive():
        inst.ingestor._feed_thread.join(timeout=0.2)
    inst._save_checkpoint()


# ---- tokenizer ---------------------------------------------------------


def test_tokenize_lowercases_and_filters():
    ids = _tokenize("Hello, World!")
    expected = [STOI[c] for c in "hello, world!"]
    assert ids == expected
    for i in ids:
        assert 0 <= i < VOCAB


def test_tokenize_drops_unknown_chars():
    assert _tokenize("a#b$c") == [STOI["a"], STOI["b"], STOI["c"]]


def test_tokenize_empty():
    assert _tokenize("") == []
    assert _tokenize("###") == []


def test_detokenize_round_trip():
    text = "hello world, how are you?"
    ids = _tokenize(text)
    assert _detokenize(ids) == text


def test_detokenize_unknown_id():
    assert _detokenize([999]) == "?"
    assert _detokenize([1, 999, 2]) == "a?b"


def test_char_set_maps():
    assert VOCAB == len(CHAR_SET)
    assert len(STOI) == len(CHAR_SET)
    assert len(ITOS) == len(STOI)
    for c, i in STOI.items():
        assert ITOS[i] == c
    assert STOI[" "] == UNK


# ---- builder ------------------------------------------------------------


def test_build_transformer_config():
    net = _build_transformer(n_embed=16, n_layer=1, n_head=1, soul_name="t")
    assert net.vocab_size == VOCAB
    assert isinstance(net, type(_build_transformer(n_embed=16)))


# ---- ingestion ----------------------------------------------------------


def test_ingest_text_counts_tokens(learner):
    learner.ingest_text("hello world")
    assert learner.total_tokens_ingested == 11
    assert learner._new_since_last_train == 11
    assert len(learner.buffer) == 11


def test_ingest_text_empty_is_noop(learner):
    learner.ingest_text("")
    learner.ingest_text("###")
    assert learner.total_tokens_ingested == 0
    assert learner.buffer == []


def test_ingest_text_trims_buffer_to_capacity(learner):
    text = ("hello " * (BUFFER_CAPACITY + 1000))
    learner.ingest_text(text)
    assert len(learner.buffer) <= BUFFER_CAPACITY
    assert learner.buffer == learner.buffer[-BUFFER_CAPACITY:]


def test_ingest_conversation_formats_pairs(learner):
    learner.ingest_conversation([("hi", "hello"), ("how are you", "i am fine")])
    expected = _tokenize("user: hi assistant: hello")
    expected2 = _tokenize("user: how are you assistant: i am fine")
    assert learner.buffer == expected + expected2


def test_status_keys(learner):
    learner.ingest_text("some training text")
    status = learner.status()
    assert status["soul_name"] == "continual"
    assert status["total_tokens_ingested"] == len(_tokenize("some training text"))
    assert status["train_steps_completed"] == 0
    assert status["current_loss"] == 0.0
    assert status["buffer_size"] == len(_tokenize("some training text"))
    assert status["buffer_capacity"] == BUFFER_CAPACITY
    assert status["arch"] == "transformer"
    assert status["vocab_size"] == VOCAB
    assert status["n_embed"] == 32
    assert "knowledge" in status
    assert "feeds_subscribed" in status


# ---- evaluate -----------------------------------------------------------


def test_evaluate_text_too_short(learner):
    result = learner.evaluate(text="hi")
    assert result["error"] == "text too short"
    assert result["eval_tokens"] == 0


def test_evaluate_buffer_too_small(learner):
    learner.ingest_text("hi")
    result = learner.evaluate()
    assert result["error"] == "buffer too small"


def test_evaluate_on_text_returns_metrics(learner):
    text = "hello world, this is a test " * 40
    result = learner.evaluate(text=text)
    assert "error" not in result
    assert result["eval_tokens"] > 0
    assert result["loss"] > 0
    assert result["perplexity"] > 0


def test_evaluate_on_buffer_returns_metrics(learner):
    learner.ingest_text("hello world, this is a test " * 40)
    result = learner.evaluate()
    assert "error" not in result
    assert result["eval_tokens"] > 0


# ---- training -----------------------------------------------------------


def test_train_now_runs_real_step(learner):
    learner.ingest_text("hello world, continual learning " * 30)
    status = learner.train_now()
    assert status["train_steps_completed"] >= 1
    assert status["current_loss"] > 0
    assert len(status["loss_history"]) >= 1
    assert status["loss_history"][-1]["loss"] == pytest.approx(status["current_loss"], abs=1e-3)


def test_train_now_with_insufficient_buffer(learner):
    learner.ingest_text("tiny")
    status = learner.train_now()
    assert status["train_steps_completed"] == 0


def test_train_saves_checkpoint(state_paths, learner):
    learner.ingest_text("hello world, continual learning " * 30)
    learner.train_now()
    assert state_paths.exists()


# ---- deploy / shutdown --------------------------------------------------


def test_deploy_writes_soul_file(learner, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    result = learner.deploy(name="test_checkpoint")
    assert result["soul_name"] == "continual"
    assert result["arch"] == "transformer"
    assert result["steps"] == learner.train_steps_completed
    assert Path(result["path"]).exists()
    assert result["file_size"] > 0
    assert "test_checkpoint" in result["path"]


def test_shutdown_stops_thread_and_saves(learner, state_paths):
    learner.ingest_text("hello world, continual learning " * 30)
    learner.shutdown()
    assert learner._running is False
    assert not learner._thread.is_alive()
    assert state_paths.exists()


# ---- knowledge integration -----------------------------------------------


def test_search_knowledge_and_query_topic(learner):
    learner.knowledge.add_fact(KnowledgeFact(
        content="Paris is the capital of France", topic="geography",
        source="test", timestamp=time.time(), importance=0.9,
    ))
    learner.knowledge.add_fact(KnowledgeFact(
        content="Python is a programming language", topic="code",
        source="test", timestamp=time.time(), importance=0.8,
    ))
    results = learner.search_knowledge("paris capital")
    assert any("France" in r["content"] for r in results)
    topic_facts = learner.query_knowledge("code")
    assert any("Python" in f["content"] for f in topic_facts)


def test_get_learner_singleton(learner, monkeypatch):
    from domains.learner.continual import get_learner
    monkeypatch.setattr(continual, "_learner", None)
    monkeypatch.setattr(continual, "ContinualLearner", lambda **kw: learner)
    assert get_learner() is learner
    assert get_learner() is learner


# ---- checkpoint loading --------------------------------------------------


def test_load_existing_checkpoint(state_paths):
    from domains.training.slonet import export_to_sou
    net = _build_transformer(n_embed=32, n_layer=1, n_head=1, soul_name="t")
    export_to_sou(net, str(state_paths))
    inst = ContinualLearner(n_embed=32, n_layer=1, n_head=1)
    try:
        assert isinstance(inst.net, SloTransformer)
        assert inst.train_steps_completed == 0
    finally:
        inst.shutdown()


def test_load_checkpoint_non_transformer_recreates(state_paths, monkeypatch):
    state_paths.parent.mkdir(parents=True, exist_ok=True)
    state_paths.write_text("not a soul file")
    monkeypatch.setattr(continual, "import_from_sou", lambda p: object())
    inst = ContinualLearner(n_embed=32, n_layer=1, n_head=1)
    try:
        assert isinstance(inst.net, SloTransformer)
    finally:
        inst.shutdown()


def test_load_checkpoint_raises_recreates(state_paths, monkeypatch):
    state_paths.parent.mkdir(parents=True, exist_ok=True)
    state_paths.write_text("not a soul file")
    def boom(path):
        raise RuntimeError("corrupt checkpoint")
    monkeypatch.setattr(continual, "import_from_sou", boom)
    inst = ContinualLearner(n_embed=32, n_layer=1, n_head=1)
    try:
        assert isinstance(inst.net, SloTransformer)
    finally:
        inst.shutdown()


# ---- web search + live learning -------------------------------------------


def test_search_and_learn_ingests_articles(learner, monkeypatch):
    monkeypatch.setattr(
        learner.ingestor, "search_and_ingest",
        lambda query, max_results: {"new_facts": 3, "rejected": 1, "stats": {"accepted": 3}},
    )
    monkeypatch.setattr(
        learner.knowledge, "search",
        lambda query, top_k: [{"content": "Alpha article content"}, {"content": "Beta article content"}],
    )
    result = learner.search_and_learn("latest ai news")
    assert result["new_facts"] == 3
    assert result["rejected"] == 1
    assert result["tokens_ingested"] > 0
    assert result["filter_stats"] == {"accepted": 3}
    assert learner.total_tokens_ingested > 0
    joined = _detokenize(learner.buffer)
    assert "web" in joined


def test_search_and_learn_no_facts_is_noop(learner, monkeypatch):
    monkeypatch.setattr(
        learner.ingestor, "search_and_ingest",
        lambda query, max_results: {"new_facts": 0, "rejected": 0, "stats": {}},
    )
    result = learner.search_and_learn("nothing found")
    assert result["new_facts"] == 0
    assert result["tokens_ingested"] == 0


def test_subscribe_feed_ingests_initial_articles(learner, monkeypatch):
    monkeypatch.setattr(learner.ingestor, "subscribe_feed", lambda url, poll_interval: True)
    monkeypatch.setattr(learner.ingestor, "poll_feeds", lambda max_articles: {"new_articles": 2})
    monkeypatch.setattr(learner.knowledge, "all_topics", lambda: ["rss-topic"])
    monkeypatch.setattr(
        learner.knowledge, "get_topic_facts",
        lambda topic: [{"content": "Feed article body text"}],
    )
    assert learner.subscribe_feed("http://example.com/rss") is True
    assert learner.total_tokens_ingested > 0
    assert "rss" in _detokenize(learner.buffer)


def test_subscribe_feed_rejected(learner, monkeypatch):
    monkeypatch.setattr(learner.ingestor, "subscribe_feed", lambda url, poll_interval: False)
    result = learner.subscribe_feed("http://example.com/rss")
    assert result is False
    assert learner.total_tokens_ingested == 0


def test_unsubscribe_feed(learner, monkeypatch):
    monkeypatch.setattr(learner.ingestor, "unsubscribe_feed", lambda url: True)
    assert learner.unsubscribe_feed("http://example.com/rss") is True


def test_list_feeds(learner, monkeypatch):
    feeds = [{"url": "http://example.com/rss", "poll_interval": 3600}]
    monkeypatch.setattr(learner.ingestor, "list_feeds", lambda: feeds)
    assert learner.list_feeds() == feeds


def test_ingest_url_ingests_matching_fact(learner, monkeypatch):
    url = "http://example.com/article-1"
    monkeypatch.setattr(
        learner.ingestor, "ingest_url",
        lambda u: {"new_facts": 1, "title": "Title", "content_length": 120, "rejected": False, "status": "ok"},
    )
    monkeypatch.setattr(learner.knowledge, "all_topics", lambda: ["web-topic"])
    monkeypatch.setattr(
        learner.knowledge, "get_topic_facts",
        lambda topic: [{"url": url, "content": "Scraped article body"}],
    )
    result = learner.ingest_url(url)
    assert result["new_facts"] == 1
    assert learner.total_tokens_ingested > 0
    assert "article" in _detokenize(learner.buffer)


def test_ingest_url_no_new_facts(learner, monkeypatch):
    monkeypatch.setattr(
        learner.ingestor, "ingest_url",
        lambda u: {"new_facts": 0, "title": "", "content_length": 0, "rejected": False, "status": "no_content"},
    )
    result = learner.ingest_url("http://example.com/empty")
    assert result["new_facts"] == 0
    assert learner.total_tokens_ingested == 0


# ---- training edge cases ----------------------------------------------------


def test_train_step_skips_when_loss_none(learner):
    learner.ingest_text("hello world, continual learning " * 30)
    orig = learner.net.forward
    calls = {"n": 0}

    def fake_forward(x, targets=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return None, None
        return orig(x, targets=targets)

    learner.net.forward = fake_forward
    status = learner.train_now()
    assert calls["n"] >= 2
    assert status["train_steps_completed"] >= 1


def test_train_step_trims_loss_history(learner):
    learner.ingest_text("hello world, continual learning " * 30)
    learner.loss_history = [{"step": i, "loss": 0.1, "tokens": 1, "timestamp": 0.0} for i in range(500)]
    status = learner.train_now()
    assert len(learner.loss_history) == 500
    assert status["loss_history"][-1]["step"] == 1


def test_evaluate_no_chunks_from_buffer(learner):
    learner.ingest_text("hello world, this is a test " * 2)
    result = learner.evaluate()
    assert result["error"] == "no eval chunks"
    assert result["eval_tokens"] == 0


def test_shutdown_joins_live_thread(state_paths, monkeypatch):
    monkeypatch.setattr(continual.time, "sleep", lambda _: None)
    inst = ContinualLearner(n_embed=32, n_layer=1, n_head=1, lr=1e-4)
    monkeypatch.setattr(inst.ingestor, "stop_background_polling", lambda: None)
    try:
        assert inst._thread.is_alive()
        inst.shutdown()
    finally:
        inst.shutdown()
    assert not inst._thread.is_alive()


# ---- background loop ----------------------------------------------------------


def _make_no_thread_learner(monkeypatch, state_paths):
    import threading
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    return ContinualLearner(n_embed=32, n_layer=1, n_head=1, lr=1e-4)


def test_background_loop_handles_event_loop_failure(monkeypatch, state_paths):
    import asyncio

    def boom():
        raise RuntimeError("no event loop")

    inst = _make_no_thread_learner(monkeypatch, state_paths)
    monkeypatch.setattr(asyncio, "new_event_loop", boom)
    try:
        inst._running = False
        inst._background_loop()
    finally:
        inst.shutdown()


def test_background_loop_polls_feed_articles(monkeypatch, state_paths):
    inst = _make_no_thread_learner(monkeypatch, state_paths)
    monkeypatch.setattr(inst.knowledge, "all_topics", lambda: ["rss-topic"])
    monkeypatch.setattr(
        inst.knowledge, "get_topic_facts",
        lambda topic: [{"source": "rss", "content": "Feed snippet for training"}],
    )
    calls = {"n": 0}

    def fake_sleep(secs):
        calls["n"] += 1
        if calls["n"] >= 7:
            inst._running = False

    monkeypatch.setattr(continual.time, "sleep", fake_sleep)
    try:
        inst._background_loop()
        assert calls["n"] >= 7
        assert "rss" in _detokenize(inst.buffer)
    finally:
        inst.shutdown()


def test_background_loop_trains_and_catches_errors(monkeypatch, state_paths):
    inst = _make_no_thread_learner(monkeypatch, state_paths)
    inst.ingest_text("hello world, continual learning " * 40)
    inst._new_since_last_train = 9999
    inst._last_train_time = 0.0

    def boom():
        raise RuntimeError("train failed")

    monkeypatch.setattr(inst, "_train_step", boom)
    calls = {"n": 0}

    def fake_sleep(secs):
        calls["n"] += 1
        if calls["n"] >= 3:
            inst._running = False

    monkeypatch.setattr(continual.time, "sleep", fake_sleep)
    try:
        inst._background_loop()
        assert calls["n"] >= 3
    finally:
        inst.shutdown()


def test_safe_get_facts_returns_empty_on_error(learner, monkeypatch):
    def boom(topic):
        raise RuntimeError("memory down")

    monkeypatch.setattr(learner.knowledge, "get_topic_facts", boom)
    assert learner._safe_get_facts("any-topic") == []
