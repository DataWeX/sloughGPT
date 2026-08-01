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
    inst.shutdown()


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
