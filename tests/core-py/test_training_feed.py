"""Tests for the training feed — local + network feed client, text loading, and FeedBatchSampler.

Covers the feed contract: subscribe to a live corpus (API logs) instead of a
static dataset, point a URL at the training flow, and train on whatever the
feed returns (with a persisted offset cursor).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from domain.training._internal.training_feed import (
    FeedBatchSampler,
    TrainingFeedClient,
    extract_pair,
    is_feed_source,
    load_feed_text,
    pairs_to_text,
)


def _corpus_record(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {"model": "test", "captured_at": "2026-09-22T00:00:00+00:00"},
    }


def _write_corpus(path: Path, records) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def data_root(tmp_path) -> Path:
    """Temporary data root with a live conversation corpus.

    Mirrors the production layout: ``<root>/api_conversations/corpus.jsonl``.
    """
    d = tmp_path / "api_conversations"
    d.mkdir(parents=True, exist_ok=True)
    pairs = [
        _corpus_record(
            "Tell me about the neural network architecture used for training.",
            "It uses layered transformer blocks with self-attention and residual "
            "connections to model long-range dependencies in the sequence.",
        ),
        _corpus_record(
            "Explain the difference between supervised and unsupervised learning.",
            "Supervised learning uses labeled examples while unsupervised learning "
            "discovers structure in unlabeled data on its own.",
        ),
    ]
    _write_corpus(d / "corpus.jsonl", pairs)
    return tmp_path


@pytest.fixture()
def offset_store(tmp_path) -> Path:
    return tmp_path / "feed-cursor.json"


# ── is_feed_source ────────────────────────────────────────────────────────────


class TestIsFeedSource:
    """Recognise feed specifications in data_path positions."""

    def test_feed_scheme(self):
        assert is_feed_source("feed:api-conversations") is True

    def test_http_url(self):
        assert is_feed_source("http://localhost:8000/training/feed") is True

    def test_https_url(self):
        assert is_feed_source("https://feeds.example.com/corpus") is True

    def test_plain_path_is_not_feed(self):
        assert is_feed_source("data/foo/input.txt") is False

    def test_none_is_not_feed(self):
        assert is_feed_source(None) is False

    def test_list_is_not_feed(self):
        assert is_feed_source(["data/foo"]) is False


# ── extract_pair ──────────────────────────────────────────────────────────────


class TestExtractPair:
    """Extract user/assistant pairs from corpus record formats."""

    def test_messages_format(self):
        rec = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        assert extract_pair(rec) == {"user_msg": "hi", "assistant_msg": "hello"}

    def test_messages_three_turns_returns_first_pair(self):
        rec = {
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ]
        }
        assert extract_pair(rec) == {"user_msg": "q1", "assistant_msg": "a1"}

    def test_flat_user_msg_assistant_msg(self):
        assert extract_pair({"user_msg": "user query", "assistant_msg": "assistant reply"}) == {
            "user_msg": "user query",
            "assistant_msg": "assistant reply",
        }

    def test_flat_prompt_completion(self):
        assert extract_pair({"prompt": "user query", "completion": "assistant reply"}) == {
            "user_msg": "user query",
            "assistant_msg": "assistant reply",
        }

    def test_empty_record_returns_none(self):
        assert extract_pair({}) is None

    def test_missing_assistant_returns_none(self):
        assert extract_pair({"messages": [{"role": "user", "content": "hi"}]}) is None

    def test_empty_content_returns_none(self):
        assert (
            extract_pair(
                {
                    "messages": [
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            )
            is None
        )

    def test_short_content_returns_none(self):
        assert (
            extract_pair(
                {
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                    ]
                }
            )
            is None
        )


# ── pairs_to_text ─────────────────────────────────────────────────────────────


class TestPairsToText:
    """Build structured training text from pairs."""

    def test_contains_user_and_assistant_content(self):
        text = pairs_to_text(
            [{"user_msg": "hello", "assistant_msg": "hi there"}], use_prompt_engine=False
        )
        assert "hello" in text
        assert "hi there" in text

    def test_has_template_delimiters(self):
        text = pairs_to_text([{"user_msg": "x", "assistant_msg": "y"}], use_prompt_engine=False)
        assert "<|im_start|>user\nx<|im_end|>" in text
        assert "<|im_start|>assistant\ny<|im_end|>" in text


# ── TrainingFeedClient (local) ────────────────────────────────────────────────


class TestTrainingFeedClientLocal:
    """Local read of a live corpus file with offset cursoring."""

    def test_read_returns_all_records(self, data_root):
        client = TrainingFeedClient(source="api-conversations", local_base=data_root)
        page = client.read()
        assert page.total == 2
        assert page.offset == 0
        assert page.next_offset == 2
        assert len(page.records) == 2
        assert page.records[0]["messages"][0]["content"].startswith(
            "Tell me about the neural network"
        )

    def test_read_from_offset(self, data_root):
        client = TrainingFeedClient(source="api-conversations", local_base=data_root)
        page = client.read(offset=1)
        assert page.offset == 1
        assert page.next_offset == 2
        assert len(page.records) == 1

    def test_read_with_limit(self, data_root):
        client = TrainingFeedClient(source="api-conversations", local_base=data_root)
        page = client.read(limit=1)
        assert len(page.records) == 1
        assert page.next_offset == 1

    def test_read_pairs(self, data_root):
        client = TrainingFeedClient(source="api-conversations", local_base=data_root)
        pairs = client.read_pairs()
        assert len(pairs) == 2
        assert pairs[0]["user_msg"].startswith("Tell me about the neural")
        assert "layered transformer blocks" in pairs[0]["assistant_msg"]

    def test_empty_corpus_returns_empty_page(self, tmp_path):
        d = tmp_path / "api_conversations"
        d.mkdir(parents=True, exist_ok=True)
        (d / "corpus.jsonl").write_text("", encoding="utf-8")
        client = TrainingFeedClient(source="api-conversations", local_base=tmp_path)
        page = client.read()
        assert page.total == 0
        assert page.records == []

    def test_malformed_lines_are_skipped_not_crashed(self, data_root):
        (data_root / "api_conversations" / "corpus.jsonl").write_text(
            "not json\n"
            + json.dumps(_corpus_record("middle", "still a valid row here"))
            + "\n"
            + "{broken",
            encoding="utf-8",
        )
        client = TrainingFeedClient(source="api-conversations", local_base=data_root)
        page = client.read()
        assert page.total == 3
        assert len(page.records) == 1

    def test_cursor_persists_across_clients(self, data_root, offset_store):
        client = TrainingFeedClient(
            source="api-conversations", local_base=data_root, offset_store_path=offset_store
        )
        client.read()
        client.save_offset(2)
        again = TrainingFeedClient(
            source="api-conversations", local_base=data_root, offset_store_path=offset_store
        )
        assert again.offset == 2

    def test_source_name_in_page(self, data_root):
        client = TrainingFeedClient(source="api-conversations", local_base=data_root)
        assert client.read().name == "api-conversations"
        assert client.read().source == "api-conversations"

    def test_unknown_source_raises(self, tmp_path):
        client = TrainingFeedClient(source="nope", local_base=tmp_path)
        with pytest.raises(ValueError, match="nope"):
            client.read()


# ── TrainingFeedClient (network) ──────────────────────────────────────────────


class _FakeHttpResponse:
    """Stand-in for urllib.request.urlopen context manager result."""

    def __init__(self, payload: dict) -> None:
        self._bytes = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._bytes


class TestTrainingFeedClientNetwork:
    """Fetch the feed endpoint over HTTP with offset forwarding."""

    def test_read_forwards_offset_and_limit(self, offset_store):
        payload = {
            "data": {
                "name": "api-conversations",
                "source": "api-conversations",
                "offset": 3,
                "next_offset": 4,
                "total": 4,
                "records": [
                    _corpus_record("network q", "network answer with enough text to be useful")
                ],
            }
        }
        client = TrainingFeedClient(
            feed_url="http://feed.example/corpus", offset_store_path=offset_store
        )
        client.save_offset(3)
        with patch("urllib.request.urlopen", return_value=_FakeHttpResponse(payload)) as m:
            page = client.read(limit=1)
        url_arg = m.call_args.args[0]
        assert "offset=3" in url_arg
        assert "limit=1" in url_arg
        assert page.total == 4
        assert page.next_offset == 4
        assert len(page.records) == 1

    def test_channel_errors_raise_normal_error(self, offset_store):
        client = TrainingFeedClient(
            feed_url="http://feed.example/corpus", timeout=0.01, offset_store_path=offset_store
        )
        with pytest.raises(Exception):
            client.read()

    def test_plain_envelope_fallback(self, offset_store):
        payload = {
            "name": "feeds",
            "source": "x",
            "offset": 0,
            "next_offset": 1,
            "total": 1,
            "records": [_corpus_record("u", "a long enough assistant reply text here")],
        }
        client = TrainingFeedClient(
            feed_url="http://feed.example/corpus", offset_store_path=offset_store
        )
        with patch("urllib.request.urlopen", return_value=_FakeHttpResponse(payload)):
            page = client.read()
        assert page.total == 1
        assert len(page.records) == 1


# ── load_feed_text + prepare_data wiring ──────────────────────────────────────


class TestLoadFeedText:
    """Feed path → structured training text (before tokenization)."""

    def test_local_feed_builds_template_text(self, data_root):
        text = load_feed_text(
            "feed:api-conversations", use_prompt_engine=False, local_base=data_root
        )
        assert "Tell me about the neural network" in text
        assert "layered transformer blocks" in text
        assert "<|im_start|>user\n" in text

    def test_http_url_builds_template_text(self, offset_store):
        payload = {
            "data": {
                "name": "api-conversations",
                "source": "api-conversations",
                "offset": 0,
                "next_offset": 1,
                "total": 1,
                "records": [_corpus_record("remote q", "remote a with plenty of text")],
            }
        }
        with patch("urllib.request.urlopen", return_value=_FakeHttpResponse(payload)):
            text = load_feed_text(
                "http://feed.example/corpus",
                use_prompt_engine=False,
                offset_store_path=offset_store,
            )
        assert "remote q" in text
        assert "remote a with plenty of text" in text


class TestPrepareDataFeedWiring:
    """prepare_data accepts a feed path and trains on the feed text."""

    def test_prepare_data_uses_feed_text(self, monkeypatch):
        from domain.training._internal import train_pipeline

        text = "the quick brown fox jumps over the lazy dog and keeps running. " * 8
        monkeypatch.setattr(train_pipeline, "load_feed_text", lambda *a, **k: text)
        data, vocab_size, stoi, itos = train_pipeline.prepare_data("feed:api-conversations")
        assert len(data) == len(text)
        assert "q" in stoi


# ── FeedBatchSampler ──────────────────────────────────────────────────────────


class TestFeedBatchSampler:
    """BatchSampler that drains a feed and feeds TrainingLoop."""

    def test_get_batch_shapes(self, data_root):
        stoi = {"<|im_start|>": 0, "|": 1, "\n": 2, "u": 3, "a": 4, "s": 5, "t": 6}
        sampler = FeedBatchSampler(
            stoi, block_size=8, source="api-conversations", local_base=data_root
        )
        assert len(sampler) > 0
        x, y = sampler.get_batch(2)
        assert x.shape == (2, 8)
        assert y.shape == (2, 8)
        assert x.dtype == np.int32
        assert y.dtype == np.int32
        assert sampler.stats()["pairs"] == 2

    def test_starts_from_persisted_cursor_when_set(self, data_root, offset_store):
        first = TrainingFeedClient(
            source="api-conversations", local_base=data_root, offset_store_path=offset_store
        )
        first.save_offset(1)
        sampler = FeedBatchSampler(
            stoi={"a": 0},
            block_size=4,
            source="api-conversations",
            local_base=data_root,
            offset_store_path=offset_store,
        )
        assert sampler.stats()["pairs"] == 1

    def test_refresh_ingests_new_records(self, data_root):
        stoi = {"<|im_start|>": 0, "\n": 1, "u": 2}
        sampler = FeedBatchSampler(
            stoi,
            block_size=8,
            source="api-conversations",
            local_base=data_root,
            refresh_interval=0,
        )
        before = sampler.stats()["pairs"]
        corpus_file = data_root / "api_conversations" / "corpus.jsonl"
        corpus_file.write_text(
            corpus_file.read_text(encoding="utf-8")
            + json.dumps(
                _corpus_record("Yet another user query", "And yet another helpful assistant reply")
            )
            + "\n",
            encoding="utf-8",
        )
        new = sampler.refresh()
        assert new == 1
        assert sampler.stats()["pairs"] == before + 1
        assert len(sampler) > 0

    def test_synthetic_fallback_when_feed_empty(self, tmp_path, offset_store):
        d = tmp_path / "api_conversations"
        d.mkdir(parents=True, exist_ok=True)
        (d / "corpus.jsonl").write_text(
            json.dumps(_corpus_record("a", "b")) + "\n", encoding="utf-8"
        )
        stoi = {c: i for i, c in enumerate(sorted(set("<|im_start|>user\nassistant\nab")))}
        sampler = FeedBatchSampler(
            stoi,
            block_size=4,
            source="api-conversations",
            local_base=tmp_path,
            offset_store_path=offset_store,
        )
        x, y = sampler.get_batch(1)
        assert x.shape == (1, 4)
        assert y.shape == (1, 4)
