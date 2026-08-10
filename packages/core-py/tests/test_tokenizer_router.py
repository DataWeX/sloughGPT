"""Tests for the tokenizer API router (routers/tokenizer.py).

Covers: TokenizerRouter stats, tokenize, detokenize, vocab, merges, sample, train,
pretokenize, decompose, analyze.
All domain calls are mocked; only HTTP-level behavior is tested.

Note: the tokenizer router imports get_tokenizer_manager at MODULE level
(line 14: from domains.training.tokenizer_manager import get_tokenizer_manager),
so patching 'routers.tokenizer.get_tokenizer_manager' works directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.tokenizer import router  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokenizer(**overrides):
    defaults = dict(
        vocab_size=5,
        SPECIAL_TOKENS={"<unk>", "<pad>", "<bos>", "<eos>"},
        itos={0: "<unk>", 1: "<pad>", 2: "a", 3: "b", 4: "c"},
        vocab={0: "<unk>", 1: "<pad>", 2: "a", 3: "b", 4: "c"},
        merges=[("a", "b"), ("b", "c")],
        encode=lambda text: [2, 3] if text == "ab" else [2],
        decode=lambda ids: "ab" if ids == [2, 3] else "a",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_mgr(**overrides):
    defaults = dict(
        _trained=True,
        is_trained=lambda: True,
        borrow_from_autotrain=lambda: None,
        stats=lambda: {
            "vocab_size": 5,
            "base_chars": 50,
            "merged_subwords": 30,
            "special_tokens": 4,
            "total_merges": 30,
            "trained": True,
        },
        tokenize=lambda text: [2, 3],
        detokenize=lambda ids: "ab",
        get_tokenizer=lambda: _make_tokenizer(),
        show_pretokenization=lambda text: {"pretokens": text.split()},
        decompose_token=lambda text: {"token": text, "tree": "a+b"},
        analyze_corpus=lambda texts: {"total_tokens": 100, "unique_tokens": 50, "compression_ratio": 2.0},
        train=lambda texts, vocab_size=512, min_frequency=3, lowercase=False: None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _app():
    app = FastAPI()
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Tests — patch at module level (imported directly at module scope)
# ---------------------------------------------------------------------------

MOCK_TARGET = "routers.tokenizer.get_tokenizer_manager"


class TestTokenizerStats:
    @patch(MOCK_TARGET)
    def test_returns_stats(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.get("/tokenizer/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["vocab_size"] == 5
        assert data["special_tokens"] == 4
        assert data["total_merges"] == 30

    @patch(MOCK_TARGET)
    def test_fallback_key_names(self, mock_get):
        mgr = _make_mgr(
            stats=lambda: {
                "vocab_size": 128,
                "base_chars": 30,
                "subwords": 20,
                "special_tokens": 3,
                "total_merges_learned": 20,
                "trained": True,
            }
        )
        mock_get.return_value = mgr
        client = TestClient(_app())
        resp = client.get("/tokenizer/stats")
        data = resp.json()["data"]
        assert data["merged_subwords"] == 20
        assert data["total_merges"] == 20


class TestTokenize:
    @patch(MOCK_TARGET)
    def test_returns_tokens_and_ids(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.post("/tokenizer/tokenize", json={"text": "ab"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["ids"] == [2, 3]
        assert len(data["tokens"]) == 2


class TestDetokenize:
    @patch(MOCK_TARGET)
    def test_returns_text(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.post("/tokenizer/detokenize", json={"ids": [2, 3]})
        assert resp.status_code == 200
        assert resp.json()["data"]["text"] == "ab"


class TestGetVocab:
    @patch(MOCK_TARGET)
    def test_returns_entries(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.get("/tokenizer/vocab")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 5
        assert len(data["entries"]) > 0

    @patch(MOCK_TARGET)
    def test_pagination(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.get("/tokenizer/vocab", params={"limit": 2, "offset": 2})
        data = resp.json()["data"]
        assert len(data["entries"]) == 2
        assert data["entries"][0]["id"] == 2


class TestGetMerges:
    @patch(MOCK_TARGET)
    def test_returns_merges(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.get("/tokenizer/merges")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 2
        assert data["merges"][0]["left"] == "a"


class TestGetSample:
    @patch(MOCK_TARGET)
    def test_returns_sample_words(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.get("/tokenizer/sample")
        assert resp.status_code == 200
        samples = resp.json()["data"]["samples"]
        assert len(samples) > 0
        assert "word" in samples[0]
        assert "ids" in samples[0]


class TestPretokenize:
    @patch(MOCK_TARGET)
    def test_returns_pretokens(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.post("/tokenizer/pretokenize", json={"text": "hello world"})
        assert resp.status_code == 200
        assert "pretokens" in resp.json()["data"]


class TestDecompose:
    @patch(MOCK_TARGET)
    def test_returns_decomposition(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.post("/tokenizer/decompose", json={"text": "ab"})
        assert resp.status_code == 200
        assert "tree" in resp.json()["data"]


class TestAnalyzeCorpus:
    @patch(MOCK_TARGET)
    def test_returns_corpus_stats(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.post("/tokenizer/analyze", json={"texts": ["hello", "world"]})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "total_tokens" in data
        assert "compression_ratio" in data


class TestTrainTokenizer:
    @patch(MOCK_TARGET)
    def test_train_with_texts(self, mock_get):
        mock_get.return_value = _make_mgr()
        client = TestClient(_app())
        resp = client.post("/tokenizer/train", json={"vocab_size": 128, "texts": ["hello world"]})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "trained"
        assert data["corpus_size"] == 1
