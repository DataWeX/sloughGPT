"""
Tests for the tokenizer router — stats, tokenize, detokenize, vocab, merges, sample, train.

Uses a standalone FastAPI app with only the router under test.
The router imports get_tokenizer_manager at module level, so we patch the
name in the router's namespace.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.tokenizer import router

app = FastAPI()
register_all_handlers(app)
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

MGR_TARGET = "apps.api.server.routers.tokenizer.get_tokenizer_manager"


def _mock_tokenizer():
    """Create a mock tokenizer matching the real Tokenizer interface."""
    tok = MagicMock()
    tok.vocab_size = 256
    tok.vocab = {i: f"tok_{i}" for i in range(256)}
    tok.itos = {i: f"tok_{i}" for i in range(256)}
    tok.SPECIAL_TOKENS = {"<pad>", "<bos>", "<eos>", "<unk>"}
    tok.merges = [("t", "h"), ("th", "e"), ("e", "r"), ("a", "n"), ("an", "d")]
    tok.encode.return_value = [104, 101, 108, 108]
    return tok


def _mock_manager():
    """Create a mock TokenizerManager matching the real interface."""
    mgr = MagicMock()
    mgr.is_trained.return_value = True
    mgr.stats.return_value = {
        "vocab_size": 256,
        "base_chars": 200,
        "merged_subwords": 56,
        "special_tokens": 4,
        "total_merges": 56,
        "trained": True,
    }
    mgr.tokenize.return_value = [104, 101, 108, 108]
    mgr.detokenize.return_value = "hello"
    mgr.get_tokenizer.return_value = _mock_tokenizer()
    mgr.show_pretokenization.return_value = {"pretokens": ["hello", " ", "world"]}
    mgr.decompose_token.return_value = {
        "token": "the",
        "parts": ["th", "e"],
        "depth": 1,
    }
    mgr.analyze_corpus.return_value = {
        "tokens": 100,
        "unique": 50,
        "compression": 2.0,
    }
    return mgr


class TestTokenizerStats:
    """GET /tokenizer/stats"""

    @patch(MGR_TARGET)
    def test_get_stats(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()

        resp = client.get("/tokenizer/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["vocab_size"] == 256
        assert data["special_tokens"] == 4
        assert data["trained"] is True

    @patch(MGR_TARGET)
    def test_stats_not_trained_borrows(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.is_trained.side_effect = [False, True]
        mock_get_mgr.return_value = mgr

        resp = client.get("/tokenizer/stats")
        assert resp.status_code == 200
        mgr.borrow_from_autotrain.assert_called_once()

    @patch(MGR_TARGET)
    def test_stats_not_trained_trains(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.is_trained.side_effect = [False, False, True]
        mock_get_mgr.return_value = mgr

        resp = client.get("/tokenizer/stats")
        assert resp.status_code == 200
        mgr.train.assert_called_once()

    @patch(MGR_TARGET)
    def test_stats_error(self, mock_get_mgr):
        mock_get_mgr.side_effect = Exception("broken")
        resp = client.get("/tokenizer/stats")
        assert resp.status_code == 500

    @patch(MGR_TARGET)
    def test_stats_fallback_merged_subwords(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.stats.return_value = {
            "vocab_size": 256,
            "base_chars": 200,
            "subwords": 44,
            "special_tokens": 4,
            "total_merges_learned": 33,
            "trained": True,
        }
        mock_get_mgr.return_value = mgr
        resp = client.get("/tokenizer/stats")
        data = resp.json()["data"]
        assert data["merged_subwords"] == 44
        assert data["total_merges"] == 33

    @patch(MGR_TARGET)
    def test_stats_missing_keys_have_fallbacks(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.stats.return_value = {"vocab_size": 3, "special_tokens": 2, "trained": True}
        mock_get_mgr.return_value = mgr
        resp = client.get("/tokenizer/stats")
        data = resp.json()["data"]
        assert data["merged_subwords"] == 0
        assert data["total_merges"] == 0


class TestTokenize:
    """POST /tokenizer/tokenize"""

    @patch(MGR_TARGET)
    def test_tokenize(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.tokenize.return_value = [104, 101, 108, 108]
        tok = mgr.get_tokenizer.return_value
        tok.itos = {104: "h", 101: "e", 108: "l"}
        mock_get_mgr.return_value = mgr

        resp = client.post("/tokenizer/tokenize", json={"text": "hello"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "tokens" in data
        assert "ids" in data
        assert data["ids"] == [104, 101, 108, 108]
        assert len(data["tokens"]) == 4

    @patch(MGR_TARGET)
    def test_tokenize_empty_text(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.tokenize.return_value = []
        mock_get_mgr.return_value = mgr

        resp = client.post("/tokenizer/tokenize", json={"text": ""})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["ids"] == []
        assert data["tokens"] == []

    def test_tokenize_missing_text(self):
        resp = client.post("/tokenizer/tokenize", json={})
        assert resp.status_code == 422

    @patch(MGR_TARGET)
    def test_tokenize_unknown_id_uses_placeholder(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.tokenize.return_value = [104, 999]
        tok = mgr.get_tokenizer.return_value
        tok.itos = {104: "h"}
        mock_get_mgr.return_value = mgr
        resp = client.post("/tokenizer/tokenize", json={"text": "hi"})
        data = resp.json()["data"]
        assert data["tokens"] == ["h", "<?>"]

    @patch(MGR_TARGET)
    def test_tokenize_manager_error_returns_500(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.tokenize.side_effect = RuntimeError("tokenize broke")
        mock_get_mgr.return_value = mgr
        resp = client.post("/tokenizer/tokenize", json={"text": "hello"})
        assert resp.status_code == 500

    def test_tokenize_overlong_text_returns_422(self):
        resp = client.post("/tokenizer/tokenize", json={"text": "x" * 50001})
        assert resp.status_code == 422


class TestDetokenize:
    """POST /tokenizer/detokenize"""

    @patch(MGR_TARGET)
    def test_detokenize(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.detokenize.return_value = "hello"
        mock_get_mgr.return_value = mgr

        resp = client.post("/tokenizer/detokenize", json={"ids": [104, 101, 108, 108]})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["text"] == "hello"

    @patch(MGR_TARGET)
    def test_detokenize_empty(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.detokenize.return_value = ""
        mock_get_mgr.return_value = mgr

        resp = client.post("/tokenizer/detokenize", json={"ids": []})
        assert resp.status_code == 200
        assert resp.json()["data"]["text"] == ""

    def test_detokenize_missing_ids(self):
        resp = client.post("/tokenizer/detokenize", json={})
        assert resp.status_code == 422

    @patch(MGR_TARGET)
    def test_detokenize_manager_error_returns_500(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.detokenize.side_effect = RuntimeError("detokenize broke")
        mock_get_mgr.return_value = mgr
        resp = client.post("/tokenizer/detokenize", json={"ids": [104]})
        assert resp.status_code == 500

    def test_detokenize_overlong_ids_returns_422(self):
        resp = client.post("/tokenizer/detokenize", json={"ids": [0] * 10001})
        assert resp.status_code == 422


class TestVocab:
    """GET /tokenizer/vocab"""

    @patch(MGR_TARGET)
    def test_get_vocab(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()

        resp = client.get("/tokenizer/vocab", params={"limit": 10, "offset": 0})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "entries" in data
        assert data["total"] == 256
        assert len(data["entries"]) == 10
        assert data["entries"][0]["id"] == 0
        assert data["offset"] == 0
        assert data["limit"] == 10

    @patch(MGR_TARGET)
    def test_vocab_special_tokens_detected(self, mock_get_mgr):
        mgr = _mock_manager()
        tok = mgr.get_tokenizer.return_value
        tok.vocab = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>", 4: "a", 5: "b"}
        tok.vocab_size = 6
        tok.SPECIAL_TOKENS = {"<pad>", "<bos>", "<eos>", "<unk>"}
        mock_get_mgr.return_value = mgr

        resp = client.get("/tokenizer/vocab", params={"limit": 6})
        assert resp.status_code == 200
        entries = resp.json()["data"]["entries"]
        special = [e for e in entries if e["is_special"]]
        assert len(special) == 4

    @patch(MGR_TARGET)
    def test_vocab_offset(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()

        resp = client.get("/tokenizer/vocab", params={"limit": 5, "offset": 100})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["entries"][0]["id"] == 100

    @patch(MGR_TARGET)
    def test_vocab_limit_beyond_total(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()

        resp = client.get("/tokenizer/vocab", params={"limit": 500, "offset": 250})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["entries"]) == 6


class TestMerges:
    """GET /tokenizer/merges"""

    @patch(MGR_TARGET)
    def test_get_merges(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()

        resp = client.get("/tokenizer/merges")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "merges" in data
        assert data["total"] == 5
        first = data["merges"][0]
        assert first["left"] == "t"
        assert first["right"] == "h"
        assert first["token"] == "th"
        assert first["index"] == 0

    @patch(MGR_TARGET)
    def test_merges_with_limit(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()

        resp = client.get("/tokenizer/merges", params={"limit": 2})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["merges"]) == 2

    @patch(MGR_TARGET)
    def test_merges_empty(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.get_tokenizer.return_value.merges = []
        mock_get_mgr.return_value = mgr

        resp = client.get("/tokenizer/merges")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["merges"] == []
        assert data["total"] == 0

    @patch(MGR_TARGET)
    def test_merges_non_tuple_elements(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.get_tokenizer.return_value.merges = ["ab", "cd"]
        mock_get_mgr.return_value = mgr

        resp = client.get("/tokenizer/merges")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["merges"][0]["left"] == "ab"
        assert data["merges"][0]["right"] == ""

    @patch(MGR_TARGET)
    def test_merges_manager_error_returns_500(self, mock_get_mgr):
        mock_get_mgr.side_effect = RuntimeError("merges broke")
        resp = client.get("/tokenizer/merges")
        assert resp.status_code == 500

    @patch(MGR_TARGET)
    def test_vocab_manager_error_returns_500(self, mock_get_mgr):
        mock_get_mgr.side_effect = RuntimeError("vocab broke")
        resp = client.get("/tokenizer/vocab")
        assert resp.status_code == 500


class TestSample:
    """GET /tokenizer/sample"""

    @patch(MGR_TARGET)
    def test_get_sample(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()

        resp = client.get("/tokenizer/sample")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "samples" in data
        assert len(data["samples"]) > 0
        first = data["samples"][0]
        assert "word" in first
        assert "ids" in first
        assert "tokens" in first
        assert "count" in first

    @patch(MGR_TARGET)
    def test_sample_manager_error_returns_500(self, mock_get_mgr):
        mock_get_mgr.side_effect = RuntimeError("sample broke")
        resp = client.get("/tokenizer/sample")
        assert resp.status_code == 500


class TestTrain:
    """POST /tokenizer/train"""

    @patch(MGR_TARGET)
    def test_train_with_texts(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.stats.return_value = {"vocab_size": 128}
        mock_get_mgr.return_value = mgr

        resp = client.post(
            "/tokenizer/train",
            json={"vocab_size": 128, "texts": ["hello world", "foo bar"]},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "trained"
        assert data["corpus_size"] == 2
        mgr.train.assert_called_once_with(
            ["hello world", "foo bar"],
            vocab_size=128,
            min_frequency=3,
        )

    @patch("apps.api.server.routers.tokenizer.urllib.request.urlopen")
    @patch(MGR_TARGET)
    def test_train_empty_texts_downloads_default(self, mock_get_mgr, mock_urlopen):
        mgr = _mock_manager()
        mock_get_mgr.return_value = mgr
        mock_urlopen.return_value.read.return_value = b"line1\nline2\nline3\n"

        resp = client.post("/tokenizer/train", json={"vocab_size": 64, "texts": []})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "trained"
        assert data["corpus_size"] == 3

    @patch(MGR_TARGET)
    def test_train_default_vocab_size(self, mock_get_mgr):
        mgr = _mock_manager()
        mock_get_mgr.return_value = mgr

        client.post("/tokenizer/train", json={"texts": ["hello"]})
        mgr.train.assert_called_once_with(
            ["hello"],
            vocab_size=512,
            min_frequency=3,
        )

    @patch("apps.api.server.routers.tokenizer.urllib.request.urlopen")
    @patch(MGR_TARGET)
    def test_train_missing_body(self, mock_get_mgr, mock_urlopen):
        mgr = _mock_manager()
        mgr.stats.return_value = {"vocab_size": 256}
        mock_get_mgr.return_value = mgr
        mock_urlopen.return_value.read.return_value = b"line1\nline2\n"
        resp = client.post("/tokenizer/train", json={})
        assert resp.status_code == 200


class TestPretokenize:
    """POST /tokenizer/pretokenize"""

    @patch(MGR_TARGET)
    def test_pretokenize(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()

        resp = client.post("/tokenizer/pretokenize", json={"text": "hello world"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "pretokens" in data


class TestDecompose:
    """POST /tokenizer/decompose"""

    @patch(MGR_TARGET)
    def test_decompose(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()

        resp = client.post("/tokenizer/decompose", json={"text": "the"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["token"] == "the"
        assert "parts" in data or "merge_path" in data or "depth" in data

    @patch(MGR_TARGET)
    def test_decompose_not_found(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.decompose_token.side_effect = ValueError("token not found")
        mock_get_mgr.return_value = mgr

        resp = client.post("/tokenizer/decompose", json={"text": "zzzzz"})
        assert resp.status_code == 404


class TestAnalyzeCorpus:
    """POST /tokenizer/analyze"""

    @patch(MGR_TARGET)
    def test_analyze(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()

        resp = client.post("/tokenizer/analyze", json={"texts": ["hello world", "foo bar"]})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_analyze_missing_texts(self):
        resp = client.post("/tokenizer/analyze", json={})
        assert resp.status_code == 422

    @patch(MGR_TARGET)
    def test_analyze_manager_error_returns_500(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.analyze_corpus.side_effect = RuntimeError("analyze broke")
        mock_get_mgr.return_value = mgr
        resp = client.post("/tokenizer/analyze", json={"texts": ["hello"]})
        assert resp.status_code == 500

    def test_analyze_overlong_texts_returns_422(self):
        resp = client.post("/tokenizer/analyze", json={"texts": ["x"] * 501})
        assert resp.status_code == 422


class TestPretokenize:
    """POST /tokenizer/pretokenize"""

    @patch(MGR_TARGET)
    def test_pretokenize_manager_error_returns_500(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.show_pretokenization.side_effect = RuntimeError("pretokenize broke")
        mock_get_mgr.return_value = mgr
        resp = client.post("/tokenizer/pretokenize", json={"text": "hello"})
        assert resp.status_code == 500

    def test_pretokenize_missing_text_returns_422(self):
        resp = client.post("/tokenizer/pretokenize", json={})
        assert resp.status_code == 422

    @patch(MGR_TARGET)
    def test_pretokenize_overlong_text_returns_422(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()
        resp = client.post("/tokenizer/pretokenize", json={"text": "x" * 50001})
        assert resp.status_code == 422


class TestDecomposeValidation:
    """POST /tokenizer/decompose edge cases."""

    def test_decompose_missing_text_returns_422(self):
        resp = client.post("/tokenizer/decompose", json={})
        assert resp.status_code == 422

    @patch(MGR_TARGET)
    def test_decompose_manager_error_returns_500(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.decompose_token.side_effect = RuntimeError("decompose broke")
        mock_get_mgr.return_value = mgr
        resp = client.post("/tokenizer/decompose", json={"text": "the"})
        assert resp.status_code == 500


class TestMethodCoverage:
    """405s for method mismatches on all 10 routes."""

    def test_stats_wrong_method_405(self):
        resp = client.post("/tokenizer/stats")
        assert resp.status_code == 405

    def test_pretokenize_wrong_method_405(self):
        resp = client.get("/tokenizer/pretokenize")
        assert resp.status_code == 405

    def test_decompose_wrong_method_405(self):
        resp = client.get("/tokenizer/decompose")
        assert resp.status_code == 405

    def test_analyze_wrong_method_405(self):
        resp = client.get("/tokenizer/analyze")
        assert resp.status_code == 405

    def test_tokenize_wrong_method_405(self):
        resp = client.get("/tokenizer/tokenize")
        assert resp.status_code == 405

    def test_detokenize_wrong_method_405(self):
        resp = client.get("/tokenizer/detokenize")
        assert resp.status_code == 405

    def test_vocab_wrong_method_405(self):
        resp = client.post("/tokenizer/vocab")
        assert resp.status_code == 405

    def test_merges_wrong_method_405(self):
        resp = client.post("/tokenizer/merges")
        assert resp.status_code == 405

    def test_train_wrong_method_405(self):
        resp = client.get("/tokenizer/train")
        assert resp.status_code == 405

    def test_sample_wrong_method_405(self):
        resp = client.post("/tokenizer/sample")
        assert resp.status_code == 405
