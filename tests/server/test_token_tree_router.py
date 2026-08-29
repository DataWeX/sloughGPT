"""
Tests for the token-tree router — stats, train, similar, encode, decode, lineage.

Uses a standalone FastAPI app with only the router under test.
The router imports get_token_tree_manager at module level, so we patch the
name in the router's namespace.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.token_tree import router

app = FastAPI()
register_all_handlers(app)
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

MGR_TARGET = "apps.api.server.routers.token_tree.get_token_tree_manager"


def _mock_tree():
    """Create a mock TokenTree matching the interface the manager uses."""
    tree = MagicMock()
    tree.is_trained = True
    tree.stats.return_value = {
        "trained": True,
        "vocab_size": 128,
        "num_merges": 20,
        "num_base_tokens": 108,
        "embedding_points": 128,
        "embedding_compression_ratio": 2.0,
        "embed_dim": 16,
    }
    tree.resolve_token.return_value = 12
    tree.similar.return_value = [(20, 0.85), (7, 0.71), (3, 0.6)]
    tree.itos = {12: "quick</w>", 20: "brown</w>", 7: "fox</w>", 3: "the</w>"}
    tree.encode.return_value = [3, 12, 20]
    tree.decode.return_value = "the quick brown"
    tree.decompose.return_value = ["q", "u", "i", "c", "k", "</w>"]
    tree.show_tree.return_value = "quick</w>\n  q...\n"
    return tree


def _mock_manager():
    """Create a mock TokenTreeManager matching the real interface."""
    mgr = MagicMock()
    mgr.get_tree.return_value = _mock_tree()
    mgr.stats.return_value = _mock_tree().stats()
    mgr.train.return_value = _mock_tree()
    mgr.encode.return_value = {"tokens": ["the</w>", "quick</w>", "brown</w>"], "ids": [3, 12, 20]}
    mgr.decode.return_value = {"text": "the quick brown"}
    mgr.similar.return_value = {
        "query": "quick</w>",
        "neighbors": [
            {"id": 20, "token": "brown</w>", "score": 0.85},
            {"id": 7, "token": "fox</w>", "score": 0.71},
            {"id": 3, "token": "the</w>", "score": 0.6},
        ],
    }
    mgr.lineage.return_value = {
        "token": "quick</w>",
        "leaves": ["q", "u", "i", "c", "k", "</w>"],
        "tree": "quick</w>\n  q...\n",
    }
    mgr.matrix_summary.return_value = {
        "matrix": [128, 16],
        "norm_min": 0.5,
        "norm_mean": 0.8,
        "norm_max": 1.0,
        "dead_tokens": 2,
        "live_tokens": 126,
        "most_energetic": [
            ["quick</w>", 12, 1.0],
            ["brown</w>", 20, 0.98],
        ],
        "least_energetic": [
            ["the</w>", 3, 0.55],
        ],
    }
    mgr.compare.return_value = {
        "a": {"name": "tree-a", "stats": _mock_tree().stats(), "vocab": {}},
        "b": {"name": "tree-b", "stats": _mock_tree().stats(), "vocab": {}},
        "shared_tokens": 100,
        "only_a_tokens": 28,
        "only_b_tokens": 0,
        "shared_merges": 15,
        "only_a_merges": 5,
        "only_b_merges": 0,
        "shared_examples": [["the</w>", 42]],
        "only_a_examples": [["quick</w>", 9]],
        "only_b_examples": [],
    }
    return mgr


class TestStats:
    """GET /token-tree/stats"""

    @patch(MGR_TARGET)
    def test_get_stats(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()
        resp = client.get("/token-tree/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["vocab_size"] == 128
        assert data["embed_dim"] == 16
        assert data["trained"] is True
        assert data["embedding_compression_ratio"] == 2.0

    @patch(MGR_TARGET)
    def test_stats_manager_error_returns_500(self, mock_get_mgr):
        mock_get_mgr.side_effect = RuntimeError("stats broke")
        resp = client.get("/token-tree/stats")
        assert resp.status_code == 500

    def test_stats_wrong_method_405(self):
        resp = client.post("/token-tree/stats")
        assert resp.status_code == 405


class TestMerges:
    """GET /token-tree/merges"""

    @patch(MGR_TARGET)
    def test_get_merges(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.top_merges.return_value = [
            {"rank": 1, "left": "th", "right": "e", "token": "the", "count": 42},
            {"rank": 2, "left": "qu", "right": "ic", "token": "quic", "count": 30},
        ]
        mock_get_mgr.return_value = mgr
        resp = client.get("/token-tree/merges?top_n=2")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 2
        assert data[0]["token"] == "the"
        mgr.top_merges.assert_called_once_with(top_n=2)

    @patch(MGR_TARGET)
    def test_get_merges_default_top_n(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.top_merges.return_value = []
        mock_get_mgr.return_value = mgr
        client.get("/token-tree/merges")
        mgr.top_merges.assert_called_once_with(top_n=20)

    def test_get_merges_invalid_top_n_returns_422(self):
        resp = client.get("/token-tree/merges?top_n=0")
        assert resp.status_code == 422

    def test_merges_wrong_method_405(self):
        resp = client.post("/token-tree/merges")
        assert resp.status_code == 405


class TestTrain:
    """POST /token-tree/train"""

    @patch(MGR_TARGET)
    def test_train_with_texts(self, mock_get_mgr):
        mgr = _mock_manager()
        tree = mgr.train.return_value
        mock_get_mgr.return_value = mgr

        resp = client.post(
            "/token-tree/train",
            json={"texts": ["hello world", "foo bar"], "vocab_size": 128},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "trained"
        assert data["vocab_size"] == 128
        mgr.train.assert_called_once()
        call = mgr.train.call_args
        assert call.kwargs["vocab_size"] == 128

    @patch(MGR_TARGET)
    def test_train_defaults_when_no_texts(self, mock_get_mgr):
        mgr = _mock_manager()
        mock_get_mgr.return_value = mgr

        resp = client.post("/token-tree/train", json={"vocab_size": 64})
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "trained"
        mgr.train.assert_not_called()
        mgr.get_tree.assert_called_once_with(vocab_size=64, embed_dim=16)

    @patch(MGR_TARGET)
    def test_train_default_params(self, mock_get_mgr):
        mgr = _mock_manager()
        mock_get_mgr.return_value = mgr
        client.post("/token-tree/train", json={"texts": ["hello"]})
        call = mgr.train.call_args
        assert call.kwargs["vocab_size"] == 512
        assert call.kwargs["embed_dim"] == 16
        assert call.kwargs["min_frequency"] == 2

    @patch(MGR_TARGET)
    def test_train_invalid_vocab_returns_422(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()
        resp = client.post("/token-tree/train", json={"texts": ["x"], "vocab_size": 4})
        assert resp.status_code == 422

    def test_train_wrong_method_405(self):
        resp = client.get("/token-tree/train")
        assert resp.status_code == 405


class TestSimilar:
    """POST /token-tree/similar"""

    @patch(MGR_TARGET)
    def test_similar_returns_ranked_neighbors(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()
        resp = client.post("/token-tree/similar", json={"token": "quick", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["query"] == "quick</w>"
        assert len(data["neighbors"]) == 3
        assert data["neighbors"][0]["token"] == "brown</w>"
        assert data["neighbors"][0]["score"] == 0.85

    @patch(MGR_TARGET)
    def test_similar_default_top_k(self, mock_get_mgr):
        mgr = _mock_manager()
        mock_get_mgr.return_value = mgr
        client.post("/token-tree/similar", json={"token": "quick"})
        mgr.similar.assert_called_once_with("quick", top_k=5)

    @patch(MGR_TARGET)
    def test_similar_unknown_token_returns_404(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.similar.side_effect = KeyError("zzz")
        mock_get_mgr.return_value = mgr
        resp = client.post("/token-tree/similar", json={"token": "zzz"})
        assert resp.status_code == 404

    @patch(MGR_TARGET)
    def test_similar_manager_error_returns_500(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.similar.side_effect = RuntimeError("similar broke")
        mock_get_mgr.return_value = mgr
        resp = client.post("/token-tree/similar", json={"token": "quick"})
        assert resp.status_code == 500

    def test_similar_missing_token_returns_422(self):
        resp = client.post("/token-tree/similar", json={})
        assert resp.status_code == 422

    def test_similar_wrong_method_405(self):
        resp = client.get("/token-tree/similar")
        assert resp.status_code == 405


class TestEncode:
    """POST /token-tree/encode"""

    @patch(MGR_TARGET)
    def test_encode(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()
        resp = client.post("/token-tree/encode", json={"text": "the quick brown"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["ids"] == [3, 12, 20]
        assert len(data["tokens"]) == 3

    @patch(MGR_TARGET)
    def test_encode_empty_text(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.encode.return_value = {"tokens": [], "ids": []}
        mock_get_mgr.return_value = mgr
        resp = client.post("/token-tree/encode", json={"text": ""})
        assert resp.status_code == 200
        assert resp.json()["data"]["ids"] == []

    def test_encode_missing_text_returns_422(self):
        resp = client.post("/token-tree/encode", json={})
        assert resp.status_code == 422

    def test_encode_overlong_text_returns_422(self):
        resp = client.post("/token-tree/encode", json={"text": "x" * 50001})
        assert resp.status_code == 422


class TestDecode:
    """POST /token-tree/decode"""

    @patch(MGR_TARGET)
    def test_decode(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()
        resp = client.post("/token-tree/decode", json={"ids": [3, 12, 20]})
        assert resp.status_code == 200
        assert resp.json()["data"]["text"] == "the quick brown"

    def test_decode_missing_ids_returns_422(self):
        resp = client.post("/token-tree/decode", json={})
        assert resp.status_code == 422

    def test_decode_overlong_ids_returns_422(self):
        resp = client.post("/token-tree/decode", json={"ids": [0] * 10001})
        assert resp.status_code == 422

    def test_decode_wrong_method_405(self):
        resp = client.get("/token-tree/decode")
        assert resp.status_code == 405


class TestLineage:
    """POST /token-tree/lineage"""

    @patch(MGR_TARGET)
    def test_lineage(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()
        resp = client.post("/token-tree/lineage", json={"token": "quick"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["token"] == "quick</w>"
        assert "k" in data["leaves"]
        assert data["tree"]

    @patch(MGR_TARGET)
    def test_lineage_unknown_token_returns_404(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.lineage.side_effect = KeyError("zzz")
        mock_get_mgr.return_value = mgr
        resp = client.post("/token-tree/lineage", json={"token": "zzz"})
        assert resp.status_code == 404

    def test_lineage_missing_token_returns_422(self):
        resp = client.post("/token-tree/lineage", json={})
        assert resp.status_code == 422


class TestMatrix:
    @patch(MGR_TARGET)
    def test_get_matrix(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()
        resp = client.get("/token-tree/matrix")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["matrix"] == [128, 16]
        assert data["live_tokens"] == 126
        assert data["dead_tokens"] == 2
        assert data["most_energetic"][0][0] == "quick</w>"
        assert data["least_energetic"][0][2] == 0.55

    @patch(MGR_TARGET)
    def test_get_matrix_default_top_k(self, mock_get_mgr):
        mgr = _mock_manager()
        mock_get_mgr.return_value = mgr
        client.get("/token-tree/matrix")
        mgr.matrix_summary.assert_called_once_with(top_k=8)

    @patch(MGR_TARGET)
    def test_get_matrix_custom_top_k(self, mock_get_mgr):
        mgr = _mock_manager()
        mock_get_mgr.return_value = mgr
        resp = client.get("/token-tree/matrix", params={"top_k": 3})
        assert resp.status_code == 200
        mgr.matrix_summary.assert_called_once_with(top_k=3)

    @patch(MGR_TARGET)
    def test_get_matrix_disabled_embeddings(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.matrix_summary.return_value = {
            "matrix": None,
            "norm_min": 0.0,
            "norm_mean": 0.0,
            "norm_max": 0.0,
            "dead_tokens": 0,
            "live_tokens": 0,
            "most_energetic": [],
            "least_energetic": [],
        }
        mock_get_mgr.return_value = mgr
        resp = client.get("/token-tree/matrix")
        assert resp.status_code == 200
        assert resp.json()["data"]["matrix"] is None

    def test_get_matrix_invalid_top_k_returns_422(self):
        resp = client.get("/token-tree/matrix", params={"top_k": 0})
        assert resp.status_code == 422

    def test_matrix_wrong_method_405(self):
        resp = client.post("/token-tree/matrix")
        assert resp.status_code == 405


class TestCompare:
    """POST /token-tree/compare"""

    @patch(MGR_TARGET)
    def test_compare_returns_overlap(self, mock_get_mgr):
        mock_get_mgr.return_value = _mock_manager()
        resp = client.post("/token-tree/compare", json={"a": "tree-a", "b": "tree-b", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["a"]["name"] == "tree-a"
        assert data["b"]["name"] == "tree-b"
        assert data["shared_tokens"] == 100
        assert data["only_a_tokens"] == 28
        assert data["shared_examples"][0][0] == "the</w>"

    @patch(MGR_TARGET)
    def test_compare_default_top_k(self, mock_get_mgr):
        mgr = _mock_manager()
        mock_get_mgr.return_value = mgr
        client.post("/token-tree/compare", json={"a": "tree-a", "b": "tree-b"})
        mgr.compare.assert_called_once_with("tree-a", "tree-b", top_n=10)

    @patch(MGR_TARGET)
    def test_compare_custom_top_k(self, mock_get_mgr):
        mgr = _mock_manager()
        mock_get_mgr.return_value = mgr
        resp = client.post("/token-tree/compare", json={"a": "tree-a", "b": "tree-b", "top_k": 3})
        assert resp.status_code == 200
        mgr.compare.assert_called_once_with("tree-a", "tree-b", top_n=3)

    @patch(MGR_TARGET)
    def test_compare_missing_tree_returns_404(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.compare.side_effect = FileNotFoundError("Saved tree not found: ghost")
        mock_get_mgr.return_value = mgr
        resp = client.post("/token-tree/compare", json={"a": "tree-a", "b": "ghost"})
        assert resp.status_code == 404

    @patch(MGR_TARGET)
    def test_compare_self_returns_400(self, mock_get_mgr):
        mgr = _mock_manager()
        mgr.compare.side_effect = ValueError("Cannot compare a tree with itself")
        mock_get_mgr.return_value = mgr
        resp = client.post("/token-tree/compare", json={"a": "tree-a", "b": "tree-a"})
        assert resp.status_code == 400

    def test_compare_missing_fields_returns_422(self):
        resp = client.post("/token-tree/compare", json={"a": "tree-a"})
        assert resp.status_code == 422

    def test_compare_wrong_method_405(self):
        resp = client.get("/token-tree/compare")
        assert resp.status_code == 405
