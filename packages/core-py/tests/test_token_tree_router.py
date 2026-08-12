"""Tests for the token-tree API router (routers/token_tree.py).

Covers: stats, merges (with and without a query filter), train, similar,
encode, decode, lineage. Domain calls are mocked; only HTTP-level behavior
is tested. The router imports get_token_tree_manager at module level, so
patching 'routers.token_tree.get_token_tree_manager' works directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.token_tree import router  # noqa: E402


def _make_mgr(**overrides):
    defaults = dict(
        stats=lambda: {
            "trained": True,
            "vocab_size": 200,
            "num_merges": 100,
            "num_base_tokens": 90,
            "embedding_points": 200,
            "embedding_compression_ratio": 4.0,
            "embed_dim": 16,
        },
        top_merges=lambda top_n=20: [
            {"rank": 1, "left": "th", "right": "e", "token": "the", "count": 42},
        ],
        search_merges=lambda query, limit=20: [
            {"rank": 1, "left": "th", "right": "e", "token": "the", "count": 42},
        ],
        vocab_entries=lambda offset=0, limit=50: {
            "total": 3,
            "entries": [
                {"id": 0, "token": "<pad>", "freq": 0, "is_special": True, "is_merged": False},
                {"id": 1, "token": "t", "freq": 9, "is_special": False, "is_merged": False},
                {"id": 2, "token": "the</w>", "freq": 7, "is_special": False, "is_merged": True},
            ],
        },
        train=lambda texts, vocab_size=512, min_frequency=2, embed_dim=16: None,
        get_tree=lambda vocab_size=512, embed_dim=16: SimpleNamespace(
            stats=lambda: {
                "trained": True,
                "vocab_size": 512,
                "embedding_points": 512,
                "embedding_compression_ratio": 4.0,
                "embed_dim": 16,
            }
        ),
        similar=lambda token, top_k=5: {
            "query": "the",
            "neighbors": [{"id": 3, "token": "the", "score": 0.99}],
        },
        embedding_info=lambda token, top_k=8: {
            "token": "the",
            "id": 3,
            "dim": 8,
            "norm": 1.0,
            "top": [[0, 0.9], [1, -0.8]],
            "embedding_points": 200,
            "compression_ratio": 4.0,
        },
        encode=lambda text: {"tokens": ["the"], "ids": [3]},
        path=lambda text: {
            "steps": [{"remaining": "the</w>", "token": "the", "id": 3, "consumed": 7}],
            "ids": [3],
        },
        decode=lambda ids: {"text": "the"},
        lineage=lambda token: {
            "token": "the",
            "leaves": ["t", "h", "e"],
            "tree": "the",
        },
        list_saved=lambda: [
            {
                "name": "the-default",
                "path": "/data/token_trees/the-default",
                "vocab_size": 200,
                "num_merges": 100,
                "trained": True,
                "saved_at": 1000.0,
            }
        ],
        save=lambda name: {
            "name": name,
            "path": f"/data/token_trees/{name}",
            "vocab_size": 200,
            "num_merges": 100,
            "trained": True,
            "saved_at": 1000.0,
        },
        load=lambda name: {
            "name": name,
            "path": f"/data/token_trees/{name}",
            "vocab_size": 200,
            "num_merges": 100,
            "trained": True,
            "saved_at": 1000.0,
        },
        delete_saved=lambda name: True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _app():
    app = FastAPI()
    app.include_router(router)
    return app


MOCK_TARGET = "routers.token_tree.get_token_tree_manager"


class TestStats:
    @patch(MOCK_TARGET)
    def test_returns_stats(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).get("/token-tree/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["vocab_size"] == 200
        assert data["embed_dim"] == 16


class TestMerges:
    @patch(MOCK_TARGET)
    def test_returns_top_merges_without_query(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).get("/token-tree/merges?top_n=5")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data[0]["rank"] == 1
        assert data[0]["token"] == "the"

    @patch(MOCK_TARGET)
    def test_filters_merges_when_query_given(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).get("/token-tree/merges?top_n=10&query=the")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data[0]["left"] == "th"

    @patch(MOCK_TARGET)
    def test_query_is_optional_and_default_empty(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).get("/token-tree/merges")
        assert resp.status_code == 200
        assert resp.json()["data"]


class TestVocab:
    @patch(MOCK_TARGET)
    def test_returns_paged_entries(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).get("/token-tree/vocab?limit=3&offset=0")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 3
        assert len(data["entries"]) == 3
        assert data["entries"][0]["token"] == "<pad>"
        assert data["entries"][0]["is_special"] is True

    @patch(MOCK_TARGET)
    def test_accepts_offset_and_limit_params(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).get("/token-tree/vocab?limit=10&offset=5")
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] == 3

    @patch(MOCK_TARGET)
    def test_defaults_to_no_params(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).get("/token-tree/vocab")
        assert resp.status_code == 200
        assert resp.json()["data"]["entries"]

    @patch(MOCK_TARGET)
    def test_rejects_invalid_limit(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).get("/token-tree/vocab?limit=0")
        assert resp.status_code == 422


class TestTrain:
    @patch(MOCK_TARGET)
    def test_trains_with_default_corpus(self, mock_get):
        mgr = _make_mgr()
        mock_get.return_value = mgr
        resp = TestClient(_app()).post("/token-tree/train", json={"vocab_size": 512})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "trained"
        assert data["vocab_size"] == 512


class TestSimilar:
    @patch(MOCK_TARGET)
    def test_returns_neighbors(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).post("/token-tree/similar", json={"token": "the", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["query"] == "the"
        assert data["neighbors"][0]["score"] == 0.99


class TestEmbedding:
    @patch(MOCK_TARGET)
    def test_returns_embedding_info(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).post("/token-tree/embedding", json={"token": "the", "top_k": 2})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["token"] == "the"
        assert data["id"] == 3
        assert data["dim"] == 8
        assert data["norm"] == 1.0
        assert data["top"][0] == [0, 0.9]

    @patch(MOCK_TARGET)
    def test_top_k_defaults_to_eight(self, mock_get):
        mgr = _make_mgr()
        mgr.embedding_info = Mock(return_value={"token": "the", "id": 3, "dim": 8, "norm": 1.0, "top": [], "embedding_points": 200, "compression_ratio": 4.0})
        mock_get.return_value = mgr
        resp = TestClient(_app()).post("/token-tree/embedding", json={"token": "the"})
        assert resp.status_code == 200
        mgr.embedding_info.assert_called_once_with("the", top_k=8)

    @patch(MOCK_TARGET)
    def test_rejects_invalid_top_k(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).post("/token-tree/embedding", json={"token": "the", "top_k": 0})
        assert resp.status_code == 422

    @patch(MOCK_TARGET)
    def test_unknown_token_returns_404(self, mock_get):
        mgr = _make_mgr()
        mgr.embedding_info = Mock(side_effect=KeyError("no such token"))
        mock_get.return_value = mgr
        resp = TestClient(_app()).post("/token-tree/embedding", json={"token": "zzz"})
        assert resp.status_code == 404

    @patch(MOCK_TARGET)
    def test_disabled_embeddings_return_422(self, mock_get):
        mgr = _make_mgr()
        mgr.embedding_info = Mock(side_effect=ValueError("Token embeddings are not enabled (embed_dim = 0)"))
        mock_get.return_value = mgr
        resp = TestClient(_app()).post("/token-tree/embedding", json={"token": "the"})
        assert resp.status_code == 422


class TestEncodeDecode:
    @patch(MOCK_TARGET)
    def test_encode_returns_tokens_and_ids(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).post("/token-tree/encode", json={"text": "the"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["tokens"] == ["the"]
        assert data["ids"] == [3]

    @patch(MOCK_TARGET)
    def test_decode_returns_text(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).post("/token-tree/decode", json={"ids": [3]})
        assert resp.status_code == 200
        assert resp.json()["data"]["text"] == "the"


class TestPath:
    @patch(MOCK_TARGET)
    def test_returns_trace_steps(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).post("/token-tree/path", json={"text": "the"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["ids"] == [3]
        assert data["steps"][0]["token"] == "the"
        assert data["steps"][0]["remaining"] == "the</w>"

    @patch(MOCK_TARGET)
    def test_passes_text_to_manager(self, mock_get):
        mgr = _make_mgr()
        mgr.path = Mock(return_value={"steps": [], "ids": []})
        mock_get.return_value = mgr
        resp = TestClient(_app()).post("/token-tree/path", json={"text": "quick brown"})
        assert resp.status_code == 200
        mgr.path.assert_called_once_with("quick brown")


class TestLineage:
    @patch(MOCK_TARGET)
    def test_returns_lineage(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).post("/token-tree/lineage", json={"token": "the"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["leaves"] == ["t", "h", "e"]
        assert data["tree"]


class TestSavedTrees:
    @patch(MOCK_TARGET)
    def test_lists_saved_trees(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).get("/token-tree/saved")
        assert resp.status_code == 200
        trees = resp.json()["data"]["trees"]
        assert trees[0]["name"] == "the-default"
        assert trees[0]["trained"] is True


class TestSaveTree:
    @patch(MOCK_TARGET)
    def test_saves_under_given_name(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).post("/token-tree/save", json={"name": "shakespeare"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "shakespeare"
        assert data["vocab_size"] == 200

    @patch(MOCK_TARGET)
    def test_rejects_empty_name(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).post("/token-tree/save", json={"name": ""})
        assert resp.status_code == 422

    @patch(MOCK_TARGET)
    def test_rejects_invalid_name_from_manager(self, mock_get):
        mgr = _make_mgr(save=Mock(side_effect=ValueError("bad name")))
        mock_get.return_value = mgr
        resp = TestClient(_app()).post("/token-tree/save", json={"name": "../escape"})
        assert resp.status_code == 422


class TestLoadTree:
    @patch(MOCK_TARGET)
    def test_loads_saved_tree(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).post("/token-tree/load", json={"name": "the-default"})
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "the-default"

    @patch(MOCK_TARGET)
    def test_missing_tree_returns_404(self, mock_get):
        mgr = _make_mgr(load=Mock(side_effect=FileNotFoundError("no such tree")))
        mock_get.return_value = mgr
        resp = TestClient(_app()).post("/token-tree/load", json={"name": "missing"})
        assert resp.status_code == 404

    @patch(MOCK_TARGET)
    def test_invalid_name_returns_422(self, mock_get):
        mgr = _make_mgr(load=Mock(side_effect=ValueError("bad name")))
        mock_get.return_value = mgr
        resp = TestClient(_app()).post("/token-tree/load", json={"name": "a/b"})
        assert resp.status_code == 422


class TestDeleteSavedTree:
    @patch(MOCK_TARGET)
    def test_deletes_saved_tree(self, mock_get):
        mock_get.return_value = _make_mgr()
        resp = TestClient(_app()).delete("/token-tree/saved/the-default")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True

    @patch(MOCK_TARGET)
    def test_missing_tree_returns_404(self, mock_get):
        mgr = _make_mgr(delete_saved=lambda name: False)
        mock_get.return_value = mgr
        resp = TestClient(_app()).delete("/token-tree/saved/missing")
        assert resp.status_code == 404

    @patch(MOCK_TARGET)
    def test_invalid_name_returns_422(self, mock_get):
        mgr = _make_mgr(delete_saved=Mock(side_effect=ValueError("bad name")))
        mock_get.return_value = mgr
        resp = TestClient(_app()).delete("/token-tree/saved/.hidden")
        assert resp.status_code == 422
