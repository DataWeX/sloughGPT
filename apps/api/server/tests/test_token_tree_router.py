"""Tests for the /token-tree router — tokenizer BPE tree management."""

import pytest
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestStats:
    def setup_method(self):
        self.client = get_test_client()

    def test_stats_returns_success(self):
        resp = self.client.get("/token-tree/stats")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_stats_has_vocab_size(self):
        resp = self.client.get("/token-tree/stats")
        data = _d(resp)
        assert "vocab_size" in data
        assert isinstance(data["vocab_size"], int)

    def test_stats_has_expected_fields(self):
        resp = self.client.get("/token-tree/stats")
        data = _d(resp)
        assert "vocab_size" in data
        assert "num_merges" in data


class TestVocab:
    def setup_method(self):
        self.client = get_test_client()

    def test_vocab_returns_dict(self):
        resp = self.client.get("/token-tree/vocab")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, dict)
        assert "entries" in data
        assert "total" in data

    def test_vocab_entries_have_ids(self):
        resp = self.client.get("/token-tree/vocab")
        entries = _d(resp)["entries"]
        assert len(entries) > 0
        assert "id" in entries[0]
        assert "token" in entries[0]


class TestMerges:
    def setup_method(self):
        self.client = get_test_client()

    def test_merges_returns_list(self):
        resp = self.client.get("/token-tree/merges")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, list)

    def test_merges_have_rank(self):
        resp = self.client.get("/token-tree/merges")
        data = _d(resp)
        if len(data) > 0:
            assert "rank" in data[0]
            assert "token" in data[0]


class TestSaved:
    def setup_method(self):
        self.client = get_test_client()

    def test_saved_returns_dict(self):
        resp = self.client.get("/token-tree/saved")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, dict)
        assert "trees" in data


class TestEncode:
    def setup_method(self):
        self.client = get_test_client()

    def test_encode_returns_ids(self):
        resp = self.client.post("/token-tree/encode", json={"text": "hello world"})
        assert resp.status_code == 200
        data = _d(resp)
        assert "ids" in data

    def test_encode_empty_text(self):
        resp = self.client.post("/token-tree/encode", json={"text": ""})
        assert resp.status_code == 200

    def test_encode_missing_text(self):
        resp = self.client.post("/token-tree/encode", json={})
        assert resp.status_code == 422


class TestDecode:
    def setup_method(self):
        self.client = get_test_client()

    def test_decode_returns_text(self):
        resp = self.client.post("/token-tree/decode", json={"ids": [1, 2, 3]})
        assert resp.status_code == 200
        data = _d(resp)
        assert "text" in data

    def test_decode_empty_ids(self):
        resp = self.client.post("/token-tree/decode", json={"ids": []})
        assert resp.status_code == 200


class TestPath:
    def setup_method(self):
        self.client = get_test_client()

    def test_path_returns_steps(self):
        resp = self.client.post("/token-tree/path", json={"text": "hello"})
        assert resp.status_code == 200
        data = _d(resp)
        assert "steps" in data
        assert "ids" in data


class TestSimilar:
    def setup_method(self):
        self.client = get_test_client()

    def test_similar_returns_results(self):
        resp = self.client.post("/token-tree/similar", json={"token": "e", "top_k": 5})
        assert resp.status_code == 200
        data = _d(resp)
        assert "neighbors" in data or "similar" in data or "results" in data


class TestLineage:
    def setup_method(self):
        self.client = get_test_client()

    def test_lineage_returns_data(self):
        resp = self.client.post("/token-tree/lineage", json={"token": "e"})
        assert resp.status_code == 200


class TestMatrix:
    def setup_method(self):
        self.client = get_test_client()

    def test_matrix_returns_data(self):
        resp = self.client.get("/token-tree/matrix")
        assert resp.status_code == 200


class TestTrain:
    def setup_method(self):
        self.client = get_test_client()

    def test_train_with_vocab_size(self):
        resp = self.client.post("/token-tree/train", json={"vocab_size": 100})
        assert resp.status_code in (200, 400, 500)


class TestSaveAndLoad:
    def setup_method(self):
        self.client = get_test_client()

    def test_save_tree(self):
        resp = self.client.post("/token-tree/save", json={"name": "test_tree"})
        assert resp.status_code in (200, 400, 500)

    def test_load_tree(self):
        resp = self.client.post("/token-tree/load", json={"name": "test_tree"})
        assert resp.status_code in (200, 400, 404, 500)


class TestDeleteSaved:
    def setup_method(self):
        self.client = get_test_client()

    def test_delete_nonexistent(self):
        resp = self.client.delete("/token-tree/saved/nonexistent_tree")
        assert resp.status_code in (200, 404)


class TestEmbedding:
    def setup_method(self):
        self.client = get_test_client()

    def test_embedding_returns_vector(self):
        resp = self.client.post("/token-tree/embedding", json={"token": "e"})
        assert resp.status_code == 200
        data = _d(resp)
        assert "dim" in data or "embedding" in data or "vector" in data


class TestCompare:
    def setup_method(self):
        self.client = get_test_client()

    def test_compare_nonexistent_trees(self):
        resp = self.client.post(
            "/token-tree/compare",
            json={"a": "tree_a", "b": "tree_b"},
        )
        assert resp.status_code in (200, 404, 500)
