"""Tests for the /token-tree router."""
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestTokenTreeRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_stats(self):
        resp = self.client.get("/token-tree/stats")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, dict)
        assert "vocab_size" in data

    def test_vocab(self):
        resp = self.client.get("/token-tree/vocab")
        assert resp.status_code == 200

    def test_merges(self):
        resp = self.client.get("/token-tree/merges")
        assert resp.status_code == 200

    def test_saved(self):
        resp = self.client.get("/token-tree/saved")
        assert resp.status_code == 200

    def test_matrix(self):
        resp = self.client.get("/token-tree/matrix")
        assert resp.status_code == 200

    def test_encode(self):
        resp = self.client.post("/token-tree/encode", json={"text": "hello world"})
        assert resp.status_code == 200
        data = _d(resp)
        assert "ids" in data or "tokens" in data

    def test_decode(self):
        resp = self.client.post("/token-tree/decode", json={"ids": [1, 2, 3]})
        assert resp.status_code == 200

    def test_train(self):
        resp = self.client.post("/token-tree/train", json={"vocab_size": 100})
        assert resp.status_code in (200, 400, 500)

    def test_similar(self):
        resp = self.client.post("/token-tree/similar", json={"token": "hello", "top_k": 5})
        assert resp.status_code in (200, 404, 500)

    def test_lineage(self):
        resp = self.client.post("/token-tree/lineage", json={"token": "hello"})
        assert resp.status_code in (200, 404, 500)

    def test_path(self):
        resp = self.client.post("/token-tree/path", json={"text": "hello"})
        assert resp.status_code == 200
