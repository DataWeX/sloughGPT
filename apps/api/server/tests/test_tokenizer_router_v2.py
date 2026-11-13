"""Tests for the /tokenizer router."""

from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


class TestTokenizerRouter:
    def setup_method(self):
        self.client = get_test_client()

    def test_stats(self):
        resp = self.client.get("/tokenizer/stats")
        assert resp.status_code == 200

    def test_tokenize(self):
        resp = self.client.post("/tokenizer/tokenize", json={"text": "hello world"})
        assert resp.status_code == 200

    def test_detokenize(self):
        resp = self.client.post("/tokenizer/detokenize", json={"ids": [1, 2, 3]})
        assert resp.status_code == 200

    def test_vocab(self):
        resp = self.client.get("/tokenizer/vocab")
        assert resp.status_code == 200

    def test_merges(self):
        resp = self.client.get("/tokenizer/merges")
        assert resp.status_code == 200

    def test_pretokenize(self):
        resp = self.client.post("/tokenizer/pretokenize", json={"text": "hello world"})
        assert resp.status_code == 200

    def test_decompose(self):
        resp = self.client.post("/tokenizer/decompose", json={"text": "hello"})
        assert resp.status_code in (200, 404, 422)

    def test_analyze(self):
        resp = self.client.post("/tokenizer/analyze", json={"texts": ["hello world"]})
        assert resp.status_code == 200

    def test_sample(self):
        resp = self.client.get("/tokenizer/sample")
        assert resp.status_code == 200
