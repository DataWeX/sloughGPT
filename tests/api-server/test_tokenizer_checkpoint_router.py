"""Router-level checkpoint binding (card 07c4e10a).

The `/tokenizer` endpoints must be able to inspect a CHECKPOINT's own
tokenizer (resolved from `.soul` metadata) rather than only the process-global
singleton, must LABEL which one they used, and must report round-trip fidelity.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

from domain.training._internal.token_tree import TokenTree
from domain.training._internal.tokenizer_manager import TokenizerManager

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app():
    from routers.tokenizer import TokenizerRouter

    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(TokenizerRouter().router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: _AUTH_USER
    return _app


@pytest.fixture(autouse=True)
def _reset_singleton():
    TokenizerManager._instance = None
    yield
    TokenizerManager._instance = None


@pytest.fixture
def client_with_soul(tmp_path, monkeypatch):
    """A client whose checkpoint roots point at a tmp soul embedding a tree."""
    from domain.training._internal import checkpoints

    monkeypatch.setattr(checkpoints, "ckpt_roots", lambda: (tmp_path,))

    tree = TokenTree()
    tree.train(["hello world", "hello there", "world of code"], vocab_size=64)
    (tmp_path / "demo.soul").write_bytes(b"present")
    (tmp_path / "demo.soul.meta.json").write_text(
        json.dumps(
            {
                "soul_name": "demo",
                "metadata": {
                    "vocab_size": tree.vocab_size,
                    "tokenizer": {"type": "token_tree", "tree": tree.to_dict()},
                    "tokenizer_format": "native/token-tree",
                },
            }
        ),
        encoding="utf-8",
    )
    return TestClient(_build_app()), tree


class TestTokenizeCheckpointBinding:
    def test_tokenize_without_checkpoint_is_singleton(self, client_with_soul):
        client, _tree = client_with_soul
        r = client.post("/tokenizer/tokenize", json={"text": "hello world"})
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["source"]["kind"] == "singleton"

    def test_tokenize_with_checkpoint_uses_checkpoint_tokenizer(self, client_with_soul):
        client, tree = client_with_soul
        r = client.post(
            "/tokenizer/tokenize", json={"text": "hello world", "checkpoint": "demo.soul"}
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["source"]["kind"] == "checkpoint"
        assert data["source"]["path"].endswith("demo.soul")
        # The ids are the CHECKPOINT tree's encoding, not the singleton's.
        assert data["ids"] == tree.encode("hello world")
        assert "fidelity" in data

    def test_tokenize_reports_round_trip_fidelity(self, client_with_soul):
        client, _tree = client_with_soul
        r = client.post(
            "/tokenizer/tokenize", json={"text": "hello world", "checkpoint": "demo.soul"}
        )
        fid = r.json()["data"]["fidelity"]
        # decode(encode(text)) == text for this corpus.
        assert fid["round_trip"] is True

    def test_tokenize_unknown_checkpoint_404(self, client_with_soul):
        client, _tree = client_with_soul
        r = client.post("/tokenizer/tokenize", json={"text": "hello", "checkpoint": "ghost.soul"})
        assert r.status_code == 404


class TestDetokenizeCheckpointBinding:
    def test_detokenize_with_checkpoint_labels_source_and_fidelity(self, client_with_soul):
        client, tree = client_with_soul
        ids = tree.encode("hello world")
        r = client.post("/tokenizer/detokenize", json={"ids": ids, "checkpoint": "demo.soul"})
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["source"]["kind"] == "checkpoint"
        assert data["text"] == tree.decode(ids)
        assert "fidelity" in data

    def test_detokenize_without_checkpoint_is_singleton(self, client_with_soul):
        client, _tree = client_with_soul
        r = client.post("/tokenizer/detokenize", json={"ids": [1, 2, 3]})
        assert r.status_code == 200
        assert r.json()["data"]["source"]["kind"] == "singleton"


class TestStatsVocabMergesCheckpoint:
    def test_stats_with_checkpoint(self, client_with_soul):
        client, tree = client_with_soul
        r = client.get("/tokenizer/stats", params={"checkpoint": "demo.soul"})
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["source"]["kind"] == "checkpoint"
        assert data["vocab_size"] == tree.vocab_size
        assert data["trained"] is True

    def test_vocab_with_checkpoint(self, client_with_soul):
        client, tree = client_with_soul
        r = client.get("/tokenizer/vocab", params={"checkpoint": "demo.soul", "limit": 10})
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["source"]["kind"] == "checkpoint"
        assert data["entries"][0]["token"] == tree.vocab[0]

    def test_merges_with_checkpoint(self, client_with_soul):
        client, tree = client_with_soul
        r = client.get("/tokenizer/merges", params={"checkpoint": "demo.soul", "limit": 5})
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["source"]["kind"] == "checkpoint"
        assert data["total"] == len(tree.merges)
