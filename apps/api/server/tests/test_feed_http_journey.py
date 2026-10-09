"""End-to-end training-feed journey over a real HTTP socket.

Proves the advertised flow — *point a feed URL at the training flow and it reads
and trains on it* — without mocking the transport:

- the envelope is produced by the actual ``feeds._read_feed`` processor (the
  function behind ``GET /training/feed``),
- served over a real ``http.server`` TCP socket (not TestClient),
- consumed by the real ``TrainingFeedClient`` (``urllib``) with its offset
  cursor,
- and used to train a ``SloughGPTTrainer`` whose ``data_path`` is that URL.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
from urllib.parse import parse_qs, urlparse

import pytest
from training import feeds

from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig
from domain.training._internal.training_feed import TrainingFeedClient


def _corpus_record(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": "is that enough context for a follow-up?"},
            {"role": "assistant", "content": "yes, the exchange is recorded verbatim"},
        ],
        "meta": {"model": "test", "captured_at": "2026-09-22T00:00:00+00:00"},
    }


_SEED_PAIRS = [
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
    _corpus_record(
        "What makes a model robust to distribution shift over time?",
        "Continual ingestion of fresh data keeps the distribution aligned with "
        "the live environment the model is deployed into.",
    ),
]


class _FeedHandler(http.server.BaseHTTPRequestHandler):
    """Serve the feed contract via the real feeds._read_feed processor."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        parsed = urlparse(self.path)
        if parsed.path.startswith("/training/feed/"):
            source = parsed.path.split("/", 3)[-1]
        else:
            source = (parse_qs(parsed.query).get("source") or ["api-conversations"])[0]
        qs = parse_qs(parsed.query)
        offset = int((qs.get("offset") or ["0"])[0])
        limit = int((qs.get("limit") or ["500"])[0])
        body = json.dumps({"data": feeds._read_feed(source, offset, limit)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # noqa: ARG002 - silence request logs
        pass


@pytest.fixture()
def feed_server(tmp_path, monkeypatch):
    """Live HTTP feed on an ephemeral port backed by feeds._read_feed."""
    corpus_dir = tmp_path / "api_conversations"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = corpus_dir / "corpus.jsonl"
    corpus_file.write_text("\n".join(json.dumps(r) for r in _SEED_PAIRS) + "\n", encoding="utf-8")
    monkeypatch.setattr(feeds, "_SOURCES", {"api-conversations": str(corpus_file)})

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _FeedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        yield f"http://127.0.0.1:{port}/training/feed/api-conversations"
    finally:
        server.shutdown()
        server.server_close()


class TestRealHttpJourney:
    """The feed contract over an actual socket, end to end."""

    def test_client_pages_with_persisted_cursor(self, feed_server, tmp_path):
        store = tmp_path / "feed-cursor.json"
        client = TrainingFeedClient(feed_url=feed_server, offset_store_path=store)

        page = client.read(limit=2)
        assert page.total == 3
        assert page.offset == 0
        assert page.next_offset == 2
        assert len(page.records) == 2
        assert page.records[0]["messages"][0]["content"].startswith("Tell me about")

        client.save_offset(page.next_offset)
        resumed = TrainingFeedClient(feed_url=feed_server, offset_store_path=store)
        assert resumed.offset == 2
        tail = resumed.read()
        assert len(tail.records) == 1
        assert "distribution shift" in tail.records[0]["messages"][0]["content"]

    def test_trainer_trains_from_live_url(self, feed_server, tmp_path):
        cfg = TrainerConfig(
            vocab_size=0,
            n_embed=16,
            n_layer=1,
            n_head=2,
            block_size=8,
            dropout=0.0,
            batch_size=2,
            epochs=1,
            max_steps=2,
            gradient_accumulation_steps=1,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_interval=1,
            eval_interval=1000,
            checkpoint_interval=1000,
            warmup_steps=1,
            min_lr=1e-5,
            max_checkpoints=5,
            scheduler_type="cosine",
        )
        trainer = SloughGPTTrainer(feed_server, config=cfg)
        result = trainer.train()
        assert result.success is True
        assert result.global_step == 2
        assert os.path.exists(result.model_path)
        assert trainer.is_training is False
