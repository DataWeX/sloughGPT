"""Continual learner: ring buffer, multi-source web ingest, background fine-tune.

Uses SloTransformer (decoder-only with RoPE/SwiGLU/RMSNorm) — fine-tunes
the same model architecture that runs inference. Integrates with KnowledgeIngestor
for live web learning (RSS feeds, search, article scraping) and KnowledgeMemory
for structured fact storage.

Flow:
  RSS feeds ─┐
  Web search ─┼→ KnowledgeIngestor → KnowledgeMemory (facts by topic)
  URLs       ─┘         ↘
                   ContinualLearner (fine-tune SloTransformer on article text)

Usage:
    learner = get_learner()
    learner.ingest_text("Some training text...")
    learner.search_and_learn("latest AI breakthroughs")
    learner.subscribe_feed("https://news.ycombinator.com/rss")
    learner.ingest_conversation(pairs=[("user: hi", "assistant: hello")])
    status = learner.status()
"""

from __future__ import annotations

import os
import json
import time
import struct
import logging
import threading
from typing import Optional
from pathlib import Path

import numpy as np

from domains.shared import find_repo_root
from domains.training.slonet import (
    SloTransformer, SloAdam,
    cross_entropy, tensor, export_to_sou, import_from_sou,
)

logger = logging.getLogger("slo.learner")

CHAR_SET = " abcdefghijklmnopqrstuvwxyz0123456789.,!?-'"
STOI = {c: i for i, c in enumerate(CHAR_SET)}
ITOS = {i: c for i, c in enumerate(CHAR_SET)}
UNK = 0
VOCAB = len(CHAR_SET)

LEARNER_STATE_DIR = find_repo_root(Path(__file__).resolve()) / "data" / "learner"
LEARNER_STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = LEARNER_STATE_DIR / "continual.soul"

TRAIN_SEQ_LEN = 32
TRAIN_BATCH_SIZE = 8
DEFAULT_N_EMBED = 192
DEFAULT_N_LAYER = 4
DEFAULT_N_HEAD = 4
BUFFER_CAPACITY = 10000
INGEST_THRESHOLD = 512
TRAIN_INTERVAL = 30.0


def _tokenize(text: str) -> list[int]:
    return [STOI.get(c, UNK) for c in text.lower() if c in STOI]


def _detokenize(ids: list[int]) -> str:
    return "".join(ITOS.get(i, "?") for i in ids)


def _build_transformer(
    n_embed: int = DEFAULT_N_EMBED,
    n_layer: int = DEFAULT_N_LAYER,
    n_head: int = DEFAULT_N_HEAD,
    soul_name: str = "continual",
) -> SloTransformer:
    """Build a fresh SloTransformer for continual learning."""
    return SloTransformer(
        vocab_size=VOCAB,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=64,
        max_seq_len=256,
        dropout=0.1,
        use_rope=True,
        tie_weights=True,
        soul_name=soul_name,
        soul_traits={"warmth": 0.5, "creativity": 0.5, "curiosity": 0.8, "confidence": 0.5},
    )


class ContinualLearner:
    """Accumulates text + conversation data, fine-tunes SloTransformer.

    Maintains a ring buffer of token IDs, triggers training when buffer
    hits INGEST_THRESHOLD new tokens, and runs in a background thread
    to avoid blocking inference.

    The learner uses a SloTransformer (same architecture as the main
    inference model) — weight updates directly improve the model that
    generates responses.

    Integrates with KnowledgeIngestor for live web learning:
    - RSS feeds polled in background → article text → training buffer
    - Web search → full article fetch → facts + training buffer
    - KnowledgeMemory stores structured facts by topic
    """

    def __init__(
        self,
        n_embed: int = DEFAULT_N_EMBED,
        n_layer: int = DEFAULT_N_LAYER,
        n_head: int = DEFAULT_N_HEAD,
        soul_name: str = "continual",
        lr: float = 0.0005,
    ):
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.soul_name = soul_name
        self.lr = lr
        self.total_tokens_ingested = 0
        self.train_steps_completed = 0
        self.current_loss = 0.0
        self.loss_history: list[dict] = []
        self._last_train_time = time.time()
        self._new_since_last_train = 0
        self._lock = threading.Lock()
        self._running = True

        # Load existing model or create fresh
        self.net: SloTransformer = self._load_or_create()

        # Ring buffer
        self.buffer: list[int] = []

        # Knowledge ingestion pipeline
        from domains.learner.knowledge import get_knowledge_ingestor, get_knowledge_memory
        self.ingestor = get_knowledge_ingestor()
        self.knowledge = get_knowledge_memory()
        # Start background RSS polling (every 10 min)
        self.ingestor.start_background_polling(interval=600)

        # Background trainer thread
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()

    def _load_or_create(self) -> SloTransformer:
        """Load existing checkpoint or create a fresh SloTransformer."""
        if STATE_PATH.exists():
            try:
                net = import_from_sou(str(STATE_PATH))
                if isinstance(net, SloTransformer):
                    logger.info(f"Loaded learner transformer from {STATE_PATH}", extra={"tag": "INF"})
                    return net
                logger.warning("Existing checkpoint is not a SloTransformer — recreating", extra={"tag": "INF"})
            except Exception as e:
                logger.warning(f"Failed to load learner state: {e}", extra={"tag": "INF"})

        net = _build_transformer(
            n_embed=self.n_embed,
            n_layer=self.n_layer,
            n_head=self.n_head,
            soul_name=self.soul_name,
        )
        logger.info(
            f"Created fresh SloTransformer learner "
            f"(vocab={VOCAB}, embed={self.n_embed}, layers={self.n_layer}, heads={self.n_head})",
            extra={"tag": "INF"}
        )
        return net

    def _save_checkpoint(self):
        """Save learner model to disk as .soul."""
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        export_to_sou(self.net, str(STATE_PATH))
        logger.info(f"Saved learner checkpoint ({STATE_PATH.stat().st_size / 1024:.0f} KB)", extra={"tag": "INF"})

    # -----------------------------------------------------------------
    # INGESTION
    # -----------------------------------------------------------------

    def ingest_text(self, text: str):
        """Tokenize text and add to training buffer."""
        ids = _tokenize(text)
        if not ids:
            return
        with self._lock:
            self.buffer.extend(ids)
            if len(self.buffer) > BUFFER_CAPACITY:
                self.buffer = self.buffer[-BUFFER_CAPACITY:]
            self.total_tokens_ingested += len(ids)
            self._new_since_last_train += len(ids)

    def ingest_conversation(self, pairs: list[tuple[str, str]]):
        """Add (user_msg, assistant_msg) pairs to buffer.

        Args:
            pairs: list of (user_text, assistant_text) tuples
        """
        for user_text, assistant_text in pairs:
            combined = f"user: {user_text} assistant: {assistant_text}"
            self.ingest_text(combined)

    # -----------------------------------------------------------------
    # WEB SEARCH + LIVE LEARNING
    # -----------------------------------------------------------------

    def search_and_learn(self, query: str, max_results: int = 5) -> dict:
        """Search web, fetch full articles, store facts + fine-tune.

        Uses KnowledgeIngestor for full article scraping and structured
        fact storage. Article text also feeds into the training buffer
        for gradient updates. Quality filter applied before storage.

        Args:
            query: search query string
            max_results: number of articles to ingest

        Returns:
            dict with new_facts, rejected, tokens_ingested, filter_stats
        """
        result = self.ingestor.search_and_ingest(query, max_results=max_results)
        new_facts = result.get("new_facts", 0)
        rejected = result.get("rejected", 0)

        if new_facts == 0:
            return {"new_facts": 0, "rejected": rejected, "tokens_ingested": 0, "filter_stats": result.get("stats", {})}

        # Feed article text into training buffer for fine-tuning
        topics = self.knowledge.search(query, top_k=3)
        total_tokens = 0
        for fact in topics:
            content = fact.get("content", "")
            text = f"web: {content}"
            self.ingest_text(text)
            total_tokens += len(_tokenize(text))

        logger.info(f"search_and_learn({query!r}): {new_facts} facts, {rejected} rejected, {total_tokens} tokens", extra={"tag": "INF"})
        return {
            "new_facts": new_facts,
            "rejected": rejected,
            "tokens_ingested": total_tokens,
            "filter_stats": result.get("stats", {}),
        }

    def search_knowledge(self, query: str, top_k: int = 5) -> list[dict]:
        """Search learned knowledge by keyword. Returns relevant facts."""
        return self.knowledge.search(query, top_k=top_k)

    def query_knowledge(self, topic: str) -> list[dict]:
        """Get all facts for a topic."""
        return self.knowledge.get_topic_facts(topic)

    # -----------------------------------------------------------------
    # RSS FEEDS
    # -----------------------------------------------------------------

    def subscribe_feed(self, url: str, poll_interval: int = 3600) -> bool:
        """Subscribe to an RSS/Atom feed. New articles are auto-ingested."""
        result = self.ingestor.subscribe_feed(url, poll_interval)
        if result:
            # Do an immediate first fetch
            new = self.ingestor.poll_feeds(max_articles=5)
            new_count = new.get("new_articles", 0)
            if new_count > 0:
                for topic in self.knowledge.all_topics():
                    for fact in self.knowledge.get_topic_facts(topic):
                        self.ingest_text(f"rss: {fact.get('content', '')}")
            logger.info(f"Feed {url}: {new_count} initial articles ingested", extra={"tag": "INF"})
        return result

    def unsubscribe_feed(self, url: str) -> bool:
        return self.ingestor.unsubscribe_feed(url)

    def list_feeds(self) -> list[dict]:
        return self.ingestor.list_feeds()

    def ingest_url(self, url: str) -> dict:
        """Ingest a single URL: scrape article, store facts, feed buffer."""
        result = self.ingestor.ingest_url(url)
        if result.get("new_facts", 0) > 0:
            for topic in self.knowledge.all_topics():
                for fact in self.knowledge.get_topic_facts(topic):
                    if fact.get("url") == url:
                        self.ingest_text(f"article: {fact.get('content', '')}")
                        break
        return result

    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------

    def _train_step(self):
        """Run one fine-tune step on the ring buffer data.

        Extracts (x, y) pairs from the buffer, runs SloTransformer
        forward + backward, and updates weights via SloAdam.
        Low learning rate for stable continual learning.
        """
        if len(self.buffer) < TRAIN_SEQ_LEN + 1:
            return

        with self._lock:
            buf = list(self.buffer)
            self._new_since_last_train = 0
        self._last_train_time = time.time()

        opt = SloAdam(lr=self.lr)
        losses = []

        for start in range(0, len(buf) - TRAIN_SEQ_LEN, TRAIN_BATCH_SIZE):
            chunk = buf[start:start + TRAIN_BATCH_SIZE + TRAIN_SEQ_LEN]

            xi = chunk[:TRAIN_SEQ_LEN]
            yi = chunk[1:TRAIN_SEQ_LEN + 1]

            x = tensor([xi], requires_grad=True)
            y = tensor([yi])

            logits, loss = self.net.forward(x, targets=y)
            if loss is None:
                continue
            loss.backward()
            opt.step(self.net.parameters())
            losses.append(float(loss.data.flatten()[0]))

        if losses:
            avg_loss = sum(losses) / len(losses)
            self.current_loss = avg_loss
            self.train_steps_completed += 1
            self.loss_history.append({
                "step": self.train_steps_completed,
                "loss": round(avg_loss, 4),
                "tokens": len(buf) if 'buf' in dir() else 0,
                "timestamp": time.time(),
            })
            if len(self.loss_history) > 500:
                self.loss_history = self.loss_history[-500:]
            self._save_checkpoint()

    def _background_loop(self):
        """Background thread: poll feeds, train when enough new data arrives."""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = None
        feed_check_counter = 0
        while self._running:
            time.sleep(5)
            feed_check_counter += 1
            try:
                # Feed new RSS article content into training buffer every ~30s
                if feed_check_counter % 6 == 0:
                    for topic in self.knowledge.all_topics():
                        facts = self._safe_get_facts(topic)
                        for fact in facts:
                            if fact.get("source") == "rss":
                                self.ingest_text(f"rss: {fact.get('content', '')}")
                            break  # one per topic per cycle to avoid flooding
                # Train when threshold reached
                elapsed = time.time() - self._last_train_time
                if self._new_since_last_train >= INGEST_THRESHOLD and elapsed >= TRAIN_INTERVAL:
                    self._train_step()
            except Exception as e:
                logger.error(f"Background train error: {e}", extra={"tag": "INF"})
        if loop is not None:
            loop.close()

    def _safe_get_facts(self, topic: str) -> list[dict]:
        """Get facts for a topic, safe to call from background thread."""
        try:
            return self.knowledge.get_topic_facts(topic)
        except Exception:
            return []

    # -----------------------------------------------------------------
    # STATUS / CONTROL
    # -----------------------------------------------------------------

    def status(self) -> dict:
        """Return learner state summary with knowledge stats."""
        with self._lock:
            buf_size = len(self.buffer)
        know_stats = self.knowledge.stats()
        feed_count = len(self.ingestor.list_feeds())
        filter_stats = self.ingestor.filter.get_stats()
        filter_config = self.ingestor.filter.get_config()
        return {
            "soul_name": self.soul_name,
            "total_tokens_ingested": self.total_tokens_ingested,
            "train_steps_completed": self.train_steps_completed,
            "current_loss": self.current_loss,
            "loss_history": self.loss_history[-50:],
            "buffer_size": buf_size,
            "buffer_capacity": BUFFER_CAPACITY,
            "pending_tokens": self._new_since_last_train,
            "arch": "transformer",
            "n_embed": self.n_embed,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "vocab_size": VOCAB,
            "knowledge": know_stats,
            "feeds_subscribed": feed_count,
            "filter_stats": filter_stats,
            "filter_config": {k: v for k, v in filter_config.items() if k != "topic_blacklist"},
        }

    def train_now(self) -> dict:
        """Force an immediate training step. Returns status after training."""
        self._train_step()
        return self.status()

    def evaluate(self, text: Optional[str] = None) -> dict:
        """Evaluate the learner on test data and return metrics.

        Args:
            text: optional test text. If None, uses the ring buffer (80/20 split).

        Returns:
            dict with loss, perplexity, eval_tokens, and history
        """
        import math

        if text:
            ids = _tokenize(text)
            if len(ids) < TRAIN_SEQ_LEN + 1:
                return {"loss": 0.0, "perplexity": 0.0, "eval_tokens": 0, "error": "text too short"}
            chunks = [ids[i:i+TRAIN_SEQ_LEN+1] for i in range(0, len(ids), TRAIN_BATCH_SIZE)
                      if i + TRAIN_SEQ_LEN + 1 <= len(ids)]
        else:
            with self._lock:
                buf = list(self.buffer)
            if len(buf) < TRAIN_SEQ_LEN + 1:
                return {"loss": 0.0, "perplexity": 0.0, "eval_tokens": 0, "error": "buffer too small"}
            split = int(len(buf) * 0.8)
            test_buf = buf[split:]
            chunks = [test_buf[i:i+TRAIN_SEQ_LEN+1] for i in range(0, len(test_buf), TRAIN_BATCH_SIZE)
                      if i + TRAIN_SEQ_LEN + 1 <= len(test_buf)]

        if not chunks:
            return {"loss": 0.0, "perplexity": 0.0, "eval_tokens": 0, "error": "no eval chunks"}

        total_loss = 0.0
        total_tokens = 0
        for chunk in chunks:
            xi = chunk[:TRAIN_SEQ_LEN]
            yi = chunk[1:TRAIN_SEQ_LEN + 1]
            x = tensor([xi])
            y = tensor([yi])
            _, loss = self.net.forward(x, targets=y)
            if loss is not None:
                lv = float(loss.data.flatten()[0])
                total_loss += lv * len(yi)
                total_tokens += len(yi)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss > 0 else float('inf')

        return {
            "loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 2),
            "eval_tokens": total_tokens,
            "train_steps": self.train_steps_completed,
            "total_tokens_ingested": self.total_tokens_ingested,
            "buffer_size": len(self.buffer),
        }

    def deploy(self, name: Optional[str] = None) -> dict:
        """Export the learner's SloTransformer as a deployable .soul file.

        Args:
            name: optional checkpoint name (default: ``learner-step-{N}``)

        Returns:
            dict with path, soul_name, steps, loss, and file_size
        """
        from domains.training.slonet import export_to_sou
        safe = self.soul_name.lower().replace(" ", "_")[:32]
        step = self.train_steps_completed
        name = name or f"learner-{safe}-step-{step}"
        deploy_dir = Path(str(Path.home() / ".local" / "share" / "sloughgpt" / "souls"))
        deploy_dir.mkdir(parents=True, exist_ok=True)
        path = str(deploy_dir / f"{name}.soul")
        export_to_sou(self.net, path)
        loss = self.current_loss
        return {
            "path": path,
            "soul_name": self.soul_name,
            "steps": step,
            "loss": round(loss, 4) if loss else 0.0,
            "file_size": Path(path).stat().st_size,
            "arch": "transformer",
            "n_embed": self.n_embed,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
        }

    def shutdown(self):
        """Stop background threads and save."""
        self._running = False
        self.ingestor.stop_background_polling()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        self._save_checkpoint()


# Global singleton
_learner: Optional[ContinualLearner] = None


def get_learner() -> ContinualLearner:
    """Get or create the global ContinualLearner singleton."""
    global _learner
    if _learner is None:
        _learner = ContinualLearner()
    return _learner
