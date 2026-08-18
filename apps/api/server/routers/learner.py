"""Continual learner API — live web learning, RSS, knowledge, fine-tuning.

Flow:
  POST /learn/search     — search web + fetch articles + store facts + fine-tune
  POST /learn/feed       — subscribe/unsubscribe/list RSS feeds
  POST /learn/ingest-url — ingest a single URL
  GET  /learn/knowledge  — query learned knowledge by topic or keyword
  POST /learn/ingest     — ingest raw text or conversation pairs
  POST /learn/train      — force an immediate training step
  GET  /learn/status     — learner + knowledge state summary
"""

from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

from schemas.common import success_response, raise_error


class LearnSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class LearnerRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/learn", tags=["learner"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/search", self.learn_search, methods=["POST"])
        self.router.add_api_route("/feed", self.learn_feed, methods=["POST"])
        self.router.add_api_route("/ingest-url", self.learn_ingest_url, methods=["POST"])
        self.router.add_api_route("/knowledge", self.learn_knowledge, methods=["GET"])
        self.router.add_api_route("/ingest", self.learn_ingest, methods=["POST"])
        self.router.add_api_route("/train", self.learn_train, methods=["POST"])
        self.router.add_api_route("/deploy", self.learn_deploy, methods=["POST"])
        self.router.add_api_route("/evaluate", self.learn_evaluate, methods=["POST"])
        self.router.add_api_route("/status", self.learn_status, methods=["GET"])

    @staticmethod
    def learn_search(req: LearnSearchRequest) -> dict:
        """Search web, fetch full articles, store facts, and fine-tune.

        Args:
            query: search query string
            max_results: number of articles to fetch (default 5)

        Returns:
            learner status with tokens_ingested and new_facts
        """
        from domains.learner import get_learner
        learner = get_learner()
        result = learner.search_and_learn(req.query, req.max_results)
        status = learner.status()
        return success_response(data={
            "tokens_ingested": result.get("tokens_ingested", 0),
            "new_facts": result.get("new_facts", 0),
            "rejected": result.get("rejected", 0),
            "filter_stats": result.get("filter_stats", {}),
            **status,
        })

    @staticmethod
    def learn_feed(
        action: str = Query(..., max_length=20),
        url: Optional[str] = Query(None, max_length=2000),
        poll_interval: int = Query(3600, ge=60, le=86400),
    ) -> dict:
        """Manage RSS feed subscriptions.

        Args:
            action: "subscribe" — add a feed, "unsubscribe" — remove, "list" — show all
            url: feed URL (required for subscribe/unsubscribe)
            poll_interval: seconds between polls (default 3600)

        Returns:
            feed list or operation result
        """
        from domains.learner import get_learner
        learner = get_learner()
        if action == "subscribe":
            if not url:
                raise_error("url required", code="E_VAL_REQUEST")
            ok = learner.subscribe_feed(url, poll_interval)
            return success_response(data={"status": "ok" if ok else "already_subscribed", "feeds": learner.list_feeds()})
        elif action == "unsubscribe":
            if not url:
                raise_error("url required", code="E_VAL_REQUEST")
            ok = learner.unsubscribe_feed(url)
            return success_response(data={"status": "ok" if ok else "not_found", "feeds": learner.list_feeds()})
        elif action == "list":
            return success_response(data={"feeds": learner.list_feeds()})
        raise_error("unknown action", code="E_VAL_REQUEST")

    @staticmethod
    def learn_ingest_url(url: str = Query(..., min_length=1, max_length=2000)) -> dict:
        """Ingest a single URL: scrape article, store facts, fine-tune.

        Args:
            url: the URL to scrape and learn from

        Returns:
            ingestion result with fact count and status
        """
        from domains.learner import get_learner
        learner = get_learner()
        result = learner.ingest_url(url)
        return success_response(data=result)

    @staticmethod
    def learn_knowledge(
        topic: Optional[str] = Query(None, max_length=100),
        query: Optional[str] = Query(None, max_length=1000),
        top_k: int = Query(10, ge=1, le=100),
    ) -> dict:
        """Query learned knowledge by topic or keyword search.

        Args:
            topic: exact topic name to retrieve facts for
            query: keyword search across all topics
            top_k: max results (default 10)

        Returns:
            facts matching the query
        """
        from domains.learner import get_learner
        learner = get_learner()
        if topic:
            facts = learner.query_knowledge(topic)
        elif query:
            facts = learner.search_knowledge(query, top_k=top_k)
        else:
            facts = []
        return success_response(data={"facts": facts, "count": len(facts)})

    @staticmethod
    def learn_ingest(
        text: Optional[str] = None,
        conversations: Optional[list[list[str]]] = None,
    ) -> dict:
        """Ingest raw text or conversation pairs into the learner.

        Args:
            text: raw text string to tokenize and ingest
            conversations: list of [user_message, assistant_message] pairs

        Returns:
            learner status after ingestion
        """
        from domains.learner import get_learner
        learner = get_learner()

        if text:
            learner.ingest_text(text)
        if conversations:
            pairs = [(c[0], c[1]) for c in conversations if len(c) >= 2]
            learner.ingest_conversation(pairs)

        return success_response(data=learner.status())

    @staticmethod
    def learn_train() -> dict:
        """Force an immediate training step on accumulated data.

        Returns:
            updated learner status (current_loss, steps_completed, etc.)
        """
        from domains.learner import get_learner
        learner = get_learner()
        status = learner.train_now()
        return success_response(data=status)

    @staticmethod
    def learn_deploy(name: Optional[str] = None) -> dict:
        """Export the learner's SloTransformer as a deployable .soul file.

        Args:
            name: optional checkpoint name

        Returns:
            path, soul_name, steps, loss, file_size for loading via POST /souls/switch
        """
        from domains.learner import get_learner
        learner = get_learner()
        result = learner.deploy(name=name)
        return success_response(data=result)

    @staticmethod
    def learn_evaluate(text: Optional[str] = None) -> dict:
        """Evaluate the learner on provided text or ring buffer test split.

        Args:
            text: optional test text. If omitted, uses 20% of the ring buffer.

        Returns:
            loss, perplexity, eval_tokens, train_steps, total_tokens_ingested
        """
        from domains.learner import get_learner
        learner = get_learner()
        result = learner.evaluate(text=text)
        return success_response(data=result)

    @staticmethod
    def learn_status() -> dict:
        """Return current learner + knowledge state.

        Returns:
            soul_name, total_tokens_ingested, train_steps_completed,
            current_loss, buffer_size, pending_tokens, knowledge stats,
            feed subscriptions
        """
        from domains.learner import get_learner
        learner = get_learner()
        return success_response(data=learner.status())


router = LearnerRouter().router
