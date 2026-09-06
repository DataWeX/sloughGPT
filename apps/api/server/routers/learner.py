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

from fastapi import APIRouter, Depends, Query
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, raise_error, safe_audit_log, success_response


class LearnSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_results: int = Field(default=5, ge=1, le=100)


class LearnerRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/learn", tags=["learner"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/search", self.learn_search, methods=["POST"])
        self.router.add_api_route(
            "/feed", self.learn_feed, methods=["GET"], operation_id="learn_feed_get"
        )
        self.router.add_api_route(
            "/feed", self.learn_feed, methods=["POST"], operation_id="learn_feed_post"
        )
        self.router.add_api_route("/ingest-url", self.learn_ingest_url, methods=["POST"])
        self.router.add_api_route("/knowledge", self.learn_knowledge, methods=["GET"])
        self.router.add_api_route("/ingest", self.learn_ingest, methods=["POST"])
        self.router.add_api_route("/train", self.learn_train, methods=["POST"])
        self.router.add_api_route("/deploy", self.learn_deploy, methods=["POST"])
        self.router.add_api_route("/evaluate", self.learn_evaluate, methods=["POST"])
        self.router.add_api_route("/status", self.learn_status, methods=["GET"])

    @staticmethod
    def learn_search(
        req: LearnSearchRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Search web, fetch full articles, store facts, and fine-tune."""
        try:
            import time as _time

            _t0 = _time.monotonic()
            from domains.learner import get_learner

            learner = get_learner()
            result = learner.search_and_learn(req.query, req.max_results)
            status = learner.status()
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log(
                "learner.search",
                resource=req.query,
                detail=f"elapsed={_elapsed_ms:.0f}ms facts={result.get('new_facts', 0)}",
            )
            return success_response(
                data={
                    "tokens_ingested": result.get("tokens_ingested", 0),
                    "new_facts": result.get("new_facts", 0),
                    "rejected": result.get("rejected", 0),
                    "filter_stats": result.get("filter_stats", {}),
                    "elapsed_ms": round(_elapsed_ms, 1),
                    **status,
                }
            )
        except Exception as e:
            classify_and_raise(e, source="learner.search")

    @staticmethod
    def learn_feed(
        action: str = Query(..., max_length=20),
        url: str | None = Query(None, max_length=2000),
        poll_interval: int = Query(3600, ge=60, le=86400),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Manage RSS feed subscriptions."""
        try:
            from domains.learner import get_learner

            learner = get_learner()
            if action == "subscribe":
                if not url:
                    raise_error("url required", code="E_VAL_REQUEST")
                ok = learner.subscribe_feed(url, poll_interval)
                safe_audit_log(
                    "learner.feed_subscribe", resource=url, detail=f"poll_interval={poll_interval}"
                )
                return success_response(
                    data={
                        "status": "ok" if ok else "already_subscribed",
                        "feeds": learner.list_feeds(),
                    }
                )
            elif action == "unsubscribe":
                if not url:
                    raise_error("url required", code="E_VAL_REQUEST")
                ok = learner.unsubscribe_feed(url)
                safe_audit_log("learner.feed_unsubscribe", resource=url)
                return success_response(
                    data={"status": "ok" if ok else "not_found", "feeds": learner.list_feeds()}
                )
            elif action == "list":
                return success_response(data={"feeds": learner.list_feeds()})
            raise_error("unknown action", code="E_VAL_REQUEST")
        except Exception as e:
            classify_and_raise(e, source="learner.feed")

    @staticmethod
    def learn_ingest_url(
        url: str = Query(..., min_length=1, max_length=2000),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Ingest a single URL: scrape article, store facts, fine-tune."""
        try:
            import time as _time

            _t0 = _time.monotonic()
            from domains.learner import get_learner

            learner = get_learner()
            result = learner.ingest_url(url)
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log(
                "learner.ingest_url", resource=url, detail=f"elapsed={_elapsed_ms:.0f}ms"
            )
            return success_response(data={**result, "elapsed_ms": round(_elapsed_ms, 1)})
        except Exception as e:
            classify_and_raise(e, source="learner.ingest_url")

    @staticmethod
    def learn_knowledge(
        topic: str | None = Query(None, max_length=100),
        query: str | None = Query(None, max_length=1000),
        top_k: int = Query(10, ge=1, le=100),
    ) -> dict:
        """Query learned knowledge by topic or keyword search."""
        try:
            from domains.learner import get_learner

            learner = get_learner()
            if topic:
                facts = learner.query_knowledge(topic)
            elif query:
                facts = learner.search_knowledge(query, top_k=top_k)
            else:
                facts = []
            return success_response(data={"facts": facts, "count": len(facts)})
        except Exception as e:
            classify_and_raise(e, source="learner.knowledge")

    @staticmethod
    def learn_ingest(
        text: str | None = None,
        conversations: list[list[str]] | None = None,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Ingest raw text or conversation pairs into the learner."""
        try:
            import time as _time

            _t0 = _time.monotonic()
            from domains.learner import get_learner

            learner = get_learner()
            if text:
                learner.ingest_text(text)
            if conversations:
                pairs = [(c[0], c[1]) for c in conversations if len(c) >= 2]
                learner.ingest_conversation(pairs)
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log(
                "learner.ingest",
                detail=f"elapsed={_elapsed_ms:.0f}ms text={'yes' if text else 'no'} conversations={'yes' if conversations else 'no'}",
            )
            return success_response(data={**learner.status(), "elapsed_ms": round(_elapsed_ms, 1)})
        except Exception as e:
            classify_and_raise(e, source="learner.ingest")

    @staticmethod
    def learn_train(auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Force an immediate training step on accumulated data."""
        import time as _time

        _t0 = _time.monotonic()
        from domains.learner import get_learner

        learner = get_learner()
        status = learner.train_now()
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log("learner.train", detail=f"elapsed={_elapsed_ms:.0f}ms")
        return success_response(data={**status, "elapsed_ms": round(_elapsed_ms, 1)})

    @staticmethod
    def learn_deploy(
        name: str | None = None, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Export the learner's SloTransformer as a deployable .soul file."""
        import time as _time

        _t0 = _time.monotonic()
        from domains.learner import get_learner

        learner = get_learner()
        result = learner.deploy(name=name)
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log(
            "learner.deploy", resource=name or "default", detail=f"elapsed={_elapsed_ms:.0f}ms"
        )
        return success_response(data={**result, "elapsed_ms": round(_elapsed_ms, 1)})

    @staticmethod
    def learn_evaluate(
        text: str | None = None, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Evaluate the learner on provided text or ring buffer test split."""
        import time as _time

        _t0 = _time.monotonic()
        from domains.learner import get_learner

        learner = get_learner()
        result = learner.evaluate(text=text)
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log("learner.evaluate", detail=f"elapsed={_elapsed_ms:.0f}ms")
        return success_response(data={**result, "elapsed_ms": round(_elapsed_ms, 1)})

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
