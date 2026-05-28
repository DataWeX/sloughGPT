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

router = APIRouter(prefix="/learn", tags=["learner"])


class LearnSearchRequest(BaseModel):
    query: str
    max_results: int = 5

@router.post("/search")
def learn_search(req: LearnSearchRequest):
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
    return {
        "status": "ok",
        "tokens_ingested": result.get("tokens_ingested", 0),
        "new_facts": result.get("new_facts", 0),
        "rejected": result.get("rejected", 0),
        "filter_stats": result.get("filter_stats", {}),
        **status,
    }


@router.post("/feed")
def learn_feed(
    action: str = Query(...),        # "subscribe", "unsubscribe", "list"
    url: Optional[str] = Query(None),
    poll_interval: int = 3600,
):
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
            return {"status": "error", "message": "url required"}
        ok = learner.subscribe_feed(url, poll_interval)
        return {"status": "ok" if ok else "already_subscribed", "feeds": learner.list_feeds()}
    elif action == "unsubscribe":
        if not url:
            return {"status": "error", "message": "url required"}
        ok = learner.unsubscribe_feed(url)
        return {"status": "ok" if ok else "not_found", "feeds": learner.list_feeds()}
    elif action == "list":
        return {"status": "ok", "feeds": learner.list_feeds()}
    return {"status": "error", "message": "unknown action"}


@router.post("/ingest-url")
def learn_ingest_url(url: str = Query(...)):
    """Ingest a single URL: scrape article, store facts, fine-tune.

    Args:
        url: the URL to scrape and learn from

    Returns:
        ingestion result with fact count and status
    """
    from domains.learner import get_learner
    learner = get_learner()
    result = learner.ingest_url(url)
    return {"status": "ok", **result}


@router.get("/knowledge")
def learn_knowledge(
    topic: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    top_k: int = 10,
):
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
    return {"status": "ok", "facts": facts, "count": len(facts)}


@router.post("/ingest")
def learn_ingest(
    text: Optional[str] = None,
    conversations: Optional[list[list[str]]] = None,
):
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

    return {"status": "ok", **learner.status()}


@router.post("/train")
def learn_train():
    """Force an immediate training step on accumulated data.

    Returns:
        updated learner status (current_loss, steps_completed, etc.)
    """
    from domains.learner import get_learner
    learner = get_learner()
    status = learner.train_now()
    return {"status": "ok", **status}


@router.post("/deploy")
def learn_deploy(name: Optional[str] = None):
    """Export the learner's SloTransformer as a deployable .soul file.

    Args:
        name: optional checkpoint name

    Returns:
        path, soul_name, steps, loss, file_size for loading via POST /souls/switch
    """
    from domains.learner import get_learner
    learner = get_learner()
    result = learner.deploy(name=name)
    return {"status": "ok", **result}


@router.post("/evaluate")
def learn_evaluate(text: Optional[str] = None):
    """Evaluate the learner on provided text or ring buffer test split.

    Args:
        text: optional test text. If omitted, uses 20% of the ring buffer.

    Returns:
        loss, perplexity, eval_tokens, train_steps, total_tokens_ingested
    """
    from domains.learner import get_learner
    learner = get_learner()
    result = learner.evaluate(text=text)
    return {"status": "ok", **result}


@router.get("/status")
def learn_status():
    """Return current learner + knowledge state.

    Returns:
        soul_name, total_tokens_ingested, train_steps_completed,
        current_loss, buffer_size, pending_tokens, knowledge stats,
        feed subscriptions
    """
    from domains.learner import get_learner
    learner = get_learner()
    return {"status": "ok", **learner.status()}
