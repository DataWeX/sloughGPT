"""
Mobile BFF (Backend For Frontend) router.

Aggregates multiple backend endpoints into mobile-optimized responses.
Provides paginated, trimmed payloads suitable for the React Native mobile app.
All endpoints prefixed with /mobile.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
import httpx

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mobile", tags=["mobile"])


async def _internal_get(request: Request, path: str):
    """Call an internal GET endpoint via httpx."""
    base_url = str(request.base_url).rstrip("/")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        try:
            resp = await client.get(path)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.warning(f"Internal GET {path} failed: {e}")
            return None


async def _internal_post(request: Request, path: str, body: dict = None):
    """Call an internal POST endpoint via httpx."""
    base_url = str(request.base_url).rstrip("/")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        try:
            resp = await client.post(path, json=body or {})
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.warning(f"Internal POST {path} failed: {e}")
            return None


async def _internal_patch(request: Request, path: str, body: dict = None):
    """Call an internal PATCH endpoint via httpx."""
    base_url = str(request.base_url).rstrip("/")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        try:
            resp = await client.patch(path, json=body or {})
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.warning(f"Internal PATCH {path} failed: {e}")
            return None


async def _internal_delete(request: Request, path: str):
    """Call an internal DELETE endpoint via httpx."""
    base_url = str(request.base_url).rstrip("/")
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        try:
            resp = await client.delete(path)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.warning(f"Internal DELETE {path} failed: {e}")
            return None


@router.get("/dashboard")
async def get_dashboard(request: Request):
    """
    Mobile dashboard — aggregated home screen data.

    Returns:
        status, model info, current soul, recent conversations, stats.

    Side effects:
        - Calls /health, /souls/current, /chat/sessions, /models internally.
    """
    health = await _internal_get(request, "/health") or {}
    soul = await _internal_get(request, "/souls/current") or {}
    sessions_data = await _internal_get(request, "/chat/sessions") or {}
    models = await _internal_get(request, "/models") or []

    sessions = sessions_data.get("sessions", []) if isinstance(sessions_data, dict) else []

    recent = []
    for s in sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)[:5]:
        msgs = s.get("messages", [])
        last_msg = msgs[-1].get("content", "") if msgs else ""
        recent.append({
            "id": s.get("id", ""),
            "title": s.get("title", "New Chat"),
            "last_message": last_msg[:120],
            "updated_at": s.get("updated_at", ""),
        })

    return {
        "status": health.get("status", "unknown"),
        "model": {
            "name": health.get("model_type", "None"),
            "loaded": health.get("model_loaded", False),
        },
        "soul": {
            "name": soul.get("name", "Default"),
            "description": soul.get("description", ""),
        },
        "recent_conversations": recent,
        "stats": {
            "model_count": len(models) if isinstance(models, list) else 0,
            "inference_count": health.get("inference_count", 0),
        },
    }


@router.get("/conversations")
async def list_conversations(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
):
    """
    Paginated conversation list for mobile.

    Args:
        page: Page number (1-indexed).
        per_page: Items per page (max 100).
        search: Optional search filter on title/message content.

    Returns:
        Paginated conversations with last message preview.

    Side effects:
        - Calls /chat/sessions internally.
    """
    sessions_data = await _internal_get(request, "/chat/sessions") or {}
    sessions = sessions_data.get("sessions", []) if isinstance(sessions_data, dict) else []

    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

    if search:
        search_lower = search.lower()
        sessions = [
            s for s in sessions
            if search_lower in (s.get("title", "") or "").lower()
            or any(
                search_lower in m.get("content", "").lower()
                for m in s.get("messages", [])[-3:]
            )
        ]

    total = len(sessions)
    start = (page - 1) * per_page
    end = start + per_page
    page_sessions = sessions[start:end]

    result = []
    for s in page_sessions:
        msgs = s.get("messages", [])
        last_msg = msgs[-1].get("content", "") if msgs else ""
        result.append({
            "id": s.get("id", ""),
            "title": s.get("title", "New Chat"),
            "last_message": last_msg[:200],
            "updated_at": s.get("updated_at", ""),
            "created_at": s.get("created_at", ""),
            "message_count": len(msgs),
            "starred": s.get("starred", False),
            "pinned": s.get("pinned", False),
        })

    return {
        "conversations": result,
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.get("/conversations/{session_id}")
async def get_conversation(request: Request, session_id: str):
    """
    Single conversation with full message history.

    Args:
        session_id: The session identifier.

    Returns:
        Session ID and full message list.

    Side effects:
        - Calls /session/{id}/messages internally.
    """
    data = await _internal_get(request, f"/session/{session_id}/messages") or {}
    return {
        "id": session_id,
        "messages": data.get("messages", []),
        "created_at": data.get("created_at", ""),
    }


@router.get("/models")
async def get_models(request: Request):
    """
    Model catalog with souls and checkpoints for mobile.

    Returns:
        Current pipeline state, available models, souls, and checkpoints.

    Side effects:
        - Calls /models, /souls, /souls/current, /auto-train/checkpoints, /health.
    """
    models = await _internal_get(request, "/models") or []
    souls = await _internal_get(request, "/souls") or []
    current_soul = await _internal_get(request, "/souls/current") or {}
    checkpoints = await _internal_get(request, "/auto-train/checkpoints") or []
    health = await _internal_get(request, "/health") or {}

    model_list = []
    if isinstance(models, list):
        for m in models:
            model_list.append({
                "id": m.get("model_id", m.get("id", "")),
                "name": m.get("name", m.get("model_id", "")),
                "loaded": m.get("loaded", m.get("status") == "loaded"),
                "size_gb": m.get("size_gb", 0),
                "source": m.get("source", "local"),
            })

    soul_list = []
    if isinstance(souls, list):
        for s in souls:
            soul_list.append({
                "name": s.get("name", ""),
                "description": s.get("description", ""),
                "traits": s.get("traits", []),
            })

    cp_list = []
    if isinstance(checkpoints, list):
        for cp in checkpoints:
            cp_list.append({
                "name": cp.get("name", ""),
                "soul": cp.get("soul", ""),
                "loss": cp.get("loss"),
                "steps": cp.get("steps"),
            })

    return {
        "current": {
            "model_id": health.get("model_type", ""),
            "soul": current_soul.get("name", ""),
            "checkpoint": None,
        },
        "models": model_list,
        "souls": soul_list,
        "checkpoints": cp_list,
    }


class SwitchRequest(BaseModel):
    """Request body for model/soul switching."""
    model_id: Optional[str] = None
    soul_name: Optional[str] = None
    checkpoint_name: Optional[str] = None


@router.post("/models/switch")
async def switch_model(request: Request, body: SwitchRequest):
    """
    Switch model and/or soul in one call.

    Args:
        body: model_id, soul_name, checkpoint_name (all optional).

    Returns:
        Updated status after switch.

    Side effects:
        - Calls /models/load and/or /souls/switch internally.
    """
    model_result = None
    soul_result = None

    if body.model_id:
        model_result = await _internal_post(request, "/models/load", {
            "model_id": body.model_id,
        })

    if body.soul_name:
        soul_body = {"soul": body.soul_name}
        if body.checkpoint_name:
            soul_body["checkpoint_name"] = body.checkpoint_name
        soul_result = await _internal_post(request, "/souls/switch", soul_body)

    health = await _internal_get(request, "/health") or {}

    return {
        "status": "ok",
        "model": health.get("model_type", ""),
        "soul": body.soul_name or "",
        "checkpoint": body.checkpoint_name,
    }


@router.get("/health")
async def get_health(request: Request):
    """
    System health summary for mobile.

    Returns:
        API status, model info, uptime, CPU, memory, disk, inference count.

    Side effects:
        - Calls /health/detailed and /system/metrics internally.
    """
    detailed = await _internal_get(request, "/health/detailed") or {}
    metrics = await _internal_get(request, "/system/metrics") or {}

    system_info = detailed.get("system", {})

    return {
        "status": detailed.get("status", "unknown"),
        "model": {
            "name": detailed.get("model_type", ""),
            "loaded": detailed.get("model_loaded", False),
            "type": detailed.get("model_type", ""),
        },
        "uptime_seconds": detailed.get("uptime_seconds", 0),
        "cpu_percent": system_info.get("cpu_percent", metrics.get("cpu_percent", 0)),
        "memory_percent": system_info.get("memory_percent", metrics.get("memory_percent", 0)),
        "memory_available_gb": round(
            system_info.get("memory_available_mb", 0) / 1024, 2
        ),
        "disk_used_gb": round(metrics.get("disk_used_bytes", 0) / (1024 ** 3), 2),
        "disk_free_gb": round(metrics.get("disk_free_bytes", 0) / (1024 ** 3), 2),
        "inference_count": detailed.get("inference", {}).get(
            "inference_count", detailed.get("inference_count", 0)
        ),
    }


@router.get("/knowledge")
async def list_knowledge(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    topic: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
):
    """
    Paginated knowledge items for mobile.

    Args:
        page: Page number (1-indexed).
        per_page: Items per page (max 100).
        topic: Optional topic filter.
        search: Optional search query.

    Returns:
        Paginated knowledge items.

    Side effects:
        - Calls /knowledge or /knowledge/search internally.
    """
    if search:
        data = await _internal_get(
            request, f"/knowledge/search?query={search}"
        ) or {}
        items = data.get("results", [])
    else:
        params = f"limit={per_page * 5}&offset=0"
        if topic:
            params += f"&topic={topic}"
        items = await _internal_get(request, f"/knowledge?{params}") or []

    if not isinstance(items, list):
        items = []

    if topic and not search:
        items = [i for i in items if i.get("topic") == topic]

    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = items[start:end]

    return {
        "items": [
            {
                "id": i.get("id", ""),
                "content": i.get("content", ""),
                "topic": i.get("topic", ""),
                "importance": i.get("importance", 0.5),
                "source": i.get("source", "manual"),
            }
            for i in page_items
        ],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


class KnowledgeCreateRequest(BaseModel):
    """Request body for creating a knowledge item."""
    content: str
    topic: Optional[str] = None


@router.post("/knowledge")
async def create_knowledge(request: Request, body: KnowledgeCreateRequest):
    """
    Add a knowledge item.

    Args:
        body: content and optional topic.

    Returns:
        Created knowledge item.

    Side effects:
        - Calls POST /knowledge internally.
    """
    result = await _internal_post(request, "/knowledge", {
        "content": body.content,
        "topic": body.topic,
    })
    return result or {"status": "error", "message": "Failed to create"}


class KnowledgeUpdateRequest(BaseModel):
    """Request body for updating a knowledge item."""
    content: Optional[str] = None
    topic: Optional[str] = None
    importance: Optional[float] = None


@router.patch("/knowledge/{item_id}")
async def update_knowledge(request: Request, item_id: str, body: KnowledgeUpdateRequest):
    """
    Update a knowledge item.

    Args:
        item_id: Knowledge item ID.
        body: Fields to update.

    Returns:
        Updated knowledge item.

    Side effects:
        - Calls PATCH /knowledge/{id} internally.
    """
    update_body = {}
    if body.content is not None:
        update_body["content"] = body.content
    if body.topic is not None:
        update_body["topic"] = body.topic
    if body.importance is not None:
        update_body["importance"] = body.importance

    result = await _internal_patch(request, f"/knowledge/{item_id}", update_body)
    return result or {"status": "error", "message": "Failed to update"}


@router.delete("/knowledge/{item_id}")
async def delete_knowledge(request: Request, item_id: str):
    """
    Delete a knowledge item.

    Args:
        item_id: Knowledge item ID.

    Returns:
        Deletion status.

    Side effects:
        - Calls DELETE /knowledge/{id} internally.
    """
    result = await _internal_delete(request, f"/knowledge/{item_id}")
    return {"status": "deleted", "id": item_id}
