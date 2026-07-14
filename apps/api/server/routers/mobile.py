"""
Mobile BFF (Backend For Frontend) router.

Aggregates multiple backend endpoints into mobile-optimized responses.
Provides paginated, trimmed payloads suitable for the React Native mobile app.
All endpoints prefixed with /mobile.
"""

import asyncio
import json
import logging
import subprocess
import time as _time
from typing import Optional, List
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx

from schemas.common import success_response, error_response

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


# ── Offline Sync ─────────────────────────────────────────────────────────────

class PendingMessage(BaseModel):
    """A single pending message from the offline queue."""
    id: str
    session_id: str
    content: str
    timestamp: int
    retry_count: int = 0


class SyncRequest(BaseModel):
    """Offline sync payload from mobile."""
    pending_messages: List[PendingMessage] = []
    last_sync_timestamp: Optional[int] = None


class SyncResult(BaseModel):
    """Result of syncing a single pending message."""
    id: str
    status: str  # "sent" | "error"
    assistant_message: Optional[dict] = None
    error: Optional[str] = None


@router.post("/sync")
async def sync_offline(request: Request, body: SyncRequest):
    """
    Sync offline messages when mobile reconnects.

    Processes queued messages from the offline cache, sends them to the
    chat endpoint, and returns results for each.

    Args:
        body: List of pending messages and optional last sync timestamp.

    Returns:
        List of sync results (one per pending message) + updated sessions.

    Side effects:
        - Calls POST /chat for each pending message.
        - Calls /chat/sessions to refresh session list.
    """
    results: List[SyncResult] = []

    for msg in body.pending_messages:
        try:
            chat_result = await _internal_post(
                request,
                "/chat",
                {
                    "messages": [{"role": "user", "content": msg.content}],
                    "session_id": msg.session_id,
                },
            )

            if chat_result and chat_result.get("message"):
                results.append(SyncResult(
                    id=msg.id,
                    status="sent",
                    assistant_message={
                        "role": "assistant",
                        "content": chat_result["message"],
                        "timestamp": chat_result.get("timestamp", 0),
                    },
                ))
            else:
                results.append(SyncResult(
                    id=msg.id,
                    status="error",
                    error="No response from server",
                ))
        except Exception as e:
            results.append(SyncResult(
                id=msg.id,
                status="error",
                error=str(e),
            ))

    # Return updated session list — internal /chat/sessions now returns StandardResponse
    sessions_data = await _internal_get(request, "/chat/sessions") or {}
    if isinstance(sessions_data, dict) and "data" in sessions_data:
        sessions = sessions_data["data"] if isinstance(sessions_data["data"], list) else []
    elif isinstance(sessions_data, dict) and "sessions" in sessions_data:
        sessions = sessions_data["sessions"]
    else:
        sessions = []

    return success_response(data={
        "results": [r.model_dump() for r in results],
        "synced_count": sum(1 for r in results if r.status == "sent"),
        "failed_count": sum(1 for r in results if r.status == "error"),
        "sessions": sessions[:20],
    })


@router.get("/sync/status")
async def sync_status(request: Request):
    """
    Check server connectivity and last sync state.

    Returns:
        Server reachable status, model state, and current timestamp.
    """
    health = await _internal_get(request, "/health") or {}

    return success_response(data={
        "reachable": health.get("status") == "healthy",
        "model_loaded": health.get("model_loaded", False),
        "server_time": int(__import__("time").time() * 1000),
        "inference_count": health.get("inference_count", 0),
    })


# ── Push Notifications ───────────────────────────────────────────────────────

class DeviceRegistrationRequest(BaseModel):
    """Request body for registering a push notification device."""
    token: str
    platform: str  # "ios" | "android" | "web"
    user_id: str = "default"
    topics: Optional[List[str]] = None


class NotificationSendRequest(BaseModel):
    """Request body for sending a push notification."""
    title: str
    body: str
    topic: Optional[str] = None
    data: Optional[dict] = None
    tokens: Optional[List[str]] = None
    badge: Optional[int] = None


@router.post("/notifications/register")
async def register_device(body: DeviceRegistrationRequest):
    """
    Register a mobile device for push notifications.

    Args:
        body: Device token, platform, user_id, and topic subscriptions.

    Returns:
        Registration status.
    """
    from domains.mobile.notifications import get_notification_service

    svc = get_notification_service()
    result = svc.register_device(
        token=body.token,
        platform=body.platform,
        user_id=body.user_id,
        topics=body.topics,
    )
    return result


@router.post("/notifications/unregister")
async def unregister_device(body: dict):
    """
    Unregister a device from push notifications.

    Args:
        body: {"token": "expo-push-token..."}.

    Returns:
        Unregistration status.
    """
    from domains.mobile.notifications import get_notification_service

    svc = get_notification_service()
    token = body.get("token", "")
    removed = svc.unregister_device(token)
    return {"status": "removed" if removed else "not_found"}


@router.get("/notifications/devices")
async def list_devices(topic: Optional[str] = Query(None)):
    """
    List registered devices.

    Args:
        topic: Optional topic filter.

    Returns:
        List of registered devices.
    """
    from domains.mobile.notifications import get_notification_service

    svc = get_notification_service()
    return {"devices": svc.get_devices(topic=topic)}


@router.post("/notifications/send")
async def send_notification(body: NotificationSendRequest):
    """
    Send a push notification to registered devices.

    Args:
        body: Title, body, topic filter, data payload, optional token list.

    Returns:
        Send result with recipient count.
    """
    from domains.mobile.notifications import get_notification_service, NotificationPayload

    svc = get_notification_service()
    payload = NotificationPayload(
        title=body.title,
        body=body.body,
        data=body.data or {},
        badge=body.badge,
        topic=body.topic,
    )
    result = svc.send_notification(
        payload=payload,
        tokens=body.tokens,
        topic=body.topic,
    )
    return result


@router.get("/notifications/history")
async def notification_history(limit: int = Query(50, ge=1, le=200)):
    """
    Get recent notification history.

    Args:
        limit: Max entries to return.

    Returns:
        Recent notification records.
    """
    from domains.mobile.notifications import get_notification_service

    svc = get_notification_service()
    return {"history": svc.get_history(limit=limit)}


@router.post("/notifications/cleanup")
async def cleanup_devices():
    """
    Remove devices inactive for 30+ days.

    Returns:
        Count of removed devices.
    """
    from domains.mobile.notifications import get_notification_service

    svc = get_notification_service()
    removed = svc.cleanup_stale()
    return {"removed": removed}


@router.post("/notify/training-complete")
async def notify_training_complete(request: Request):
    """
    Send a training-complete notification to all registered devices.

    Side effects:
        - Reads training status from internal endpoint.
        - Sends push notification to all devices subscribed to 'training' topic.
    """
    from domains.mobile.notifications import get_notification_service, NotificationPayload

    svc = get_notification_service()
    training = await _internal_get(request, "/training/status") or {}

    status = training.get("status", "unknown")
    loss = training.get("final_loss")

    payload = NotificationPayload(
        title="Training Complete",
        body=f"Model training finished. Final loss: {loss:.4f}" if loss else f"Training {status}",
        data={"type": "training_complete", "status": status},
        topic="training",
    )

    result = svc.send_notification(payload=payload, topic="training")
    return result


# ── On-Device Training ──────────────────────────────────────────────────────

class TrainingPair(BaseModel):
    """A single (user, assistant) conversation pair for on-device training."""
    id: str
    user_msg: str
    assistant_msg: str
    quality: float = 0.0
    timestamp: int = 0
    session_id: str = ""


class MobileTrainRequest(BaseModel):
    """Batch of training pairs from the mobile app."""
    pairs: List[TrainingPair]
    checkpoint: str


class MobileTrainResult(BaseModel):
    """Result of on-device training triggered by mobile."""
    success: bool
    checkpoint_name: str = ""
    loss: float = 0.0
    steps: int = 0
    elapsed_ms: int = 0


@router.post("/train")
async def mobile_train(body: MobileTrainRequest):
    """
    Train the SloNet model on conversation pairs from the mobile app.

    Receives (user_msg, assistant_msg) pairs collected on-device,
    stores them in MogDB, runs a short fine-tune via subprocess
    (using the venv Python with torch), and returns the new checkpoint.

    Args:
        body: Batch of training pairs + base checkpoint name.

    Returns:
        MobileTrainResult with checkpoint name, loss, steps, timing.

    Side effects:
        - Stores pairs in MogDB training data collection.
        - Spawns subprocess for HF fine-tuning.
        - Saves new checkpoint to models/auto-training/.
    """
    import time
    from pathlib import Path
    from fastapi import HTTPException

    if len(body.pairs) < 5:
        raise HTTPException(400, "Need at least 5 training pairs")

    t0 = int(time.time() * 1000)

    # Store pairs in MogDB
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    pair_ids = store.add_batch([
        {
            "user_msg": p.user_msg,
            "assistant_msg": p.assistant_msg,
            "session_id": p.session_id,
            "quality": p.quality,
        }
        for p in body.pairs
    ])

    # Write training text file
    repo_root = Path(__file__).resolve().parents[4]
    train_dir = repo_root / "data" / "mobile_training"
    train_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    text_file = train_dir / f"mobile_{ts}.txt"
    with open(text_file, "w") as f:
        for pair in body.pairs:
            f.write(f"User: {pair.user_msg}\nAssistant: {pair.assistant_msg}\n\n")

    checkpoint_name = f"mobile_{ts}"
    output_dir = repo_root / "models" / "auto-training" / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find venv Python with torch
    venv_python = repo_root / ".venv" / "bin" / "python3"
    train_script = repo_root / "scripts" / "hf_train.py"

    if not venv_python.exists():
        raise HTTPException(500, "Training environment not found (.venv missing)")

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            [
                str(venv_python),
                str(train_script),
                "--data", str(text_file),
                "--output", str(output_dir),
                "--model", "gpt2",
                "--epochs", "1",
                "--batch-size", "2",
                "--lr", "5e-5",
                "--max-seq-length", "256",
                "--use-lora",
                "--lora-rank", "8",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if proc.returncode != 0:
            logger.error("Training subprocess failed: %s", proc.stderr[-500:])
            raise HTTPException(500, f"Training failed: {proc.stderr[-200:]}")

        result = json.loads(proc.stdout.strip().split("\n")[-1])

        if not result.get("success"):
            raise HTTPException(500, f"Training failed: {result.get('error', 'unknown')}")

        # Mark pairs as used for training
        store.mark_used(pair_ids)
        store.mark_synced(pair_ids)

        elapsed = int(time.time() * 1000) - t0

        return MobileTrainResult(
            success=True,
            checkpoint_name=checkpoint_name,
            loss=result.get("loss", 0.0),
            steps=result.get("steps", 0),
            elapsed_ms=elapsed,
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Training timed out (300s limit)")
    except json.JSONDecodeError as e:
        logger.error("Failed to parse training output: %s", e)
        raise HTTPException(500, "Training produced invalid output")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Mobile training failed: %s", e)
        raise HTTPException(500, f"Training failed: {e}")


@router.get("/train/stats")
async def get_training_stats():
    """
    Get training data statistics.

    Returns:
        Total pairs, pending, synced, used counts, and quality breakdown.

    Side effects:
        - Reads from MogDB training data collection.
    """
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    stats = store.stats()
    quality_counts = store.quality_breakdown()

    return {
        "total": stats["total"],
        "pending": stats["pending"],
        "synced": stats["synced"],
        "used": stats["used"],
        "by_quality": quality_counts,
    }


@router.get("/train/pending")
async def get_pending_pairs(limit: int = Query(50, ge=1, le=500)):
    """
    Get pending (unsynced) training pairs.

    Args:
        limit: Max pairs to return (default 50, max 500).

    Returns:
        List of unsynced training pair documents.

    Side effects:
        - Reads from MogDB training data collection.
    """
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    pairs = store.get_pending_pairs(limit=limit)
    return {
        "pairs": [
            {
                "id": p.get("_id", ""),
                "user_msg": p.get("user_msg", ""),
                "assistant_msg": p.get("assistant_msg", ""),
                "quality": p.get("quality", 0),
                "session_id": p.get("session_id", ""),
                "timestamp": p.get("timestamp", 0),
            }
            for p in pairs
        ],
        "count": len(pairs),
    }


@router.get("/train/pairs")
async def list_training_pairs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    min_quality: Optional[float] = Query(None),
    session_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
):
    """
    List training pairs with optional filters.

    Args:
        limit: Max pairs to return (default 50, max 500).
        offset: Skip first N pairs (for pagination).
        min_quality: Filter to quality >= this value.
        session_id: Filter to specific session.
        search: Search in user_msg and assistant_msg content.

    Returns:
        List of training pair documents, newest first.

    Side effects:
        - Reads from MogDB training data collection.
    """
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    pairs = store.list_pairs(
        limit=limit,
        offset=offset,
        min_quality=min_quality,
        session_id=session_id,
        search=search,
    )
    total = store.count()
    return {
        "pairs": [
            {
                "id": p.get("_id", ""),
                "user_msg": p.get("user_msg", ""),
                "assistant_msg": p.get("assistant_msg", ""),
                "quality": p.get("quality", 0),
                "session_id": p.get("session_id", ""),
                "timestamp": p.get("timestamp", 0),
            }
            for p in pairs
        ],
        "total": total,
        "count": len(pairs),
        "offset": offset,
    }


@router.get("/train/export")
async def export_training_pairs(
    min_quality: Optional[float] = Query(None),
    session_id: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
):
    """
    Export training pairs as JSONL for download.

    Args:
        min_quality: Filter to quality >= this value.
        session_id: Filter to specific session.
        limit: Max pairs to export (default 500, max 5000).

    Returns:
        StreamingResponse with JSONL content.

    Side effects:
        - Reads from MogDB training data collection.
    """
    from fastapi.responses import StreamingResponse
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    pairs = store.list_pairs(limit=limit, min_quality=min_quality, session_id=session_id)

    def generate():
        for p in pairs:
            yield json.dumps({
                "user_msg": p.get("user_msg", ""),
                "assistant_msg": p.get("assistant_msg", ""),
                "quality": p.get("quality", 0),
                "session_id": p.get("session_id", ""),
            }) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=training_pairs.jsonl"},
    )


@router.get("/train/session/{session_id}")
async def get_session_pairs(session_id: str):
    """
    Get all training pairs from a specific session.

    Args:
        session_id: Chat session identifier.

    Returns:
        List of training pairs for that session.

    Side effects:
        - Reads from MogDB training data collection.
    """
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    pairs = store.get_pairs_by_session(session_id)
    return {
        "session_id": session_id,
        "pairs": [
            {
                "id": p.get("_id", ""),
                "user_msg": p.get("user_msg", ""),
                "assistant_msg": p.get("assistant_msg", ""),
                "quality": p.get("quality", 0),
                "timestamp": p.get("timestamp", 0),
            }
            for p in pairs
        ],
        "count": len(pairs),
    }


class QualityUpdateRequest(BaseModel):
    """Request body for updating pair quality."""
    quality: float


@router.patch("/train/pair/{pair_id}")
async def update_pair_quality(pair_id: str, body: QualityUpdateRequest):
    """
    Update quality signal on a training pair.

    Args:
        pair_id: Training pair document ID.
        body: New quality value.

    Returns:
        Update status.

    Side effects:
        - Updates quality field in MogDB training data collection.
    """
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    updated = store.update_quality(pair_id, body.quality)
    if not updated:
        from fastapi import HTTPException
        raise HTTPException(404, "Pair not found")
    return {"status": "updated", "pair_id": pair_id, "quality": body.quality}


@router.delete("/train/pair/{pair_id}")
async def delete_pair(pair_id: str):
    """
    Delete a single training pair.

    Args:
        pair_id: Training pair document ID.

    Returns:
        Deletion status.

    Side effects:
        - Deletes pair from MogDB training data collection.
    """
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    deleted = store.delete_pair(pair_id)
    if not deleted:
        from fastapi import HTTPException
        raise HTTPException(404, "Pair not found")
    return {"status": "deleted", "pair_id": pair_id}


@router.delete("/train/synced")
async def delete_synced_pairs():
    """
    Delete all synced training pairs (already used for training).

    Returns:
        Count of deleted pairs.

    Side effects:
        - Deletes all synced pairs from MogDB training data collection.
    """
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    count = store.delete_synced()
    return {"status": "deleted", "count": count}


@router.delete("/train/pairs/bulk")
async def delete_pairs_bulk(ids: list[str] = Query(..., description="Pair IDs to delete")):
    """
    Delete multiple training pairs by ID.

    Args:
        ids: List of pair IDs to delete.

    Returns:
        Count of deleted pairs.

    Side effects:
        - Deletes matching pairs from MogDB training data collection.
    """
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    count = 0
    for pair_id in ids:
        if store.delete_pair(pair_id):
            count += 1
    return {"status": "deleted", "count": count}


@router.post("/train/compact")
async def compact_training_store():
    """
    Compact the training data store (reclaim space from deleted records).

    Returns:
        Remaining document count after compaction.

    Side effects:
        - Compacts MogDB training data collection journals.
    """
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    count = store.compact()
    return {"status": "compacted", "count": count}


class FromSessionsRequest(BaseModel):
    """Optional params for training from server sessions."""
    limit: int = Field(default=50, ge=5, le=500)
    min_length: int = Field(default=5, ge=1)
    model: Optional[str] = None
    session_ids: Optional[list[str]] = None


@router.post("/train/from-sessions")
async def train_from_sessions(body: FromSessionsRequest = FromSessionsRequest(), request: Request = None):
    """
    Train the model from server-side inference logs (SSE streaming).

    Extracts (user_msg, assistant_msg) pairs from session JSON files,
    writes a training text file, and spawns a subprocess fine-tune.
    Streams progress as SSE events in real-time.

    SSE phases: GENERATE_DATA → TRAIN → COMPLETE | ERROR

    Args:
        body: limit (max pairs), min_length (filter tiny messages), model (optional filter).

    Yields:
        SSE events with training progress (step, loss, epoch, progress_pct).

    Side effects:
        - Reads session files from data/chat_sessions/.
        - Writes training text to data/mobile_training/.
        - Spawns HF fine-tune subprocess (non-blocking Popen).
    """
    from pathlib import Path
    from domains.api.sse_envelope import sse_event, sse_error, sse_complete

    t0 = _time.time()

    # Extract pairs from server logs
    from domains.training.pair_extractor import (
        extract_pairs_from_sessions,
        extract_pairs_from_logs,
        write_training_text,
    )

    pairs = extract_pairs_from_sessions(
        limit=body.limit, min_length=body.min_length, session_ids=body.session_ids
    )
    if len(pairs) < 5:
        pairs = extract_pairs_from_logs(
            limit=body.limit, min_length=body.min_length, model=body.model
        )

    if len(pairs) < 5:
        yield sse_error("training", "GENERATE_DATA",
                         f"Need at least 5 training pairs, found {len(pairs)} in server logs")
        return

    # Emit GENERATE_DATA phase
    yield sse_event(
        stream="training", phase="GENERATE_DATA", status="working",
        data={"pairs": len(pairs)},
        message=f"Extracted {len(pairs)} training pairs",
    )

    # Write training text file
    text_file = write_training_text(pairs)

    # Store pairs in MogDB
    from domains.training.mobile_training_store import get_training_store

    store = get_training_store()
    store.add_batch([
        {
            "user_msg": p["user_msg"],
            "assistant_msg": p["assistant_msg"],
            "session_id": p.get("session_id", ""),
            "quality": 0,
        }
        for p in pairs
    ])

    # Prepare subprocess
    repo_root = Path(__file__).resolve().parents[4]
    ts = int(_time.time())
    checkpoint_name = f"sessions_{ts}"
    output_dir = repo_root / "models" / "auto-training" / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)

    venv_python = repo_root / ".venv" / "bin" / "python3"
    train_script = repo_root / "scripts" / "hf_train.py"

    if not venv_python.exists():
        yield sse_error("training", "TRAIN", "Training environment not found (.venv missing)")
        return

    # Launch subprocess with streaming stdout
    cmd = [
        str(venv_python),
        str(train_script),
        "--stream",
        "--data", str(text_file),
        "--output", str(output_dir),
        "--model", "gpt2",
        "--epochs", "1",
        "--batch-size", "2",
        "--lr", "5e-5",
        "--max-seq-length", "256",
        "--use-lora",
        "--lora-rank", "8",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        logger.error("Failed to start training subprocess: %s", e)
        yield sse_error("training", "TRAIN", f"Failed to start training: {e}")
        return

    # Stream stdout lines as SSE events
    deadline = _time.time() + 600  # 10-minute hard timeout
    disconnect_check_interval = 5.0  # Check disconnect every 5s while waiting for lines
    result = None

    try:
        while True:
            if _time.time() > deadline:
                proc.kill()
                yield sse_error("training", "TRAIN", "Training timed out (10 min)")
                return

            # Check client disconnect periodically
            if request and await request.is_disconnected():
                proc.kill()
                logger.info("Client disconnected from training stream")
                return

            # Read one line (with timeout via select-like approach)
            line = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, proc.stdout.readline),
                timeout=disconnect_check_interval,
            )

            if not line:
                break  # stdout closed — process finished

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            phase = event.get("phase", "TRAIN")
            status = event.get("status", "working")

            # Check if this is the final result
            if event.get("success") is not None or phase == "COMPLETE":
                result = event
                break

            # Forward progress event as SSE
            elapsed_ms = round((_time.time() - t0) * 1000)
            yield sse_event(
                stream="training", phase=phase, status="working",
                data={
                    "step": event.get("step"),
                    "loss": event.get("loss"),
                    "epoch": event.get("epoch"),
                    "progress_pct": event.get("progress_pct"),
                    "total_steps": event.get("total_steps"),
                },
                meta={"elapsed_ms": elapsed_ms, "checkpoint": checkpoint_name},
                message=event.get("message", ""),
            )

        # Wait for process to finish
        proc.wait(timeout=30)

        if proc.returncode != 0:
            stderr_tail = proc.stderr.read()[-500:] if proc.stderr else ""
            logger.error("Training subprocess failed (rc=%d): %s", proc.returncode, stderr_tail)
            yield sse_error("training", "TRAIN", f"Training failed (exit code {proc.returncode})")
            return

        # Parse final result if not already captured from stdout
        if result is None:
            # Try reading any remaining stderr for errors
            stderr_tail = proc.stderr.read()[-500:] if proc.stderr else ""
            yield sse_error("training", "TRAIN",
                            f"Training produced no result output. stderr: {stderr_tail[:200]}")
            return

        elapsed_ms = round((_time.time() - t0) * 1000)

        yield sse_complete(
            stream="training", phase="COMPLETE",
            data={
                "checkpoint_name": checkpoint_name,
                "loss": result.get("loss", 0.0),
                "steps": result.get("steps", 0),
                "elapsed_ms": elapsed_ms,
                "model_path": result.get("model_path", ""),
            },
            meta={"elapsed_ms": elapsed_ms},
            message=f"Training complete — loss={result.get('loss', 0):.4f} steps={result.get('steps', 0)}",
        )

    except asyncio.TimeoutError:
        # readline timed out — send heartbeat check and continue the loop
        pass
    except Exception as e:
        logger.error("Session training failed: %s", e)
        yield sse_error("training", "TRAIN", f"Training failed: {e}")
    finally:
        if proc.poll() is None:
            proc.kill()


@router.get("/train/auto-status")
async def get_auto_train_status():
    """
    Get auto-trainer status and configuration.

    Returns:
        Enabled state, threshold, pending count, last train info.

    Side effects:
        - None (reads from AutoTrainer singleton).
    """
    from domains.training.auto_trainer import get_auto_trainer

    trainer = get_auto_trainer()
    return trainer.status()


@router.patch("/train/auto-config")
async def update_auto_train_config(
    threshold: Optional[int] = Query(None, ge=1, le=100),
    interval_s: Optional[int] = Query(None, ge=30, le=3600),
):
    """
    Update auto-trainer configuration at runtime.

    Args:
        threshold: New conversation threshold (1-100).
        interval_s: New minimum interval between trains in seconds (30-3600).

    Returns:
        Updated auto-trainer status.

    Side effects:
        - Modifies AutoTrainer singleton attributes.
    """
    from domains.training.auto_trainer import get_auto_trainer

    trainer = get_auto_trainer()
    if threshold is not None:
        trainer.threshold = threshold
    if interval_s is not None:
        trainer.interval_s = interval_s
    return trainer.status()
