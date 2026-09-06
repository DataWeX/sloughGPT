"""
Mobile BFF (Backend For Frontend) router.

Aggregates multiple backend endpoints into mobile-optimized responses.
Provides paginated, trimmed payloads suitable for the React Native mobile app.
All endpoints prefixed with /mobile.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time as _time
from collections.abc import AsyncGenerator

from domains.infrastructure.errors import AppError
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, raise_error, safe_audit_log, success_response

logger = logging.getLogger(__name__)


class UnregisterDeviceRequest(BaseModel):
    """Schema for unregistering a push notification device."""

    token: str = Field(..., max_length=512)


class SwitchRequest(BaseModel):
    """Request body for model/soul switching."""

    model_id: str | None = None
    soul_name: str | None = None
    checkpoint_name: str | None = None


class KnowledgeCreateRequest(BaseModel):
    """Request body for creating a knowledge item."""

    content: str
    topic: str | None = None


class KnowledgeUpdateRequest(BaseModel):
    """Request body for updating a knowledge item."""

    content: str | None = None
    topic: str | None = None
    importance: float | None = None


class PendingMessage(BaseModel):
    """A single pending message from the offline queue."""

    id: str
    session_id: str
    content: str
    timestamp: int
    retry_count: int = 0


class SyncRequest(BaseModel):
    """Offline sync payload from mobile."""

    pending_messages: list[PendingMessage] = []
    last_sync_timestamp: int | None = None


class SyncResult(BaseModel):
    """Result of syncing a single pending message."""

    id: str
    status: str  # "sent" | "error"
    assistant_message: dict | None = None
    error: str | None = None


class DeviceRegistrationRequest(BaseModel):
    """Request body for registering a push notification device."""

    token: str
    platform: str  # "ios" | "android" | "web"
    user_id: str = "default"
    topics: list[str] | None = None


class NotificationSendRequest(BaseModel):
    """Request body for sending a push notification."""

    title: str
    body: str
    topic: str | None = None
    data: dict | None = None
    tokens: list[str] | None = None
    badge: int | None = None


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

    pairs: list[TrainingPair]
    checkpoint: str


class MobileTrainResult(BaseModel):
    """Result of on-device training triggered by mobile."""

    success: bool
    checkpoint_name: str = ""
    loss: float = 0.0
    steps: int = 0
    elapsed_ms: int = 0


class QualityUpdateRequest(BaseModel):
    """Request body for updating pair quality."""

    quality: float


class FromSessionsRequest(BaseModel):
    """Optional params for training from server sessions."""

    limit: int = Field(default=50, ge=5, le=500)
    min_length: int = Field(default=5, ge=1)
    model: str | None = None
    session_ids: list[str] | None = None


class MobileRouter:
    """Mobile BFF (Backend For Frontend) router."""

    def __init__(self):
        self.router = APIRouter(prefix="/mobile", tags=["mobile"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="/dashboard", endpoint=self.get_dashboard, methods=["GET"])
        self.router.add_api_route(
            path="/conversations", endpoint=self.list_conversations, methods=["GET"]
        )
        self.router.add_api_route(
            path="/conversations/{session_id}", endpoint=self.get_conversation, methods=["GET"]
        )
        self.router.add_api_route(path="/models", endpoint=self.get_models, methods=["GET"])
        self.router.add_api_route(
            path="/models/switch", endpoint=self.switch_model, methods=["POST"]
        )
        self.router.add_api_route(path="/health", endpoint=self.get_health, methods=["GET"])
        self.router.add_api_route(path="/knowledge", endpoint=self.list_knowledge, methods=["GET"])
        self.router.add_api_route(
            path="/knowledge", endpoint=self.create_knowledge, methods=["POST"]
        )
        self.router.add_api_route(
            path="/knowledge/{item_id}", endpoint=self.update_knowledge, methods=["PATCH"]
        )
        self.router.add_api_route(
            path="/knowledge/{item_id}", endpoint=self.delete_knowledge, methods=["DELETE"]
        )
        self.router.add_api_route(path="/sync", endpoint=self.sync_offline, methods=["POST"])
        self.router.add_api_route(path="/sync/status", endpoint=self.sync_status, methods=["GET"])
        self.router.add_api_route(
            path="/notifications/register", endpoint=self.register_device, methods=["POST"]
        )
        self.router.add_api_route(
            path="/notifications/unregister", endpoint=self.unregister_device, methods=["POST"]
        )
        self.router.add_api_route(
            path="/notifications/devices", endpoint=self.list_devices, methods=["GET"]
        )
        self.router.add_api_route(
            path="/notifications/send", endpoint=self.send_notification, methods=["POST"]
        )
        self.router.add_api_route(
            path="/notifications/history", endpoint=self.notification_history, methods=["GET"]
        )
        self.router.add_api_route(
            path="/notifications/cleanup", endpoint=self.cleanup_devices, methods=["POST"]
        )
        self.router.add_api_route(
            path="/notify/training-complete",
            endpoint=self.notify_training_complete,
            methods=["POST"],
        )
        self.router.add_api_route(path="/train", endpoint=self.mobile_train, methods=["POST"])
        self.router.add_api_route(
            path="/train/stats", endpoint=self.get_training_stats, methods=["GET"]
        )
        self.router.add_api_route(
            path="/train/pending", endpoint=self.get_pending_pairs, methods=["GET"]
        )
        self.router.add_api_route(
            path="/train/pairs", endpoint=self.list_training_pairs, methods=["GET"]
        )
        self.router.add_api_route(
            path="/train/export",
            endpoint=self.export_training_pairs,
            methods=["GET"],
            response_model=None,
        )
        self.router.add_api_route(
            path="/train/session/{session_id}", endpoint=self.get_session_pairs, methods=["GET"]
        )
        self.router.add_api_route(
            path="/train/pair/{pair_id}", endpoint=self.update_pair_quality, methods=["PATCH"]
        )
        self.router.add_api_route(
            path="/train/pair/{pair_id}", endpoint=self.delete_pair, methods=["DELETE"]
        )
        self.router.add_api_route(
            path="/train/synced", endpoint=self.delete_synced_pairs, methods=["DELETE"]
        )
        self.router.add_api_route(
            path="/train/pairs/bulk", endpoint=self.delete_pairs_bulk, methods=["DELETE"]
        )
        self.router.add_api_route(
            path="/train/compact", endpoint=self.compact_training_store, methods=["POST"]
        )
        self.router.add_api_route(
            path="/train/from-sessions",
            endpoint=self.train_from_sessions,
            methods=["POST"],
            response_model=None,
        )
        self.router.add_api_route(
            path="/train/auto-status", endpoint=self.get_auto_train_status, methods=["GET"]
        )
        self.router.add_api_route(
            path="/train/auto-config", endpoint=self.update_auto_train_config, methods=["PATCH"]
        )

    @staticmethod
    def _get_health_data():
        """Get basic health data directly from the health controller."""
        from controllers.health import get_health_controller

        return get_health_controller().get_basic_health()

    @staticmethod
    def _get_detailed_health():
        """Get detailed health data directly from the health controller."""
        from controllers.health import get_health_controller

        return get_health_controller().get_detailed_health()

    @staticmethod
    def _get_sessions_list():
        """Get session list directly from the inference router."""
        try:
            from routers.inference import _instance

            return _instance._build_session_metadata_index()
        except Exception as exc:
            logger.debug("Session list fallback to SessionCore: %s", exc)
            from domains.infrastructure.session_core import SessionCore

            return SessionCore.list_sessions()

    @staticmethod
    def _get_session_messages(session_id: str):
        """Get session messages directly."""
        from domains.infrastructure.session_core import SessionCore

        return SessionCore.get_messages(session_id)

    @staticmethod
    def _get_souls():
        """Get soul list directly from the SloManager."""
        from domains.inference.slo_manager import get_slo_manager

        mgr = get_slo_manager()
        souls = mgr.list_souls()
        return [
            {"name": s.name, "description": s.description, "traits": getattr(s, "traits", [])}
            for s in souls
        ]

    @staticmethod
    def _get_current_soul():
        """Get current soul directly from the SloManager."""
        from domains.inference.slo_manager import get_slo_manager

        soul = get_slo_manager().get_current_soul()
        if soul is None:
            return {"name": None}
        return {
            "name": soul.name,
            "description": soul.description,
            "traits": getattr(soul, "traits", []),
        }

    @staticmethod
    def _get_models_list():
        """Get model list directly from the models controller."""
        from controllers.models import get_models_controller

        ctrl = get_models_controller()
        current = ctrl.get_current_model()
        models = ctrl.list_hf_models()
        result = []
        for m in models:
            result.append(
                {
                    "model_id": m.get("model_id", m.get("id", "")),
                    "name": m.get("name", m.get("model_id", "")),
                    "loaded": m.get("status") == "loaded",
                    "size_gb": m.get("size_gb", 0),
                    "source": m.get("source", "local"),
                }
            )
        if current:
            result.insert(
                0,
                {
                    "model_id": current.get("model_id", current.get("id", "")),
                    "name": current.get("name", current.get("model_id", "")),
                    "loaded": True,
                    "size_gb": current.get("size_gb", 0),
                    "source": current.get("source", "local"),
                },
            )
        return result

    @staticmethod
    def _get_knowledge_items(limit: int = 200, offset: int = 0, topic: str | None = None):
        """Get knowledge items directly."""
        from domains.learner.knowledge import get_knowledge_memory

        km = get_knowledge_memory()
        items = km.list_all(top_k=limit + offset)
        if topic:
            items = [i for i in items if getattr(i, "topic", None) == topic]
        return [
            {
                "id": getattr(i, "id", ""),
                "content": getattr(i, "content", ""),
                "topic": getattr(i, "topic", ""),
                "source": getattr(i, "source", "manual"),
                "importance": getattr(i, "importance", 0.5),
                "url": getattr(i, "url", ""),
                "timestamp": getattr(i, "timestamp", 0),
                "score": getattr(i, "score", 0),
            }
            for i in items[offset : offset + limit]
        ]

    @staticmethod
    def _search_knowledge(query: str, limit: int = 10):
        """Search knowledge items directly."""
        from domains.learner.knowledge import get_knowledge_memory

        km = get_knowledge_memory()
        results = km.search(query, top_k=limit)
        return [
            {
                "id": getattr(r, "id", ""),
                "content": getattr(r, "content", ""),
                "topic": getattr(r, "topic", ""),
                "source": getattr(r, "source", "manual"),
                "importance": getattr(r, "importance", 0.5),
                "score": getattr(r, "score", 0),
            }
            for r in results
        ]

    @staticmethod
    def _create_knowledge_item(content: str, topic: str | None = None):
        """Create a knowledge item directly."""
        from domains.learner.knowledge import get_knowledge_memory

        km = get_knowledge_memory()
        return km.store(content, topic=topic)

    @staticmethod
    def _update_knowledge_item(
        item_id: str,
        content: str | None = None,
        topic: str | None = None,
        importance: float | None = None,
    ):
        """Update a knowledge item directly."""
        from domains.learner.knowledge import get_knowledge_memory

        km = get_knowledge_memory()
        return km.update(item_id, content=content, topic=topic, importance=importance)

    @staticmethod
    def _delete_knowledge_item(item_id: str):
        """Delete a knowledge item directly."""
        from domains.learner.knowledge import get_knowledge_memory

        km = get_knowledge_memory()
        return km.delete(item_id)

    @staticmethod
    def _get_checkpoints():
        """Get training checkpoints directly."""
        try:
            from training.controller import get_training_controller

            return get_training_controller().list_checkpoints()
        except Exception as exc:
            import logging
            logging.getLogger("slo.mobile").warning("Failed to list checkpoints: %s", exc)
            return []

    @staticmethod
    def _get_training_status():
        """Get training status directly."""
        from training.controller import get_training_controller

        return get_training_controller().get_status()

    @staticmethod
    def _get_system_metrics():
        """Get system metrics directly."""
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
        }

    @staticmethod
    def _get_disk_info():
        """Get disk usage info."""
        import psutil

        disk = psutil.disk_usage("/")
        return {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent": disk.percent,
        }

    @staticmethod
    def _switch_soul(soul_name: str, checkpoint_name: str | None = None):
        """Switch soul directly."""
        from domains.inference.slo_manager import get_slo_manager

        mgr = get_slo_manager()
        return mgr.switch_soul(soul_name, checkpoint_name=checkpoint_name)

    @staticmethod
    def _load_model(model_id: str):
        """Load a model directly."""
        from controllers.models import get_models_controller

        return get_models_controller().load_model(model_id)

    async def get_dashboard(self, request: Request) -> dict:
        try:
            """
            Mobile dashboard — aggregated home screen data.

            Returns:
                status, model info, current soul, recent conversations, stats.
            """
            health = self._get_health_data()
            soul = self._get_current_soul()
            sessions = self._get_sessions_list()
            models = self._get_models_list()

            if isinstance(sessions, dict):
                sessions = sessions.get("data", [])

            recent = []
            for s in sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)[:5]:
                recent.append(
                    {
                        "id": s.get("id", s.get("session_id", "")),
                        "title": s.get("name", s.get("title", "New Chat")),
                        "last_message": "",
                        "updated_at": s.get("updated_at", ""),
                    }
                )

            return success_response(
                data={
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
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.get_dashboard")

    async def list_conversations(
        self,
        request: Request,
        page: int = Query(1, ge=1),
        per_page: int = Query(20, ge=1, le=100),
        search: str | None = Query(None),
    ) -> dict:
        """Paginated conversation list for mobile."""
        sessions = self._get_sessions_list()
        if isinstance(sessions, dict):
            sessions = sessions.get("data", [])

        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        if search:
            search_lower = search.lower()
            sessions = [
                s
                for s in sessions
                if search_lower in (s.get("name", s.get("title", "")) or "").lower()
            ]

        total = len(sessions)
        start = (page - 1) * per_page
        end = start + per_page
        page_sessions = sessions[start:end]

        result = []
        for s in page_sessions:
            result.append(
                {
                    "id": s.get("id", s.get("session_id", "")),
                    "title": s.get("name", s.get("title", "New Chat")),
                    "last_message": "",
                    "updated_at": s.get("updated_at", ""),
                    "created_at": s.get("created_at", ""),
                    "message_count": s.get("message_count", 0),
                    "starred": s.get("starred", False),
                    "pinned": s.get("pinned", False),
                }
            )

        return success_response(
            data={
                "conversations": result,
                "total": total,
                "page": page,
                "per_page": per_page,
            }
        )

    async def get_conversation(self, request: Request, session_id: str) -> dict:
        try:
            """Single conversation with full message history."""
            messages = self._get_session_messages(session_id)
            return success_response(
                data={
                    "id": session_id,
                    "messages": messages or [],
                    "created_at": "",
                }
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.get_conversation")

    async def get_models(self, request: Request) -> dict:
        try:
            """Available models with current selection and checkpoint options."""
            models = self._get_models_list()
            soul = self._get_current_soul()
            souls = self._get_souls()
            checkpoints = self._get_checkpoints()
            health = self._get_health_data()

            model_list = []
            if isinstance(models, list):
                for m in models:
                    model_list.append(
                        {
                            "id": m.get("model_id", m.get("id", "")),
                            "name": m.get("name", m.get("model_id", "")),
                            "loaded": m.get("loaded", m.get("status") == "loaded"),
                            "size_gb": m.get("size_gb", 0),
                            "source": m.get("source", "local"),
                        }
                    )

            soul_list = []
            if isinstance(souls, list):
                for s in souls:
                    soul_list.append(
                        {
                            "name": s.get("name", ""),
                            "description": s.get("description", ""),
                            "traits": s.get("traits", []),
                        }
                    )

            cp_list = []
            if isinstance(checkpoints, list):
                for cp in checkpoints:
                    if isinstance(cp, dict):
                        cp_list.append(
                            {
                                "name": cp.get("name", cp.get("checkpoint_name", "")),
                                "soul": cp.get("soul", ""),
                                "loss": cp.get("loss"),
                                "steps": cp.get("steps"),
                            }
                        )

            return success_response(
                data={
                    "current": {
                        "model_id": health.get("model_type", ""),
                        "soul": soul.get("name", ""),
                        "checkpoint": None,
                    },
                    "models": model_list,
                    "souls": soul_list,
                    "checkpoints": cp_list,
                }
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.get_models")

    async def switch_model(
        self,
        request: Request,
        body: SwitchRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        try:
            """Switch model and/or soul in one call."""
            errors = []
            if body.model_id:
                try:
                    self._load_model(body.model_id)
                except Exception as e:
                    errors.append(f"model_load: {e}")

            if body.soul_name:
                try:
                    self._switch_soul(body.soul_name, body.checkpoint_name)
                except Exception as e:
                    errors.append(f"soul_switch: {e}")

            health = self._get_health_data()

            if errors:
                logger.warning(
                    "Mobile switch_model partial failure: %s",
                    "; ".join(errors),
                    extra={"tag": "REQ"},
                )

            return success_response(
                data={
                    "status": "ok" if not errors else "partial",
                    "model": health.get("model_type", ""),
                    "soul": body.soul_name or "",
                    "checkpoint": body.checkpoint_name,
                    "errors": errors or None,
                }
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.switch_model")

    async def get_health(self, request: Request) -> dict:
        try:
            """System health summary for mobile."""
            detailed = self._get_detailed_health()
            metrics = self._get_system_metrics()
            disk = self._get_disk_info()

            system_info = detailed.get("system", {})

            return success_response(
                data={
                    "status": detailed.get("status", "unknown"),
                    "model": {
                        "name": detailed.get("model_type", ""),
                        "loaded": detailed.get("model_loaded", False),
                        "type": detailed.get("model_type", ""),
                    },
                    "uptime_seconds": detailed.get("uptime_seconds", 0),
                    "cpu_percent": system_info.get("cpu_percent", metrics.get("cpu_percent", 0)),
                    "memory_percent": system_info.get(
                        "memory_percent", metrics.get("memory_percent", 0)
                    ),
                    "memory_available_gb": round(
                        system_info.get("memory_available_mb", 0) / 1024, 2
                    ),
                    "disk_used_gb": disk.get("used_gb", 0),
                    "disk_free_gb": disk.get("free_gb", 0),
                    "inference_count": detailed.get("inference", {}).get(
                        "inference_count", detailed.get("inference_count", 0)
                    ),
                }
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.get_health")

    async def list_knowledge(
        self,
        request: Request,
        page: int = Query(1, ge=1),
        per_page: int = Query(20, ge=1, le=100),
        topic: str | None = Query(None),
        search: str | None = Query(None),
    ) -> dict:
        """Paginated knowledge items for mobile."""
        if search:
            items = self._search_knowledge(search, per_page * 5)
        else:
            items = self._get_knowledge_items(per_page * 5, 0, topic)

        if not isinstance(items, list):
            items = []

        if topic and not search:
            items = [i for i in items if i.get("topic") == topic]

        total = len(items)
        start = (page - 1) * per_page
        end = start + per_page
        page_items = items[start:end]

        return success_response(
            data={
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
        )

    async def create_knowledge(
        self,
        request: Request,
        body: KnowledgeCreateRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        try:
            """Add a knowledge item."""
            try:
                item_id = self._create_knowledge_item(body.content, body.topic)
                return success_response(
                    data={"id": item_id, "content": body.content, "topic": body.topic}
                )
            except Exception as e:
                raise_error(f"Failed to create knowledge: {e}", "E_DOMAIN")

        except Exception as e:
            classify_and_raise(e, source="mobile.create_knowledge")

    async def update_knowledge(
        self,
        request: Request,
        item_id: str,
        body: KnowledgeUpdateRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        try:
            """Update a knowledge item."""
            try:
                self._update_knowledge_item(
                    item_id, content=body.content, topic=body.topic, importance=body.importance
                )
                return success_response(data={"id": item_id, "updated": True})
            except Exception as e:
                raise_error(f"Failed to update knowledge: {e}", "E_DOMAIN")

        except Exception as e:
            classify_and_raise(e, source="mobile.update_knowledge")

    async def delete_knowledge(
        self, request: Request, item_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
            """Delete a knowledge item."""
            try:
                self._delete_knowledge_item(item_id)
            except Exception as e:
                raise_error(f"Failed to delete knowledge: {e}", "E_DOMAIN")
            safe_audit_log("mobile.knowledge_delete", resource=item_id)
            return success_response(data={"status": "deleted", "id": item_id})

        except Exception as e:
            classify_and_raise(e, source="mobile.delete_knowledge")

    async def sync_offline(
        self,
        request: Request,
        body: SyncRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        try:
            """
            Sync offline messages when mobile reconnects.

            Processes queued messages from the offline cache, sends them to the
            chat endpoint, and returns results for each.
            """
            from routers.inference import ChatRequest, Message
            from routers.inference import _instance as _inference

            results: list[SyncResult] = []

            for msg in body.pending_messages:
                try:
                    chat_req = ChatRequest(
                        messages=[Message(role="user", content=msg.content)],
                        session_id=msg.session_id,
                    )
                    chat_resp = await _inference.chat(chat_req)
                    chat_result = (
                        chat_resp.model_dump()
                        if hasattr(chat_resp, "model_dump")
                        else (chat_resp if isinstance(chat_resp, dict) else {})
                    )

                    if chat_result and chat_result.get("message"):
                        results.append(
                            SyncResult(
                                id=msg.id,
                                status="sent",
                                assistant_message={
                                    "role": "assistant",
                                    "content": chat_result["message"],
                                    "timestamp": chat_result.get("timestamp", 0),
                                },
                            )
                        )
                    else:
                        results.append(
                            SyncResult(
                                id=msg.id,
                                status="error",
                                error="No response from server",
                            )
                        )
                except Exception as e:
                    results.append(
                        SyncResult(
                            id=msg.id,
                            status="error",
                            error=str(e),
                        )
                    )

            sessions = self._get_sessions_list()
            if isinstance(sessions, dict):
                sessions = sessions.get("data", [])
            elif not isinstance(sessions, list):
                sessions = []

            return success_response(
                data={
                    "results": [r.model_dump() for r in results],
                    "synced_count": sum(1 for r in results if r.status == "sent"),
                    "failed_count": sum(1 for r in results if r.status == "error"),
                    "sessions": sessions[:20],
                }
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.sync_offline")

    async def sync_status(self, request: Request) -> dict:
        try:
            """Check server connectivity and last sync state."""
            health = self._get_health_data()

            return success_response(
                data={
                    "reachable": health.get("status") == "healthy",
                    "model_loaded": health.get("model_loaded", False),
                    "server_time": int(__import__("time").time() * 1000),
                    "inference_count": health.get("inference_count", 0),
                }
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.sync_status")

    async def register_device(
        self, body: DeviceRegistrationRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
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
            return success_response(data=result)

        except Exception as e:
            classify_and_raise(e, source="mobile.register_device")

    async def unregister_device(
        self, body: UnregisterDeviceRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
            """Unregister a device from push notifications."""
            from domains.mobile.notifications import get_notification_service

            svc = get_notification_service()
            removed = svc.unregister_device(body.token)
            return success_response(data={"status": "removed" if removed else "not_found"})

        except Exception as e:
            classify_and_raise(e, source="mobile.unregister_device")

    async def list_devices(self, topic: str | None = Query(None)) -> dict:
        try:
            """
            List registered devices.

            Args:
                topic: Optional topic filter.

            Returns:
                List of registered devices.
            """
            from domains.mobile.notifications import get_notification_service

            svc = get_notification_service()
            return success_response(data={"devices": svc.get_devices(topic=topic)})

        except Exception as e:
            classify_and_raise(e, source="mobile.list_devices")

    async def send_notification(
        self, body: NotificationSendRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
            """
            Send a push notification to registered devices.

            Args:
                body: Title, body, topic filter, data payload, optional token list.

            Returns:
                Send result with recipient count.
            """
            from domains.mobile.notifications import NotificationPayload, get_notification_service

            svc = get_notification_service()
            payload = NotificationPayload(
                title=body.title,
                body=body.body,
                data=body.data or {},
                badge=body.badge,
                topic=body.topic,
            )
            result = await svc.send_notification_async(
                payload=payload,
                tokens=body.tokens,
                topic=body.topic,
            )
            return success_response(data=result)

        except Exception as e:
            classify_and_raise(e, source="mobile.send_notification")

    async def notification_history(self, limit: int = Query(50, ge=1, le=200)) -> dict:
        try:
            """Get recent notification history."""
            from domains.mobile.notifications import get_notification_service

            svc = get_notification_service()
            return success_response(data={"history": svc.get_history(limit=limit)})

        except Exception as e:
            classify_and_raise(e, source="mobile.notification_history")

    async def cleanup_devices(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """
            Remove devices inactive for 30+ days.

            Returns:
                Count of removed devices.
            """
            from domains.mobile.notifications import get_notification_service

            svc = get_notification_service()
            removed = svc.cleanup_stale()
            return success_response(data={"removed": removed})

        except Exception as e:
            classify_and_raise(e, source="mobile.cleanup_devices")

    async def notify_training_complete(
        self, request: Request, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
            """Send a training-complete notification to all registered devices."""
            from domains.mobile.notifications import NotificationPayload, get_notification_service

            svc = get_notification_service()
            training = self._get_training_status()

            status = training.get("status", "unknown")
            loss = training.get("final_loss")

            payload = NotificationPayload(
                title="Training Complete",
                body=f"Model training finished. Final loss: {loss:.4f}"
                if loss is not None
                else f"Training {status}",
                data={"type": "training_complete", "status": status},
                topic="training",
            )

            result = svc.send_notification(payload=payload, topic="training")
            return success_response(data=result)

        except Exception as e:
            classify_and_raise(e, source="mobile.notify_training_complete")

    async def mobile_train(
        self, body: MobileTrainRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Train the SloNet model on conversation pairs from the mobile app."""
        import time as _time
        from pathlib import Path

        if len(body.pairs) < 5:
            raise_error("Need at least 5 training pairs", "E_BAD_REQUEST", status_code=400)

        t0 = int(_time.time() * 1000)

        from domains.training.mobile_training_store import get_training_store

        store = get_training_store()
        pair_ids = store.add_batch(
            [
                {
                    "user_msg": p.user_msg,
                    "assistant_msg": p.assistant_msg,
                    "session_id": p.session_id,
                    "quality": p.quality,
                }
                for p in body.pairs
            ]
        )

        repo_root = Path(__file__).resolve().parents[4]
        train_dir = repo_root / "data" / "mobile_training"
        await asyncio.to_thread(train_dir.mkdir, parents=True, exist_ok=True)

        ts = int(_time.time())
        text_file = train_dir / f"mobile_{ts}.txt"

        def _write_pairs():
            with open(text_file, "w") as f:
                for pair in body.pairs:
                    f.write(f"User: {pair.user_msg}\nAssistant: {pair.assistant_msg}\n\n")

        await asyncio.to_thread(_write_pairs)

        checkpoint_name = f"mobile_{ts}"
        output_dir = repo_root / "models" / "auto-training" / checkpoint_name
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

        venv_python = repo_root / ".venv" / "bin" / "python3"
        train_script = repo_root / "scripts" / "hf_train.py"

        if not await asyncio.to_thread(venv_python.exists):
            raise_error(
                "Training environment not found (.venv missing)", "E_INFRA_STARTUP", status_code=500
            )

        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                [
                    str(venv_python),
                    str(train_script),
                    "--data",
                    str(text_file),
                    "--output",
                    str(output_dir),
                    "--model",
                    "gpt2",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--lr",
                    "5e-5",
                    "--max-seq-length",
                    "256",
                    "--use-lora",
                    "--lora-rank",
                    "8",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if proc.returncode != 0:
                logger.error(
                    "Training subprocess failed: %s", proc.stderr[-500:], extra={"tag": "REQ"}
                )
                raise_error(
                    f"Training failed: {proc.stderr[-200:]}", "E_INFRA_STARTUP", status_code=500
                )

            try:
                result = json.loads(proc.stdout.strip().split("\n")[-1])
            except json.JSONDecodeError:
                logger.error(
                    "Non-JSON subprocess output: %s", proc.stdout[-500:], extra={"tag": "REQ"}
                )
                raise_error(
                    f"Training produced invalid output: {proc.stdout[-200:]}",
                    "E_INFRA_STARTUP",
                    status_code=500,
                )

            if not result.get("success"):
                raise_error(
                    f"Training failed: {result.get('error', 'unknown')}",
                    "E_INFRA_STARTUP",
                    status_code=500,
                )

            store.mark_used(pair_ids)
            store.mark_synced(pair_ids)

            elapsed = int(_time.time() * 1000) - t0
            safe_audit_log(
                "mobile.train",
                resource=checkpoint_name,
                detail=f"elapsed_ms={elapsed}",
                pairs=len(body.pairs),
                loss=result.get("loss", 0.0),
            )

            return success_response(
                data=MobileTrainResult(
                    success=True,
                    checkpoint_name=checkpoint_name,
                    loss=result.get("loss", 0.0),
                    steps=result.get("steps", 0),
                    elapsed_ms=elapsed,
                ).model_dump()
            )

        except subprocess.TimeoutExpired:
            raise_error("Training timed out (300s limit)", "E_TIMEOUT", status_code=504)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse training output: %s", e, extra={"tag": "REQ"})
            raise_error("Training produced invalid output", "E_INFRA_STARTUP", status_code=500)
        except AppError as e:
            classify_and_raise(e, source="mobile._write_pairs")
        except Exception as e:
            logger.warning("Mobile train failed: %s", e, extra={"tag": "REQ"})
            classify_and_raise(e, source="mobile_train")

    async def get_training_stats(self) -> dict:
        try:
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

            return success_response(
                data={
                    "total": stats["total"],
                    "pending": stats["pending"],
                    "synced": stats["synced"],
                    "used": stats["used"],
                    "by_quality": quality_counts,
                }
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.get_training_stats")

    async def get_pending_pairs(self, limit: int = Query(50, ge=1, le=500)) -> dict:
        try:
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
            return success_response(
                data={
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
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.get_pending_pairs")

    async def list_training_pairs(
        self,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        min_quality: float | None = Query(None),
        session_id: str | None = Query(None),
        search: str | None = Query(None),
    ) -> dict:
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
        return success_response(
            data={
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
        )

    async def export_training_pairs(
        self,
        min_quality: float | None = Query(None),
        session_id: str | None = Query(None),
        limit: int = Query(500, ge=1, le=5000),
    ) -> AsyncGenerator[str, None]:
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
        from domains.training.mobile_training_store import get_training_store

        store = get_training_store()
        pairs = store.list_pairs(limit=limit, min_quality=min_quality, session_id=session_id)

        def generate() -> AsyncGenerator[str, None]:
            """generate."""
            for p in pairs:
                yield (
                    json.dumps(
                        {
                            "user_msg": p.get("user_msg", ""),
                            "assistant_msg": p.get("assistant_msg", ""),
                            "quality": p.get("quality", 0),
                            "session_id": p.get("session_id", ""),
                        }
                    )
                    + "\n"
                )

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=training_pairs.jsonl"},
        )

    async def get_session_pairs(self, session_id: str) -> dict:
        try:
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
            return success_response(
                data={
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
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.get_session_pairs")

    async def update_pair_quality(
        self,
        pair_id: str,
        body: QualityUpdateRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        try:
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
                raise_error("Pair not found", "E_NOT_FOUND", status_code=404)
            return success_response(
                data={"status": "updated", "pair_id": pair_id, "quality": body.quality}
            )

        except Exception as e:
            classify_and_raise(e, source="mobile.update_pair_quality")

    async def delete_pair(
        self, pair_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
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
                raise_error("Pair not found", "E_NOT_FOUND", status_code=404)
            safe_audit_log("mobile.pair_delete", resource=pair_id)
            return success_response(data={"status": "deleted", "pair_id": pair_id})

        except Exception as e:
            classify_and_raise(e, source="mobile.delete_pair")

    async def delete_synced_pairs(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
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
            safe_audit_log("mobile.pairs_delete_synced", detail=f"count={count}")
            return success_response(data={"status": "deleted", "count": count})

        except Exception as e:
            classify_and_raise(e, source="mobile.delete_synced_pairs")

    async def delete_pairs_bulk(
        self,
        ids: list[str] = Query(..., description="Pair IDs to delete"),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        try:
            """Delete multiple training pairs by ID."""
            from domains.training.mobile_training_store import get_training_store

            store = get_training_store()
            count = 0
            for pair_id in ids:
                if store.delete_pair(pair_id):
                    count += 1
            safe_audit_log("mobile.pairs_delete_bulk", detail=f"count={count} requested={len(ids)}")
            return success_response(data={"status": "deleted", "count": count})

        except Exception as e:
            classify_and_raise(e, source="mobile.delete_pairs_bulk")

    async def compact_training_store(
        self, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
            """Compact the training data store (reclaim space from deleted records)."""
            from domains.training.mobile_training_store import get_training_store

            store = get_training_store()
            count = store.compact()
            return success_response(data={"status": "compacted", "count": count})

        except Exception as e:
            classify_and_raise(e, source="mobile.compact_training_store")

    async def train_from_sessions(
        self,
        body: FromSessionsRequest = FromSessionsRequest(),
        request: Request = None,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> AsyncGenerator[str, None]:
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

        from domains.api.sse_envelope import sse_complete, sse_error, sse_event

        t0 = _time.time()

        # Extract pairs from server logs
        from domains.training.pair_extractor import (
            extract_pairs_from_logs,
            extract_pairs_from_sessions,
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
            yield sse_error(
                "training",
                "GENERATE_DATA",
                f"Need at least 5 training pairs, found {len(pairs)} in server logs",
                code="E_VAL_REQUEST",
                http_status=400,
            )
            return

        # Emit GENERATE_DATA phase
        yield sse_event(
            stream="training",
            phase="GENERATE_DATA",
            status="working",
            data={"pairs": len(pairs)},
            message=f"Extracted {len(pairs)} training pairs",
        )

        # Write training text file
        text_file = write_training_text(pairs)

        # Store pairs in MogDB (with quality scoring)
        from domains.training.mobile_training_store import get_training_store
        from domains.training.quality_scorer import score_batch

        quality_scores = score_batch(pairs)
        store = get_training_store()
        store.add_batch(
            [
                {
                    "user_msg": p["user_msg"],
                    "assistant_msg": p["assistant_msg"],
                    "session_id": p.get("session_id", ""),
                    "quality": quality_scores[i] if i < len(quality_scores) else 0,
                }
                for i, p in enumerate(pairs)
            ]
        )

        # Prepare subprocess
        repo_root = Path(__file__).resolve().parents[4]
        ts = int(_time.time())
        checkpoint_name = f"sessions_{ts}"
        output_dir = repo_root / "models" / "auto-training" / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)

        venv_python = repo_root / ".venv" / "bin" / "python3"
        train_script = repo_root / "scripts" / "hf_train.py"

        if not venv_python.exists():
            yield sse_error(
                "training",
                "TRAIN",
                "Training environment not found (.venv missing)",
                code="E_ENV_MISSING",
                http_status=500,
            )
            return

        # Launch subprocess with streaming stdout
        cmd = [
            str(venv_python),
            str(train_script),
            "--stream",
            "--data",
            str(text_file),
            "--output",
            str(output_dir),
            "--model",
            "gpt2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--lr",
            "5e-5",
            "--max-seq-length",
            "256",
            "--use-lora",
            "--lora-rank",
            "8",
        ]

        try:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            logger.warning("Mobile train subprocess spawn failed: %s", e, extra={"tag": "REQ"})
            classify_and_raise(e, source="mobile_train_from_sessions")

        # Stream stdout lines as SSE events
        deadline = _time.time() + 600  # 10-minute hard timeout
        disconnect_check_interval = 5.0  # Check disconnect every 5s while waiting for lines
        result = None

        try:
            while True:
                if _time.time() > deadline:
                    proc.kill()
                    yield sse_error(
                        "training",
                        "TRAIN",
                        "Training timed out (10 min)",
                        code="E_TIMEOUT",
                        http_status=408,
                    )
                    return

                # Check client disconnect periodically
                if request and await request.is_disconnected():
                    proc.kill()
                    logger.info("Client disconnected from training stream", extra={"tag": "REQ"})
                    return

                try:
                    # Read one line (with timeout for periodic disconnect checks)
                    line = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(None, proc.stdout.readline),
                        timeout=disconnect_check_interval,
                    )
                except asyncio.TimeoutError:
                    # readline timed out — loop back to check deadline/disconnect
                    continue

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

                # Check if this is the final result
                if event.get("success") is not None or phase == "COMPLETE":
                    result = event
                    break

                # Forward progress event as SSE
                elapsed_ms = round((_time.time() - t0) * 1000)
                yield sse_event(
                    stream="training",
                    phase=phase,
                    status="working",
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
            await asyncio.to_thread(proc.wait, 30)

            if proc.returncode != 0:
                stderr_tail = (
                    (await asyncio.to_thread(proc.stderr.read))[-500:] if proc.stderr else ""
                )
                logger.error(
                    "Training subprocess failed (rc=%d): %s",
                    proc.returncode,
                    stderr_tail,
                    extra={"tag": "REQ"},
                )
                yield sse_error(
                    "training",
                    "TRAIN",
                    f"Training failed (exit code {proc.returncode})",
                    code="E_INFRA_GENERATION",
                    http_status=500,
                )
                return

            # Parse final result if not already captured from stdout
            if result is None:
                # Try reading any remaining stderr for errors
                stderr_tail = (
                    (await asyncio.to_thread(proc.stderr.read))[-500:] if proc.stderr else ""
                )
                yield sse_error(
                    "training",
                    "TRAIN",
                    f"Training produced no result output. stderr: {stderr_tail[:200]}",
                    code="E_INFRA_GENERATION",
                    http_status=500,
                )
                return

            elapsed_ms = round((_time.time() - t0) * 1000)
            safe_audit_log(
                "mobile.train_sessions",
                resource=checkpoint_name,
                detail=f"elapsed_ms={elapsed_ms}",
                pairs=len(pairs),
                loss=result.get("loss", 0.0),
            )

            yield sse_complete(
                stream="training",
                phase="COMPLETE",
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

        except Exception as e:
            logger.error("Session training failed: %s", e, extra={"tag": "REQ"})
            yield sse_error(
                "training",
                "TRAIN",
                f"Training failed: {e}",
                code="E_INFRA_GENERATION",
                http_status=500,
            )
        finally:
            if proc.poll() is None:
                proc.kill()

    async def get_auto_train_status(self) -> dict:
        try:
            """
            Get auto-trainer status and configuration.

            Returns:
                Enabled state, threshold, pending count, last train info.

            Side effects:
                - None (reads from AutoTrainer singleton).
            """
            from domains.training.auto_trainer import get_auto_trainer

            trainer = get_auto_trainer()
            return success_response(data=trainer.status())

        except Exception as e:
            classify_and_raise(e, source="mobile.get_auto_train_status")

    async def update_auto_train_config(
        self,
        threshold: int | None = Query(None, ge=1, le=100),
        interval_s: int | None = Query(None, ge=30, le=3600),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
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
        return success_response(data=trainer.status())


router = MobileRouter().router
