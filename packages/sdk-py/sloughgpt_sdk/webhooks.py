"""
SloughGPT SDK - Webhooks
Webhook system for event notifications.
"""

import hashlib
import hmac
import secrets
import json
import time
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import queue


class WebhookEvent(Enum):
    """Available webhook events."""
    KEY_CREATED = "key.created"
    KEY_REVOKED = "key.revoked"
    KEY_DELETED = "key.deleted"
    KEY_ROTATED = "key.rotated"
    QUOTA_EXCEEDED = "quota.exceeded"
    QUOTA_WARNING = "quota.warning"
    USAGE_SPIKE = "usage.spike"
    SUBSCRIPTION_CREATED = "subscription.created"
    SUBSCRIPTION_UPDATED = "subscription.updated"
    SUBSCRIPTION_CANCELLED = "subscription.cancelled"
    PAYMENT_SUCCESS = "payment.success"
    PAYMENT_FAILED = "payment.failed"
    GENERATION_COMPLETED = "generation.completed"
    GENERATION_FAILED = "generation.failed"
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADED = "model.unloaded"
    SYSTEM_ALERT = "system.alert"


@dataclass
class Webhook:
    """Represents a webhook subscription."""
    id: str
    url: str
    events: List[WebhookEvent]
    secret: str
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_triggered_at: Optional[float] = None
    last_success_at: Optional[float] = None
    last_failure_at: Optional[float] = None
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_triggered_at": self.last_triggered_at,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "failure_count": self.failure_count,
            "metadata": self.metadata,
        }


@dataclass
class WebhookPayload:
    """Webhook event payload."""
    event: WebhookEvent
    timestamp: float
    data: Dict[str, Any]
    webhook_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event": self.event.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "webhook_id": self.webhook_id,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class WebhookDelivery:
    """Result of webhook delivery attempt."""
    def __init__(
        self,
        webhook_id: str,
        payload: Dict[str, Any],
        status_code: Optional[int] = None,
        success: bool = False,
        error: Optional[str] = None,
        duration_ms: float = 0,
    ):
        self.webhook_id = webhook_id
        self.payload = payload
        self.status_code = status_code
        self.success = success
        self.error = error
        self.duration_ms = duration_ms
        self.attempts = 1


class WebhookManager:
    """
    Manages webhook subscriptions and deliveries.
    
    Example:
    
    ```python
    from sloughgpt_sdk.webhooks import WebhookManager, WebhookEvent
    
    manager = WebhookManager()
    
    # Register a webhook
    webhook = manager.create_webhook(
        url="https://mysite.com/webhooks",
        events=[WebhookEvent.KEY_CREATED, WebhookEvent.QUOTA_EXCEEDED],
    )
    
    # Trigger an event
    manager.trigger_event(
        event=WebhookEvent.KEY_CREATED,
        data={"key_id": "sk_xxx", "tier": "pro"}
    )
    
    # Verify webhook signature
    is_valid = manager.verify_signature(signature, payload, webhook.secret)
    ```
    """
    
    def __init__(self, storage_path: str = "./.webhooks.json"):
        """
        Initialize webhook manager.
        
        Args:
            storage_path: Path to store webhook data.
        """
        self._storage_path = storage_path
        self._webhooks: Dict[str, Webhook] = {}
        self._event_handlers: Dict[WebhookEvent, List[Callable]] = {}
        self._delivery_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._load_webhooks()
    
    def _load_webhooks(self):
        """Load webhooks from storage."""
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
                for wh_data in data.get("webhooks", []):
                    events = [WebhookEvent(e) for e in wh_data.get("events", [])]
                    webhook = Webhook(
                        id=wh_data["id"],
                        url=wh_data["url"],
                        events=events,
                        secret=wh_data["secret"],
                        is_active=wh_data.get("is_active", True),
                        created_at=wh_data.get("created_at", time.time()),
                        last_triggered_at=wh_data.get("last_triggered_at"),
                        last_success_at=wh_data.get("last_success_at"),
                        last_failure_at=wh_data.get("last_failure_at"),
                        failure_count=wh_data.get("failure_count", 0),
                        metadata=wh_data.get("metadata", {}),
                    )
                    self._webhooks[webhook.id] = webhook
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def _save_webhooks(self):
        """Save webhooks to storage."""
        data = {
            "webhooks": [wh.to_dict() for wh in self._webhooks.values()]
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def generate_webhook_id() -> str:
        """Generate a unique webhook ID."""
        return f"wh_{int(time.time())}_{secrets.token_hex(4)}"
    
    @staticmethod
    def generate_secret() -> str:
        """Generate a secure webhook secret."""
        import secrets
        return secrets.token_urlsafe(32)
    
    def create_webhook(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Webhook:
        """
        Create a new webhook subscription.
        
        Args:
            url: The URL to send webhooks to.
            events: List of events to subscribe to.
            secret: Optional secret for signature verification.
            metadata: Optional metadata.
        
        Returns:
            The created Webhook object.
        """
        webhook_id = self.generate_webhook_id()
        webhook_secret = secret or self.generate_secret()
        
        webhook = Webhook(
            id=webhook_id,
            url=url,
            events=events,
            secret=webhook_secret,
            metadata=metadata or {},
        )
        
        self._webhooks[webhook_id] = webhook
        self._save_webhooks()
        
        return webhook
    
    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Get a webhook by ID."""
        return self._webhooks.get(webhook_id)
    
    def list_webhooks(self, event_filter: Optional[WebhookEvent] = None) -> List[Webhook]:
        """List all webhooks, optionally filtered by event."""
        webhooks = list(self._webhooks.values())
        if event_filter:
            webhooks = [wh for wh in webhooks if event_filter in wh.events]
        return webhooks
    
    def update_webhook(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[WebhookEvent]] = None,
        is_active: Optional[bool] = None,
    ) -> bool:
        """Update a webhook."""
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            return False
        
        if url is not None:
            webhook.url = url
        if events is not None:
            webhook.events = events
        if is_active is not None:
            webhook.is_active = is_active
        
        self._save_webhooks()
        return True
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook."""
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            self._save_webhooks()
            return True
        return False
    
    def register_handler(self, event: WebhookEvent, handler: Callable):
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def trigger_event(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
    ) -> List[WebhookDelivery]:
        """
        Trigger an event to all subscribed webhooks.
        
        Returns:
            List of delivery results.
        """
        webhooks = self.list_webhooks(event)
        deliveries = []
        
        for webhook in webhooks:
            if not webhook.is_active:
                continue
            
            payload = WebhookPayload(
                event=event,
                timestamp=time.time(),
                data=data,
                webhook_id=webhook.id,
            )
            
            delivery = self._deliver_webhook(webhook, payload)
            deliveries.append(delivery)
            
            webhook.last_triggered_at = time.time()
            if delivery.success:
                webhook.last_success_at = time.time()
                webhook.failure_count = 0
            else:
                webhook.last_failure_at = time.time()
                webhook.failure_count += 1
        
        self._save_webhooks()
        return deliveries
    
    def _deliver_webhook(
        self,
        webhook: Webhook,
        payload: WebhookPayload,
    ) -> WebhookDelivery:
        """Deliver a webhook to its URL."""
        import requests
        import secrets
        
        start_time = time.time()
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            payload=payload.to_dict(),
        )
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-ID": webhook.id,
                "X-Webhook-Event": payload.event.value,
                "X-Webhook-Timestamp": str(payload.timestamp),
                "X-Webhook-Signature": self._generate_signature(
                    payload.to_json(), webhook.secret
                ),
            }
            
            response = requests.post(
                webhook.url,
                data=payload.to_json(),
                headers=headers,
                timeout=30,
            )
            
            delivery.status_code = response.status_code
            delivery.success = 200 <= response.status_code < 300
            if not delivery.success:
                delivery.error = f"HTTP {response.status_code}"
            
        except requests.exceptions.Timeout:
            delivery.success = False
            delivery.error = "Request timeout"
        except requests.exceptions.RequestException as e:
            delivery.success = False
            delivery.error = str(e)
        finally:
            delivery.duration_ms = (time.time() - start_time) * 1000
        
        return delivery
    
    @staticmethod
    def _generate_signature(payload: str, secret: str) -> str:
        """Generate HMAC signature for payload."""
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    @staticmethod
    def verify_signature(signature: str, payload: str, secret: str) -> bool:
        """
        Verify webhook signature.
        
        Args:
            signature: The signature from the request header.
            payload: The raw request body.
            secret: The webhook's secret.
        
        Returns:
            True if signature is valid.
        """
        if not signature.startswith("sha256="):
            return False
        
        expected = WebhookManager._generate_signature(payload, secret)
        return hmac.compare_digest(signature, expected)
    
    def get_webhook_stats(self, webhook_id: str) -> Optional[Dict[str, Any]]:
        """Get delivery statistics for a webhook."""
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            return None
        
        return {
            "id": webhook.id,
            "url": webhook.url,
            "is_active": webhook.is_active,
            "events_subscribed": [e.value for e in webhook.events],
            "created_at": webhook.created_at,
            "last_triggered_at": webhook.last_triggered_at,
            "last_success_at": webhook.last_success_at,
            "last_failure_at": webhook.last_failure_at,
            "failure_count": webhook.failure_count,
            "failure_rate": (
                webhook.failure_count / 10
                if webhook.last_triggered_at
                else 0
            ),
        }


class WebhookTester:
    """
    Test webhook endpoints.
    
    Example:
    
    ```python
    from sloughgpt_sdk.webhooks import WebhookTester, WebhookEvent
    
    tester = WebhookTester()
    
    # Test a webhook URL
    result = tester.test_webhook(
        url="https://mysite.com/webhooks",
        event=WebhookEvent.KEY_CREATED,
        secret="webhook_secret"
    )
    
    print(f"Success: {result['success']}")
    print(f"Response time: {result['duration_ms']}ms")
    ```
    """
    
    @staticmethod
    def test_webhook(
        url: str,
        event: WebhookEvent,
        secret: Optional[str] = None,
        test_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Test a webhook endpoint."""
        import requests
        import secrets
        
        payload = {
            "event": event.value,
            "timestamp": time.time(),
            "data": test_data or {"test": True, "message": "This is a test webhook"},
            "webhook_id": f"test_{secrets.token_hex(4)}",
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Test": "true",
        }
        
        if secret:
            payload_str = json.dumps(payload)
            headers["X-Webhook-Signature"] = WebhookManager._generate_signature(
                payload_str, secret
            )
        
        start_time = time.time()
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=30,
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "success": 200 <= response.status_code < 300,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "response_body": response.text[:500] if response.text else None,
                "error": None,
            }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "status_code": None,
                "duration_ms": (time.time() - start_time) * 1000,
                "response_body": None,
                "error": "Request timeout",
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "status_code": None,
                "duration_ms": (time.time() - start_time) * 1000,
                "response_body": None,
                "error": str(e),
            }
