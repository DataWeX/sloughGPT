"""
SloughGPT SDK - API Key Management
Comprehensive API key management for the SDK.
"""

import hashlib
import hmac
import secrets
import time
import json
import base64
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class KeyTier(Enum):
    """API key subscription tiers."""
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class APIKey:
    """Represents an API key."""
    key_id: str
    key_hash: str
    prefix: str
    name: str
    user_id: Optional[str] = None
    tier: KeyTier = KeyTier.FREE
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_used_at: Optional[float] = None
    is_active: bool = True
    rate_limit: int = 60
    quota_daily: int = 100
    quota_monthly: int = 1000
    usage_count: int = 0
    usage_today: int = 0
    usage_this_month: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if key is valid (active, not expired)."""
        return self.is_active and not self.is_expired()
    
    def check_quota(self) -> Tuple[bool, str]:
        """Check if quota is available. Returns (allowed, reason)."""
        if self.usage_today >= self.quota_daily:
            return False, "Daily quota exceeded"
        if self.usage_this_month >= self.quota_monthly:
            return False, "Monthly quota exceeded"
        return True, "OK"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_id": self.key_id,
            "prefix": self.prefix,
            "name": self.name,
            "tier": self.tier.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used_at": self.last_used_at,
            "is_active": self.is_active,
            "rate_limit": self.rate_limit,
            "quota_daily": self.quota_daily,
            "quota_monthly": self.quota_monthly,
            "usage_today": self.usage_today,
            "usage_this_month": self.usage_this_month,
            "metadata": self.metadata,
        }


class APIKeyManager:
    """
    Manages API keys for the SloughGPT SDK.
    
    Example:
    
    ```python
    from sloughgpt_sdk.auth import APIKeyManager, KeyTier
    
    manager = APIKeyManager()
    
    # Generate a new API key
    key, key_data = manager.create_key(
        name="My App",
        tier=KeyTier.PRO,
        quota_daily=1000
    )
    
    # Store key securely
    print(f"API Key: {key}")  # Only shown once!
    
    # Validate a key
    is_valid, reason = manager.validate_key(key)
    
    # Track usage
    manager.record_usage(key, tokens_used=100)
    
    # Revoke a key
    manager.revoke_key(key_id="key_xxx")
    ```
    """
    
    def __init__(self, storage_path: str = "./.api_keys.json"):
        """
        Initialize the API key manager.
        
        Args:
            storage_path: Path to store key data.
        """
        self._storage_path = storage_path
        self._keys: Dict[str, APIKey] = {}
        self._key_lookup: Dict[str, str] = {}  # hash -> key_id
        self._load_keys()
    
    def _load_keys(self):
        """Load keys from storage."""
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
                for key_data in data.get("keys", []):
                    key = APIKey(
                        key_id=key_data["key_id"],
                        key_hash=key_data["key_hash"],
                        prefix=key_data["prefix"],
                        name=key_data["name"],
                        user_id=key_data.get("user_id"),
                        tier=KeyTier(key_data.get("tier", "free")),
                        created_at=key_data.get("created_at", time.time()),
                        expires_at=key_data.get("expires_at"),
                        last_used_at=key_data.get("last_used_at"),
                        is_active=key_data.get("is_active", True),
                        rate_limit=key_data.get("rate_limit", 60),
                        quota_daily=key_data.get("quota_daily", 100),
                        quota_monthly=key_data.get("quota_monthly", 1000),
                        usage_count=key_data.get("usage_count", 0),
                        usage_today=key_data.get("usage_today", 0),
                        usage_this_month=key_data.get("usage_this_month", 0),
                        metadata=key_data.get("metadata", {}),
                    )
                    self._keys[key.key_id] = key
                    self._key_lookup[key.key_hash] = key.key_id
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def _save_keys(self):
        """Save keys to storage."""
        data = {
            "keys": [key.to_dict() for key in self._keys.values()]
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    @staticmethod
    def _generate_key_id() -> str:
        """Generate a unique key ID."""
        return f"sk_{secrets.token_hex(8)}"
    
    @staticmethod
    def _generate_api_key(prefix: str = "slough") -> str:
        """Generate a new API key."""
        key_body = secrets.token_urlsafe(32)
        return f"{prefix}_{key_body}"
    
    def create_key(
        self,
        name: str,
        tier: KeyTier = KeyTier.FREE,
        user_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
        quota_daily: Optional[int] = None,
        quota_monthly: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, APIKey]:
        """
        Create a new API key.
        
        Returns:
            Tuple of (plaintext_key, APIKey object)
            The plaintext key is only returned once - store it securely!
        """
        key_id = self._generate_key_id()
        prefix = f"sk_{secrets.token_hex(4)}"
        plaintext_key = self._generate_api_key(prefix)
        key_hash = self._hash_key(plaintext_key)
        
        tier_quotas = {
            KeyTier.FREE: (60, 100, 1000),
            KeyTier.STARTER: (120, 1000, 10000),
            KeyTier.PRO: (300, 10000, 100000),
            KeyTier.ENTERPRISE: (1000, 100000, 1000000),
        }
        
        default_rate, default_daily, default_monthly = tier_quotas.get(tier, tier_quotas[KeyTier.FREE])
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            prefix=prefix,
            name=name,
            user_id=user_id,
            tier=tier,
            created_at=time.time(),
            expires_at=time.time() + (expires_in_days * 86400) if expires_in_days else None,
            rate_limit=rate_limit or default_rate,
            quota_daily=quota_daily or default_daily,
            quota_monthly=quota_monthly or default_monthly,
            metadata=metadata or {},
        )
        
        self._keys[key_id] = api_key
        self._key_lookup[key_hash] = key_id
        self._save_keys()
        
        return plaintext_key, api_key
    
    def validate_key(self, key: str) -> Tuple[bool, str, Optional[APIKey]]:
        """
        Validate an API key.
        
        Returns:
            Tuple of (is_valid, reason, api_key)
        """
        key_hash = self._hash_key(key)
        
        if key_hash not in self._key_lookup:
            return False, "Invalid API key", None
        
        key_id = self._key_lookup[key_hash]
        api_key = self._keys.get(key_id)
        
        if api_key is None:
            return False, "Key not found", None
        
        if not api_key.is_valid():
            if not api_key.is_active:
                return False, "API key is deactivated", api_key
            if api_key.is_expired():
                return False, "API key has expired", api_key
        
        allowed, reason = api_key.check_quota()
        if not allowed:
            return False, reason, api_key
        
        return True, "OK", api_key
    
    def record_usage(
        self,
        key: str,
        tokens_used: int = 0,
        requests_count: int = 1,
    ) -> bool:
        """
        Record usage for an API key.
        
        Returns:
            True if usage was recorded successfully.
        """
        key_hash = self._hash_key(key)
        
        if key_hash not in self._key_lookup:
            return False
        
        key_id = self._key_lookup[key_hash]
        api_key = self._keys.get(key_id)
        
        if api_key is None:
            return False
        
        api_key.usage_count += requests_count
        api_key.usage_today += requests_count
        api_key.usage_this_month += requests_count
        api_key.last_used_at = time.time()
        
        self._save_keys()
        return True
    
    def get_key_info(self, key_id: str) -> Optional[APIKey]:
        """Get information about an API key."""
        return self._keys.get(key_id)
    
    def list_keys(self, user_id: Optional[str] = None) -> List[APIKey]:
        """List all API keys, optionally filtered by user."""
        keys = list(self._keys.values())
        if user_id:
            keys = [k for k in keys if k.user_id == user_id]
        return keys
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            self._save_keys()
            return True
        return False
    
    def delete_key(self, key_id: str) -> bool:
        """Delete an API key permanently."""
        if key_id in self._keys:
            api_key = self._keys.pop(key_id)
            if api_key.key_hash in self._key_lookup:
                del self._key_lookup[api_key.key_hash]
            self._save_keys()
            return True
        return False
    
    def rotate_key(self, key_id: str) -> Tuple[str, APIKey]:
        """
        Rotate an API key (revoke old, create new).
        
        Returns:
            Tuple of (new_plaintext_key, new_APIKey)
        """
        old_key = self._keys.get(key_id)
        if old_key is None:
            raise ValueError(f"Key not found: {key_id}")
        
        self.revoke_key(key_id)
        
        new_key, new_api_key = self.create_key(
            name=old_key.name,
            tier=old_key.tier,
            user_id=old_key.user_id,
            rate_limit=old_key.rate_limit,
            quota_daily=old_key.quota_daily,
            quota_monthly=old_key.quota_monthly,
            metadata=old_key.metadata,
        )
        
        return new_key, new_api_key
    
    def reset_usage(self, key_id: str) -> bool:
        """Reset usage counters for a key."""
        if key_id in self._keys:
            self._keys[key_id].usage_today = 0
            self._keys[key_id].usage_this_month = 0
            self._save_keys()
            return True
        return False
    
    def update_tier(self, key_id: str, tier: KeyTier) -> bool:
        """Update the tier for a key."""
        if key_id in self._keys:
            self._keys[key_id].tier = tier
            self._save_keys()
            return True
        return False
    
    def set_quota(
        self,
        key_id: str,
        daily: Optional[int] = None,
        monthly: Optional[int] = None,
    ) -> bool:
        """Update quotas for a key."""
        if key_id in self._keys:
            if daily is not None:
                self._keys[key_id].quota_daily = daily
            if monthly is not None:
                self._keys[key_id].quota_monthly = monthly
            self._save_keys()
            return True
        return False
    
    def get_usage_stats(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a key."""
        api_key = self._keys.get(key_id)
        if api_key is None:
            return None
        
        return {
            "total_requests": api_key.usage_count,
            "requests_today": api_key.usage_today,
            "requests_this_month": api_key.usage_this_month,
            "daily_limit": api_key.quota_daily,
            "monthly_limit": api_key.quota_monthly,
            "daily_remaining": max(0, api_key.quota_daily - api_key.usage_today),
            "monthly_remaining": max(0, api_key.quota_monthly - api_key.usage_this_month),
            "daily_usage_percent": (api_key.usage_today / api_key.quota_daily * 100) if api_key.quota_daily else 0,
            "monthly_usage_percent": (api_key.usage_this_month / api_key.quota_monthly * 100) if api_key.quota_monthly else 0,
        }


class APIKeyMiddleware:
    """
    Middleware for validating API keys in requests.
    
    Example:
    
    ```python
    from sloughgpt_sdk.auth import APIKeyMiddleware, APIKeyManager
    
    manager = APIKeyManager()
    middleware = APIKeyMiddleware(manager)
    
    # In your FastAPI app:
    @app.middleware("http")
    async def validate_api_key(request, call_next):
        response = await middleware.validate_request(request, call_next)
        return response
    ```
    """
    
    def __init__(self, manager: APIKeyManager):
        """Initialize middleware."""
        self._manager = manager
    
    def extract_key(self, request) -> Optional[str]:
        """Extract API key from request headers."""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key
        
        return None
    
    async def validate_request(self, request, call_next):
        """Validate API key and process request."""
        key = self.extract_key(request)
        
        if not key:
            return {"error": "API key required", "status": 401}
        
        is_valid, reason, api_key = self._manager.validate_key(key)
        
        if not is_valid:
            return {"error": reason, "status": 401 if "Invalid" in reason else 429}
        
        response = await call_next(request)
        
        self._manager.record_usage(key)
        
        response.headers["X-RateLimit-Limit"] = str(api_key.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(
            api_key.rate_limit - api_key.usage_today
        )
        
        return response
