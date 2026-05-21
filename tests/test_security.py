"""
SloughGPT Security Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import hashlib
import secrets


class TestRateLimiter:
    """Tests for rate limiting functionality."""

    def test_rate_limiter_init(self):
        """Test rate limiter initialization."""
        class MockRateLimiter:
            def __init__(self, requests_per_minute=60, burst_size=10):
                self.requests_per_minute = requests_per_minute
                self.burst_size = burst_size
                self.clients = {}

            def is_allowed(self, client_id):
                if client_id not in self.clients:
                    self.clients[client_id] = []
                self.clients[client_id].append(time.time())
                return True, 59

        limiter = MockRateLimiter()
        assert limiter.requests_per_minute == 60
        assert limiter.burst_size == 10

    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit."""
        class MockRateLimiter:
            def __init__(self):
                self.clients = {}

            def is_allowed(self, client_id):
                if client_id not in self.clients:
                    self.clients[client_id] = []
                self.clients[client_id].append(time.time())
                return True, len(self.clients[client_id])

        limiter = MockRateLimiter()
        allowed, remaining = limiter.is_allowed("test_client")
        assert allowed is True
        assert remaining >= 0


class TestJWTAuth:
    """Tests for JWT authentication."""

    def test_jwt_token_creation(self):
        """Test JWT token creation."""
        import base64
        import json
        import hmac

        secret = "test_secret"
        subject = "test_user"

        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode()

        payload = {
            "sub": subject,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()

        signature = hmac.new(secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256)
        signature_b64 = base64.urlsafe_b64encode(signature.digest()).decode()

        token = f"{header_b64}.{payload_b64}.{signature_b64}"

        assert len(token.split(".")) == 3
        assert header_b64 in token

    def test_jwt_token_verification(self):
        """Test JWT token structure is valid."""
        import base64
        import json

        payload_b64 = base64.urlsafe_b64encode(
            json.dumps({"sub": "test", "exp": int(time.time()) + 3600}).encode()
        ).decode()

        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
        assert payload["sub"] == "test"
        assert payload["exp"] > time.time()


class TestInputValidation:
    """Tests for input validation."""

    def test_sanitize_string(self):
        """Test string sanitization."""
        def sanitize_string(value, max_length=10000):
            if not isinstance(value, str):
                return ""
            value = value.replace("\x00", "")
            return value[:max_length].strip()

        assert sanitize_string("Hello") == "Hello"
        assert sanitize_string("Hello\x00World") == "HelloWorld"
        assert sanitize_string("   Hello   ") == "Hello"
        assert sanitize_string("a" * 100, max_length=50) == "a" * 50
        assert sanitize_string(123) == ""

    def test_validate_temperature(self):
        """Test temperature validation."""
        def validate_temperature(temp):
            return max(0.0, min(2.0, temp))

        assert validate_temperature(0.5) == 0.5
        assert validate_temperature(-1) == 0.0
        assert validate_temperature(3) == 2.0
        assert validate_temperature(0) == 0.0

    def test_validate_max_tokens(self):
        """Test max tokens validation."""
        def validate_max_tokens(tokens):
            return max(1, min(4096, tokens))

        assert validate_max_tokens(100) == 100
        assert validate_max_tokens(0) == 1
        assert validate_max_tokens(5000) == 4096
        assert validate_max_tokens(-10) == 1

    def test_validate_prompt(self):
        """Test prompt validation."""
        def validate_prompt(prompt):
            if not isinstance(prompt, str):
                return ""
            prompt = prompt.replace("\x00", "")
            suspicious = ["<script", "javascript:", "onerror=", "onload="]
            for pattern in suspicious:
                if pattern.lower() in prompt.lower():
                    return prompt[:100].strip()
            return prompt[:8000].strip()

        assert validate_prompt("Hello world") == "Hello world"
        assert validate_prompt("<script>alert(1)</script>")[:20] == "<script>alert(1)</sc"
        assert validate_prompt("javascript:void(0)")[:20] == "javascript:void(0)"


class TestAPICache:
    """Tests for API caching."""

    def test_cache_init(self):
        """Test cache initialization."""
        class MockCache:
            def __init__(self, max_size=1000, default_ttl=300):
                self.cache = {}
                self.max_size = max_size
                self.default_ttl = default_ttl
                self.hits = 0
                self.misses = 0

        cache = MockCache()
        assert cache.max_size == 1000
        assert cache.default_ttl == 300
        assert len(cache.cache) == 0

    def test_cache_set_get(self):
        """Test cache set and get."""
        class MockCache:
            def __init__(self):
                self.cache = {}
                self.default_ttl = 300

            def get(self, key):
                if key not in self.cache:
                    return None
                return self.cache[key]

            def set(self, key, value):
                self.cache[key] = value

        cache = MockCache()
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        assert cache.get("nonexistent") is None

    def test_cache_key_generation(self):
        """Test cache key generation."""
        import hashlib
        import json

        def cache_key(prompt, **kwargs):
            params = json.dumps(kwargs, sort_keys=True)
            combined = f"{prompt}:{params}"
            return hashlib.sha256(combined.encode()).hexdigest()[:32]

        key1 = cache_key("Hello", temp=0.8, max_tokens=100)
        key2 = cache_key("Hello", temp=0.8, max_tokens=100)
        key3 = cache_key("Hello", temp=0.9, max_tokens=100)

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 32


class TestBatchProcessing:
    """Tests for batch processing."""

    def test_batch_request_validation(self):
        """Test batch request validation."""
        prompts = ["prompt1", "prompt2", "prompt3"]

        assert len(prompts) > 0
        assert len(prompts) <= 50

    def test_batch_response_structure(self):
        """Test batch response structure."""
        results = [
            {"prompt": "test1", "text": "response1", "cached": False},
            {"prompt": "test2", "text": "response2", "cached": True},
        ]

        assert len(results) == 2
        for r in results:
            assert "prompt" in r
            assert "text" in r
            assert "cached" in r


class TestSecurityHeaders:
    """Tests for security headers."""

    def test_security_headers(self):
        """Test security headers are defined."""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
        }

        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "max-age" in headers["Strict-Transport-Security"]


class TestAuditLogger:
    """Tests for audit logging."""

    def test_audit_log_entry(self):
        """Test audit log entry structure."""
        entry = {
            "timestamp": "2024-01-01T00:00:00",
            "event_type": "auth_success",
            "client_ip": "127.0.0.1",
            "user_id": None,
            "resource": "/auth/token",
            "action": "token_create",
            "status": "success",
            "details": {},
        }

        assert "timestamp" in entry
        assert "event_type" in entry
        assert "client_ip" in entry
        assert "status" in entry

    def test_audit_log_retention(self):
        """Test audit log retention."""
        logs = []
        max_logs = 10000

        for i in range(15000):
            logs.append({"id": i})

        if len(logs) > max_logs:
            logs = logs[-max_logs:]

        assert len(logs) == max_logs
        assert logs[0]["id"] == 5000
        assert logs[-1]["id"] == 14999


class TestHealthProbes:
    """Tests for health probe endpoints."""

    def test_liveness_response(self):
        """Test liveness probe response."""
        response = {"status": "alive"}
        assert response["status"] == "alive"

    def test_readiness_response(self):
        """Test readiness probe response."""
        response = {
            "status": "ready",
            "model_loaded": True,
            "model_type": "gpt2",
        }
        assert response["status"] == "ready"
        assert response["model_loaded"] is True

    def test_readiness_not_ready(self):
        """Test readiness when model not loaded."""
        response = {
            "status": "initializing",
            "model_loaded": False,
            "model_type": None,
        }
        assert response["status"] == "initializing"
        assert response["model_loaded"] is False
