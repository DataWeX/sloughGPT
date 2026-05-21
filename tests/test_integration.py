"""
Integration tests for the API server.

Marked ``slow`` because these require a running server on ``localhost:8000``.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

pytestmark = pytest.mark.slow
import requests
import time
import json
from typing import Dict, Any, Optional


BASE_URL = os.environ.get("SLOUGHGPT_INTEGRATION_BASE_URL", "http://localhost:8000")
TIMEOUT = int(os.environ.get("SLOUGHGPT_INTEGRATION_TIMEOUT", "120"))
HEALTH_CHECK_TIMEOUT = int(os.environ.get("SLOUGHGPT_INTEGRATION_HEALTH_TIMEOUT", "30"))

_QUICK_GEN = {"max_new_tokens": 12}


@pytest.fixture(scope="module", autouse=True)
def _ensure_api_running():
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
            if response.status_code == 200:
                return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if i < max_retries - 1:
                time.sleep(2)
            else:
                pytest.skip("API server not running")


class TestHealthEndpoints:
    """Integration tests for health check endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test basic health check."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_liveness(self):
        """Test liveness probe."""
        response = requests.get(f"{BASE_URL}/health/live", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "alive"

    def test_health_readiness(self):
        """Test readiness probe."""
        response = requests.get(f"{BASE_URL}/health/ready", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_info_endpoint(self):
        """Test system info endpoint."""
        response = requests.get(f"{BASE_URL}/info", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "version" in data or "api_version" in data or "model" in data


class TestGenerationEndpoints:
    """Integration tests for text generation endpoints."""

    def test_generate_endpoint(self):
        """Test basic text generation via /inference/generate."""
        payload = {"prompt": "Hello, how are you?", **_QUICK_GEN}
        response = requests.post(
            f"{BASE_URL}/inference/generate",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data

    def test_generate_with_params(self):
        """Test generation with custom parameters."""
        payload = {
            "prompt": "Tell me a joke",
            "max_new_tokens": 16,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        response = requests.post(
            f"{BASE_URL}/inference/generate",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200

    def test_generate_empty_prompt(self):
        """Test that empty prompt is handled gracefully."""
        payload = {"prompt": ""}
        response = requests.post(
            f"{BASE_URL}/inference/generate",
            json=payload,
            timeout=TIMEOUT
        )
        # Server accepts empty prompt and returns 200
        assert response.status_code == 200

    def test_generate_stream_endpoint(self):
        """Test streaming generation endpoint."""
        payload = {"prompt": "Count to 3:", **_QUICK_GEN}
        response = requests.post(
            f"{BASE_URL}/inference/generate/stream",
            json=payload,
            stream=True,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("text/event-stream")

    def test_chat_endpoint(self):
        """Test chat endpoint."""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            **_QUICK_GEN,
        }
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestModelEndpoints:
    """Integration tests for model management endpoints."""

    def test_list_models(self):
        """Test listing available models."""
        response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "models" in data or isinstance(data, list)

    def test_list_huggingface_models(self):
        """Test listing HuggingFace models."""
        response = requests.get(f"{BASE_URL}/models/hf", timeout=TIMEOUT)
        assert response.status_code == 200

    def test_model_info(self):
        """Test getting specific model info."""
        response = requests.get(f"{BASE_URL}/models/gpt2", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            assert "id" in data or "name" in data


class TestDatasetEndpoints:
    """Integration tests for dataset management endpoints."""

    def test_list_datasets(self):
        """Test listing available datasets."""
        response = requests.get(f"{BASE_URL}/datasets", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data or isinstance(data, list)

    def test_dataset_info(self):
        """Test getting specific dataset info."""
        response = requests.get(f"{BASE_URL}/datasets/openwebtext", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            assert "id" in data or "name" in data


class TestMetricsEndpoints:
    """Integration tests for metrics and monitoring endpoints."""

    def test_metrics_json(self):
        """Test metrics endpoint in JSON format."""
        response = requests.get(f"{BASE_URL}/metrics", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "uptime" in data or "requests_total" in data or "metrics" in data

    def test_metrics_prometheus(self):
        """Test metrics endpoint in Prometheus format."""
        response = requests.get(
            f"{BASE_URL}/metrics/prometheus",
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestAuthentication:
    """Integration tests for authentication endpoints."""

    def test_login_endpoint(self):
        """Test JWT token endpoint (API key)."""
        payload = {"api_key": "invalid-integration-test-key"}
        response = requests.post(
            f"{BASE_URL}/auth/token",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 401]

    def test_token_refresh(self):
        """Test token refresh endpoint."""
        response = requests.post(
            f"{BASE_URL}/auth/refresh",
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 401]


class TestPerformance:
    """Integration tests for performance and rate limiting."""

    def test_response_time(self):
        """Test that basic endpoints respond within acceptable time."""
        start = time.time()
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 10.0, f"Health check took {elapsed:.2f}s, expected < 10s"

    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import concurrent.futures

        def make_request():
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
                return response.status_code == 200
            except Exception:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success_count = sum(results)
        assert success_count >= 4, f"Only {success_count}/5 requests succeeded"


class TestErrorHandling:
    """Integration tests for error handling."""

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = requests.post(
            f"{BASE_URL}/inference/generate",
            data="not valid json",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert response.status_code == 422

    def test_missing_required_field(self):
        """Test handling of missing required fields."""
        payload = {"max_new_tokens": 100}
        response = requests.post(
            f"{BASE_URL}/inference/generate",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 422

    def test_invalid_field_type(self):
        """Test handling of invalid field types."""
        payload = {"prompt": 12345}
        response = requests.post(
            f"{BASE_URL}/inference/generate",
            json=payload,
            timeout=TIMEOUT
        )
        assert response.status_code == 422

    def test_404_not_found(self):
        """Test handling of non-existent endpoints."""
        response = requests.get(f"{BASE_URL}/nonexistent", timeout=TIMEOUT)
        assert response.status_code == 404


class TestCache:
    """Integration tests for caching functionality."""

    def test_cache_headers(self):
        """Test that responses include cache headers."""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        headers = response.headers
        assert "date" in headers

    def test_identical_requests(self):
        """Test that identical requests work consistently."""
        payload = {"prompt": "Cache test", **_QUICK_GEN}
        response1 = requests.post(f"{BASE_URL}/inference/generate", json=payload, timeout=TIMEOUT)
        response2 = requests.post(f"{BASE_URL}/inference/generate", json=payload, timeout=TIMEOUT)
        assert response1.status_code == 200
        assert response2.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
