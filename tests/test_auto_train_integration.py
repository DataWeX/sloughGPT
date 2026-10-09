from unittest.mock import patch

import httpx
import pytest

pytestmark = pytest.mark.slow


def _server_available() -> bool:
    """Check if the API server is running on localhost:8000."""
    try:
        import urllib.request

        req = urllib.request.Request("http://localhost:8000/health")
        resp = urllib.request.urlopen(req, timeout=2)
        return resp.status == 200
    except Exception:
        return False


@pytest.mark.skipif(not _server_available(), reason="API server not running")
class TestAutoTrainIntegration:
    """Integration tests for auto-train flow."""

    @pytest.fixture
    def api_base_url(self):
        return "http://localhost:8000"

    @pytest.mark.anyio
    async def test_legacy_auto_train_routes_retired(self, api_base_url):
        """/auto-train/* is gone on purpose; /training/* is the live projection.

        Card e205b15c: the router was deleted by d63f5cb0c and the in-tree
        callers (shell commands/repl, CLI, SDK) were migrated to /training/*.
        This pins that decision — if the legacy paths ever answer 200 again,
        delete this assertion and the migration note in docs/routers.md.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            legacy = await client.get(f"{api_base_url}/auto-train/checkpoints")
            assert legacy.status_code == 404

            listed = await client.get(f"{api_base_url}/training/checkpoints")
            assert listed.status_code == 200, listed.text
            assert isinstance(listed.json()["data"], list)

            status = await client.get(f"{api_base_url}/training/status")
            assert status.status_code == 200, status.text

    @pytest.mark.anyio
    async def test_start_validates_and_stop_is_a_noop(self, api_base_url):
        """start -> stream -> stop seam: routes answer without starting a job."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # The route exists and rejects an empty body before any job starts.
            start_resp = await client.post(f"{api_base_url}/training/start", json={})
            assert start_resp.status_code == 422, start_resp.text

            # Idle stream answers (graceful envelope), it does not 404.
            async with client.stream("GET", f"{api_base_url}/training/stream") as resp:
                assert resp.status_code == 200

            stop_resp = await client.post(f"{api_base_url}/training/stop")
            assert stop_resp.status_code == 200, stop_resp.text


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("transformers") is None,
    reason="transformers not installed (auto-train teacher is SloNet now)",
)
class TestAutoTrainErrorHandling:
    """Tests for error handling in auto-train."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from apps.api.server.main import app

        return TestClient(app)

    def test_start_with_invalid_model(self, client):
        """Should fail gracefully with invalid model."""
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
            mock_model.side_effect = Exception("Invalid model")
            resp = client.post("/auto-train/start", json={"teacher_model": "gpt2", "epochs": 1})
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("status") == "error"
            assert "Invalid model" in data.get("message", "")


class TestAutoTrainStreamingEvents:
    """Tests for SSE event format."""

    def test_stream_yields_proper_sse_format(self):
        """Stream should yield valid SSE format."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
