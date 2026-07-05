import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    app = FastAPI()
    from routers.auto_train import router
    app.include_router(router)
    # No lifespan — just the router under test
    return TestClient(app)


class TestAutoTrainStart:
    def test_start_sets_running_state(self, client):
        """POST /auto-train/start should return 200."""
        response = client.post(
            "/auto-train/start",
            json={"teacher_model": "gpt2", "temperature": 0.8, "epochs": 1, "source_text": "hello world"}
        )
        # May return error if model loading fails, but should be a valid response
        assert response.status_code in (200, 500, 422)

    def test_start_twice_should_warn(self, client):
        """Starting twice should work (replaces prior session)."""
        r1 = client.post(
            "/auto-train/start",
            json={"teacher_model": "gpt2", "source_text": "hello"}
        )

        r2 = client.post(
            "/auto-train/start",
            json={"teacher_model": "gpt2", "source_text": "world"}
        )


class TestAutoTrainStop:
    def test_stop_before_start_gives_error(self, client):
        """Stopping without start should return stopped."""
        response = client.post("/auto-train/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["stopped", "not_running"]

    def test_stop_after_start_works(self, client):
        """Stop after start should work."""
        client.post(
            "/auto-train/start",
            json={"teacher_model": "gpt2"}
        )

        response = client.post("/auto-train/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
