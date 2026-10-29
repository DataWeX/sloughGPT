"""
Tests for feedback system API endpoints.

All tests are marked ``slow`` because they import the full FastAPI
server (model load, lifespan startup, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "apps" / "api" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

_PKGS_DIR = _REPO_ROOT / "packages" / "core-py"
if str(_PKGS_DIR) not in sys.path:
    sys.path.insert(0, str(_PKGS_DIR))


pytestmark = pytest.mark.slow

TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture(scope="module")
def client():
    from main import app

    with TestClient(app) as test_client:
        yield app, test_client


def test_feedback_record(client):
    """Test recording feedback."""
    app, test_client = client

    response = test_client.post(
        "/feedback",
        json={
            "message_id": "test_msg_1",
            "user_message": "Hello, how are you?",
            "assistant_response": "I am doing well, thank you!",
            "rating": "thumbs_up",
            "session_id": "test_user_pytest",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"
    assert "feedback_id" in data


def test_feedback_record_thumbs_down(client):
    """Test recording thumbs down feedback."""
    app, test_client = client

    response = test_client.post(
        "/feedback",
        json={
            "message_id": "test_msg_2",
            "user_message": "What is 2+2?",
            "assistant_response": "5",
            "rating": "thumbs_down",
            "session_id": "test_user_pytest",
        },
    )
    assert response.status_code == 200


def test_feedback_invalid_rating(client):
    """Test that invalid rating is rejected."""
    app, test_client = client

    response = test_client.post(
        "/feedback",
        json={
            "message_id": "test_msg_3",
            "user_message": "Hello",
            "assistant_response": "Hi",
            "rating": "invalid",
        },
    )
    assert response.status_code == 422


def test_meta_weights_stats(client):
    """Test getting meta weights stats."""
    app, test_client = client

    response = test_client.get("/meta-weights/stats")
    assert response.status_code == 200
    data = response.json()
    assert "db_stats" in data["data"]
    assert "feedback_total" in data["data"]["db_stats"]


def test_user_adapters_list(client):
    """Test listing user adapters."""
    app, test_client = client

    response = test_client.get("/user-adapters")
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data["data"] or "adapters" in data["data"]


def test_user_adapters_aggregate(client):
    """Test aggregating user adapters."""
    app, test_client = client

    response = test_client.post(
        "/user-adapters/aggregate-best",
        json={"top_k": 5, "min_feedback_count": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_user_adapters_prune(client):
    """Test pruning user adapters."""
    app, test_client = client

    response = test_client.post(
        "/user-adapters/prune",
        json={"min_feedback_count": 1, "max_age_days": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["status"] == "pruned"


def test_workflow_status(client):
    """Test getting workflow status."""
    app, test_client = client

    response = test_client.get("/workflow/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data["data"]
    assert "stats" in data["data"]


def test_workflow_start_stop(client):
    """Test starting and stopping workflow."""
    app, test_client = client

    # Stop
    response = test_client.post("/workflow/stop")
    assert response.status_code == 200

    # Start
    response = test_client.post("/workflow/start", json={})
    assert response.status_code == 200


@pytest.mark.slow
def test_workflow_trigger_aggregate(client):
    """Test triggering workflow aggregate."""
    app, test_client = client

    response = test_client.post("/workflow/trigger/aggregate")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "aggregated"


def test_export_training(client):
    """Test exporting training data via /training/export-text."""
    app, test_client = client

    response = test_client.post(
        "/training/export-text",
        json={"max_pairs": 10},
    )
    assert response.status_code == 200
    data = response.json()
    assert "pairs_count" in data
    assert "filepath" in data


def test_export_training_invalid(client):
    """Test that invalid export params don't crash."""
    app, test_client = client

    response = test_client.post(
        "/training/export-text",
        json={"max_pairs": -1},
    )
    assert response.status_code == 200
