"""
Tests for Mobile BFF router endpoints.

Tests all /mobile/* endpoints to ensure proper aggregation and response formatting.
Mocks the static helper methods on MobileRouter instead of httpx.
"""

from unittest.mock import patch

import pytest
from test_support import get_test_client


@pytest.fixture
def client():
    """Create test client."""
    return get_test_client()


def resp_data(response):
    """Unwrap StandardResponse to get inner data."""
    return response.json()["data"]


class TestMobileDashboard:
    """Tests for GET /mobile/dashboard."""

    def test_dashboard_returns_aggregated_data(self, client):
        """Dashboard should aggregate health, soul, sessions, and models."""
        mock_health = {
            "status": "healthy",
            "model_loaded": True,
            "model_type": "gpt2",
            "inference_count": 42,
        }
        mock_soul = {"name": "Default", "description": "Default personality"}
        mock_sessions = [
            {
                "id": "session_1",
                "name": "Test Chat",
                "title": "Test Chat",
                "updated_at": "2024-01-01T00:00:00",
                "messages": [{"content": "Hello"}],
            }
        ]
        mock_models = [{"id": "gpt2", "name": "GPT-2"}]

        with (
            patch("routers.mobile.MobileRouter._get_health_data", return_value=mock_health),
            patch("routers.mobile.MobileRouter._get_current_soul", return_value=mock_soul),
            patch("routers.mobile.MobileRouter._get_sessions_list", return_value=mock_sessions),
            patch("routers.mobile.MobileRouter._get_models_list", return_value=mock_models),
        ):
            response = client.get("/mobile/dashboard")

        assert response.status_code == 200
        data = resp_data(response)
        assert data["status"] == "healthy"
        assert data["model"]["name"] == "gpt2"
        assert data["model"]["loaded"] is True
        assert data["soul"]["name"] == "Default"
        assert len(data["recent_conversations"]) == 1
        assert data["stats"]["model_count"] == 1
        assert data["stats"]["inference_count"] == 42

    def test_dashboard_handles_missing_data(self, client):
        """Dashboard should handle empty/missing data gracefully."""
        with (
            patch("routers.mobile.MobileRouter._get_health_data", return_value={}),
            patch("routers.mobile.MobileRouter._get_current_soul", return_value={}),
            patch("routers.mobile.MobileRouter._get_sessions_list", return_value=[]),
            patch("routers.mobile.MobileRouter._get_models_list", return_value=[]),
        ):
            response = client.get("/mobile/dashboard")

        assert response.status_code == 200
        data = resp_data(response)
        assert data["status"] == "unknown"


class TestMobileConversations:
    """Tests for /mobile/conversations endpoints."""

    def test_list_conversations_paginated(self, client):
        """GET /mobile/conversations should return paginated sessions."""
        sessions = [
            {
                "id": f"session_{i}",
                "name": f"Chat {i}",
                "title": f"Chat {i}",
                "updated_at": f"2024-01-0{i}T00:00:00",
                "messages": [{"content": f"Message {i}"}],
            }
            for i in range(1, 26)
        ]

        with patch("routers.mobile.MobileRouter._get_sessions_list", return_value=sessions):
            response = client.get("/mobile/conversations?page=1&per_page=10")

        assert response.status_code == 200
        data = resp_data(response)
        assert len(data["conversations"]) == 10
        assert data["total"] == 25
        assert data["page"] == 1
        assert data["per_page"] == 10

    def test_list_conversations_with_search(self, client):
        """GET /mobile/conversations should filter by search term."""
        sessions = [
            {
                "id": "session_1",
                "name": "Python Help",
                "title": "Python Help",
                "updated_at": "2024-01-01T00:00:00",
            },
            {
                "id": "session_2",
                "name": "JavaScript Help",
                "title": "JavaScript Help",
                "updated_at": "2024-01-02T00:00:00",
            },
        ]

        with patch("routers.mobile.MobileRouter._get_sessions_list", return_value=sessions):
            response = client.get("/mobile/conversations?search=Python")

        assert response.status_code == 200
        data = resp_data(response)
        assert len(data["conversations"]) == 1
        assert data["conversations"][0]["title"] == "Python Help"

    def test_get_conversation_detail(self, client):
        """GET /mobile/conversations/{id} should return full conversation."""
        messages = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00"},
            {"role": "assistant", "content": "Hi!", "timestamp": "2024-01-01T00:00:01"},
        ]

        with patch("routers.mobile.MobileRouter._get_session_messages", return_value=messages):
            response = client.get("/mobile/conversations/session_123")

        assert response.status_code == 200
        data = resp_data(response)
        assert data["id"] == "session_123"
        assert len(data["messages"]) == 2


class TestMobileModels:
    """Tests for /mobile/models endpoints."""

    def test_get_models_aggregates_all_data(self, client):
        """GET /mobile/models should aggregate models, souls, and checkpoints."""
        mock_models = [
            {"model_id": "gpt2", "name": "GPT-2", "loaded": True, "size_gb": 0.5},
            {"model_id": "llama", "name": "LLaMA", "loaded": False, "size_gb": 7.0},
        ]
        mock_souls = [
            {"name": "Default", "description": "Default", "traits": []},
            {"name": "Creative", "description": "Creative", "traits": ["creative"]},
        ]
        mock_current_soul = {"name": "Default"}
        mock_checkpoints = [{"name": "checkpoint_1", "soul": "Default", "loss": 0.5, "steps": 100}]
        mock_health = {"model_type": "gpt2"}

        with (
            patch("routers.mobile.MobileRouter._get_models_list", return_value=mock_models),
            patch("routers.mobile.MobileRouter._get_souls", return_value=mock_souls),
            patch("routers.mobile.MobileRouter._get_current_soul", return_value=mock_current_soul),
            patch("routers.mobile.MobileRouter._get_checkpoints", return_value=mock_checkpoints),
            patch("routers.mobile.MobileRouter._get_health_data", return_value=mock_health),
        ):
            response = client.get("/mobile/models")

        assert response.status_code == 200
        data = resp_data(response)
        assert data["current"]["model_id"] == "gpt2"
        assert data["current"]["soul"] == "Default"
        assert len(data["models"]) == 2
        assert len(data["souls"]) == 2
        assert len(data["checkpoints"]) == 1

    def test_switch_model_and_soul(self, client):
        """POST /mobile/models/switch should load model and switch soul."""
        mock_health = {"model_type": "llama"}

        with (
            patch("routers.mobile.MobileRouter._load_model", return_value=None),
            patch("routers.mobile.MobileRouter._switch_soul", return_value=None),
            patch("routers.mobile.MobileRouter._get_health_data", return_value=mock_health),
        ):
            response = client.post(
                "/mobile/models/switch",
                json={"model_id": "llama", "soul_name": "Creative", "checkpoint_name": "cp_1"},
            )

        assert response.status_code == 200
        data = resp_data(response)
        assert data["status"] == "ok"
        assert data["model"] == "llama"
        assert data["soul"] == "Creative"
        assert data["checkpoint"] == "cp_1"


class TestMobileHealth:
    """Tests for GET /mobile/health."""

    def test_health_returns_system_summary(self, client):
        """GET /mobile/health should aggregate detailed health and metrics."""
        mock_detailed = {
            "status": "healthy",
            "model_loaded": True,
            "model_type": "gpt2",
            "uptime_seconds": 3600,
            "system": {
                "cpu_percent": 25.5,
                "memory_percent": 60.0,
                "memory_available_mb": 8192,
            },
            "inference": {"inference_count": 100},
        }
        mock_metrics = {
            "cpu_percent": 25.5,
            "memory_percent": 60.0,
        }
        mock_disk = {
            "used_gb": 46.57,
            "free_gb": 93.13,
        }

        with (
            patch("routers.mobile.MobileRouter._get_detailed_health", return_value=mock_detailed),
            patch("routers.mobile.MobileRouter._get_system_metrics", return_value=mock_metrics),
            patch("routers.mobile.MobileRouter._get_disk_info", return_value=mock_disk),
        ):
            response = client.get("/mobile/health")

        assert response.status_code == 200
        data = resp_data(response)
        assert data["status"] == "healthy"
        assert data["model"]["name"] == "gpt2"
        assert data["model"]["loaded"] is True
        assert data["uptime_seconds"] == 3600
        assert data["cpu_percent"] == 25.5
        assert data["memory_percent"] == 60.0
        assert data["memory_available_gb"] == 8.0
        assert data["inference_count"] == 100


class TestMobileKnowledge:
    """Tests for /mobile/knowledge endpoints."""

    def test_list_knowledge_paginated(self, client):
        """GET /mobile/knowledge should return paginated items."""
        with patch(
            "routers.mobile.MobileRouter._get_knowledge_items",
            return_value=[
                {
                    "id": f"item_{i}",
                    "content": f"Knowledge {i}",
                    "topic": "tech",
                    "importance": 0.8,
                    "source": "manual",
                    "url": "",
                    "timestamp": 0,
                    "score": 0,
                }
                for i in range(1, 51)
            ],
        ):
            response = client.get("/mobile/knowledge?page=1&per_page=20")

        assert response.status_code == 200
        data = resp_data(response)
        assert len(data["items"]) == 20
        assert data["total"] == 50
        assert data["page"] == 1

    def test_list_knowledge_with_topic_filter(self, client):
        """GET /mobile/knowledge should filter by topic."""
        items = [
            {
                "id": "1",
                "content": "Tech 1",
                "topic": "tech",
                "importance": 0.8,
                "source": "manual",
                "url": "",
                "timestamp": 0,
                "score": 0,
            },
            {
                "id": "2",
                "content": "Science 1",
                "topic": "science",
                "importance": 0.9,
                "source": "manual",
                "url": "",
                "timestamp": 0,
                "score": 0,
            },
            {
                "id": "3",
                "content": "Tech 2",
                "topic": "tech",
                "importance": 0.7,
                "source": "manual",
                "url": "",
                "timestamp": 0,
                "score": 0,
            },
        ]

        with patch("routers.mobile.MobileRouter._get_knowledge_items", return_value=items):
            response = client.get("/mobile/knowledge?topic=tech")

        assert response.status_code == 200
        data = resp_data(response)
        assert len(data["items"]) == 2
        assert all(item["topic"] == "tech" for item in data["items"])

    def test_list_knowledge_with_search(self, client):
        """GET /mobile/knowledge should use search when search param provided."""
        results = [
            {
                "id": "1",
                "content": "Python programming",
                "topic": "tech",
                "importance": 0.9,
                "source": "manual",
                "score": 0.95,
            },
        ]

        with patch("routers.mobile.MobileRouter._search_knowledge", return_value=results):
            response = client.get("/mobile/knowledge?search=Python")

        assert response.status_code == 200
        data = resp_data(response)
        assert len(data["items"]) == 1
        assert "Python" in data["items"][0]["content"]

    def test_create_knowledge_item(self, client):
        """POST /mobile/knowledge should create a new item."""
        with patch("routers.mobile.MobileRouter._create_knowledge_item", return_value="new_item"):
            response = client.post(
                "/mobile/knowledge",
                json={"content": "New knowledge", "topic": "tech"},
            )

        assert response.status_code == 200
        data = resp_data(response)
        assert data["content"] == "New knowledge"
        assert data["topic"] == "tech"
        assert data["id"] == "new_item"

    def test_update_knowledge_item(self, client):
        """PATCH /mobile/knowledge/{id} should update an item."""
        with patch("routers.mobile.MobileRouter._update_knowledge_item", return_value=True):
            response = client.patch(
                "/mobile/knowledge/item_1",
                json={"content": "Updated content", "topic": "science", "importance": 0.9},
            )

        assert response.status_code == 200
        data = resp_data(response)
        assert data["updated"] is True
        assert data["id"] == "item_1"

    def test_delete_knowledge_item(self, client):
        """DELETE /mobile/knowledge/{id} should delete an item."""
        with patch("routers.mobile.MobileRouter._delete_knowledge_item", return_value=True):
            response = client.delete("/mobile/knowledge/item_1")

        assert response.status_code == 200
        data = resp_data(response)
        assert data["status"] == "deleted"
        assert data["id"] == "item_1"
