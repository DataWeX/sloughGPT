"""
Tests for Mobile BFF router endpoints.

Tests all /mobile/* endpoints to ensure proper aggregation and response formatting.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from test_support import get_test_client


@pytest.fixture
def client():
    """Create test client."""
    return get_test_client()


@pytest.fixture
def mock_httpx():
    """Mock httpx.AsyncClient for internal API calls."""
    with patch("apps.api.server.routers.mobile.httpx.AsyncClient") as mock:
        mock_client = AsyncMock()
        mock.return_value.__aenter__.return_value = mock_client
        yield mock_client


class TestMobileDashboard:
    """Tests for GET /mobile/dashboard."""

    def test_dashboard_returns_aggregated_data(self, client, mock_httpx):
        """Dashboard should aggregate health, soul, sessions, and models."""
        # Mock responses
        mock_httpx.get.side_effect = [
            # /health
            MagicMock(
                status_code=200,
                json=lambda: {
                    "status": "healthy",
                    "model_loaded": True,
                    "model_type": "gpt2",
                    "inference_count": 42,
                },
            ),
            # /souls/current
            MagicMock(
                status_code=200,
                json=lambda: {"name": "Default", "description": "Default personality"},
            ),
            # /chat/sessions
            MagicMock(
                status_code=200,
                json=lambda: {
                    "sessions": [
                        {
                            "id": "session_1",
                            "title": "Test Chat",
                            "messages": [{"content": "Hello"}],
                            "updated_at": "2024-01-01T00:00:00",
                        }
                    ]
                },
            ),
            # /models
            MagicMock(
                status_code=200,
                json=lambda: [{"id": "gpt2", "name": "GPT-2"}],
            ),
        ]

        response = client.get("/mobile/dashboard")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model"]["name"] == "gpt2"
        assert data["model"]["loaded"] is True
        assert data["soul"]["name"] == "Default"
        assert len(data["recent_conversations"]) == 1
        assert data["stats"]["model_count"] == 1
        assert data["stats"]["inference_count"] == 42

    def test_dashboard_handles_missing_data(self, client, mock_httpx):
        """Dashboard should handle failed internal calls gracefully."""
        mock_httpx.get.side_effect = [
            MagicMock(status_code=500),  # /health fails
            MagicMock(status_code=200, json=lambda: {}),
            MagicMock(status_code=200, json=lambda: {"sessions": []}),
            MagicMock(status_code=200, json=lambda: []),
        ]

        response = client.get("/mobile/dashboard")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unknown"


class TestMobileConversations:
    """Tests for /mobile/conversations endpoints."""

    def test_list_conversations_paginated(self, client, mock_httpx):
        """GET /mobile/conversations should return paginated sessions."""
        mock_httpx.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "sessions": [
                    {
                        "id": f"session_{i}",
                        "title": f"Chat {i}",
                        "messages": [{"content": f"Message {i}"}],
                        "updated_at": f"2024-01-0{i}T00:00:00",
                    }
                    for i in range(1, 26)  # 25 sessions
                ]
            },
        )

        response = client.get("/mobile/conversations?page=1&per_page=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data["conversations"]) == 10
        assert data["total"] == 25
        assert data["page"] == 1
        assert data["per_page"] == 10

    def test_list_conversations_with_search(self, client, mock_httpx):
        """GET /mobile/conversations should filter by search term."""
        mock_httpx.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "sessions": [
                    {
                        "id": "session_1",
                        "title": "Python Help",
                        "messages": [{"content": "How do I use Python?"}],
                        "updated_at": "2024-01-01T00:00:00",
                    },
                    {
                        "id": "session_2",
                        "title": "JavaScript Help",
                        "messages": [{"content": "How do I use JavaScript?"}],
                        "updated_at": "2024-01-02T00:00:00",
                    },
                ]
            },
        )

        response = client.get("/mobile/conversations?search=Python")

        assert response.status_code == 200
        data = response.json()
        assert len(data["conversations"]) == 1
        assert data["conversations"][0]["title"] == "Python Help"

    def test_get_conversation_detail(self, client, mock_httpx):
        """GET /mobile/conversations/{id} should return full conversation."""
        mock_httpx.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "messages": [
                    {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00"},
                    {"role": "assistant", "content": "Hi!", "timestamp": "2024-01-01T00:00:01"},
                ],
                "created_at": "2024-01-01T00:00:00",
            },
        )

        response = client.get("/mobile/conversations/session_123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "session_123"
        assert len(data["messages"]) == 2


class TestMobileModels:
    """Tests for /mobile/models endpoints."""

    def test_get_models_aggregates_all_data(self, client, mock_httpx):
        """GET /mobile/models should aggregate models, souls, and checkpoints."""
        mock_httpx.get.side_effect = [
            # /models
            MagicMock(
                status_code=200,
                json=lambda: [
                    {"model_id": "gpt2", "name": "GPT-2", "status": "loaded", "size_gb": 0.5},
                    {"model_id": "llama", "name": "LLaMA", "status": "available", "size_gb": 7.0},
                ],
            ),
            # /souls
            MagicMock(
                status_code=200,
                json=lambda: [
                    {"name": "Default", "description": "Default", "traits": []},
                    {"name": "Creative", "description": "Creative", "traits": ["creative"]},
                ],
            ),
            # /souls/current
            MagicMock(status_code=200, json=lambda: {"name": "Default"}),
            # /auto-train/checkpoints
            MagicMock(
                status_code=200,
                json=lambda: [{"name": "checkpoint_1", "soul": "Default", "loss": 0.5, "steps": 100}],
            ),
            # /health
            MagicMock(status_code=200, json=lambda: {"model_type": "gpt2"}),
        ]

        response = client.get("/mobile/models")

        assert response.status_code == 200
        data = response.json()
        assert data["current"]["model_id"] == "gpt2"
        assert data["current"]["soul"] == "Default"
        assert len(data["models"]) == 2
        assert len(data["souls"]) == 2
        assert len(data["checkpoints"]) == 1

    def test_switch_model_and_soul(self, client, mock_httpx):
        """POST /mobile/models/switch should load model and switch soul."""
        mock_httpx.post.side_effect = [
            # /models/load
            MagicMock(status_code=200, json=lambda: {"status": "loaded"}),
            # /souls/switch
            MagicMock(status_code=200, json=lambda: {"status": "ok"}),
        ]
        mock_httpx.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"model_type": "llama", "status": "healthy"},
        )

        response = client.post(
            "/mobile/models/switch",
            json={"model_id": "llama", "soul_name": "Creative", "checkpoint_name": "cp_1"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"] == "llama"
        assert data["soul"] == "Creative"
        assert data["checkpoint"] == "cp_1"


class TestMobileHealth:
    """Tests for GET /mobile/health."""

    def test_health_returns_system_summary(self, client, mock_httpx):
        """GET /mobile/health should aggregate detailed health and metrics."""
        mock_httpx.get.side_effect = [
            # /health/detailed
            MagicMock(
                status_code=200,
                json=lambda: {
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
                },
            ),
            # /system/metrics
            MagicMock(
                status_code=200,
                json=lambda: {
                    "cpu_percent": 25.5,
                    "memory_percent": 60.0,
                    "disk_used_bytes": 50_000_000_000,
                    "disk_free_bytes": 100_000_000_000,
                },
            ),
        ]

        response = client.get("/mobile/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model"]["name"] == "gpt2"
        assert data["model"]["loaded"] is True
        assert data["uptime_seconds"] == 3600
        assert data["cpu_percent"] == 25.5
        assert data["memory_percent"] == 60.0
        assert data["memory_available_gb"] == 8.0  # 8192 MB / 1024
        assert data["inference_count"] == 100


class TestMobileKnowledge:
    """Tests for /mobile/knowledge endpoints."""

    def test_list_knowledge_paginated(self, client, mock_httpx):
        """GET /mobile/knowledge should return paginated items."""
        mock_httpx.get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {
                    "id": f"item_{i}",
                    "content": f"Knowledge {i}",
                    "topic": "tech",
                    "importance": 0.8,
                    "source": "manual",
                }
                for i in range(1, 51)  # 50 items
            ],
        )

        response = client.get("/mobile/knowledge?page=1&per_page=20")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 20
        assert data["total"] == 50
        assert data["page"] == 1

    def test_list_knowledge_with_topic_filter(self, client, mock_httpx):
        """GET /mobile/knowledge should filter by topic."""
        mock_httpx.get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {"id": "1", "content": "Tech 1", "topic": "tech", "importance": 0.8, "source": "manual"},
                {"id": "2", "content": "Science 1", "topic": "science", "importance": 0.9, "source": "manual"},
                {"id": "3", "content": "Tech 2", "topic": "tech", "importance": 0.7, "source": "manual"},
            ],
        )

        response = client.get("/mobile/knowledge?topic=tech")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert all(item["topic"] == "tech" for item in data["items"])

    def test_list_knowledge_with_search(self, client, mock_httpx):
        """GET /mobile/knowledge should use search endpoint when search param provided."""
        mock_httpx.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "results": [
                    {"id": "1", "content": "Python programming", "topic": "tech", "importance": 0.9, "source": "manual"},
                ]
            },
        )

        response = client.get("/mobile/knowledge?search=Python")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert "Python" in data["items"][0]["content"]

    def test_create_knowledge_item(self, client, mock_httpx):
        """POST /mobile/knowledge should create a new item."""
        mock_httpx.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "id": "new_item",
                "content": "New knowledge",
                "topic": "tech",
                "importance": 0.8,
            },
        )

        response = client.post(
            "/mobile/knowledge",
            json={"content": "New knowledge", "topic": "tech"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "New knowledge"
        assert data["topic"] == "tech"

    def test_update_knowledge_item(self, client, mock_httpx):
        """PATCH /mobile/knowledge/{id} should update an item."""
        mock_httpx.patch.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "id": "item_1",
                "content": "Updated content",
                "topic": "science",
                "importance": 0.9,
            },
        )

        response = client.patch(
            "/mobile/knowledge/item_1",
            json={"content": "Updated content", "topic": "science", "importance": 0.9},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Updated content"
        assert data["topic"] == "science"

    def test_delete_knowledge_item(self, client, mock_httpx):
        """DELETE /mobile/knowledge/{id} should delete an item."""
        mock_httpx.delete.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "deleted"},
        )

        response = client.delete("/mobile/knowledge/item_1")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["id"] == "item_1"
