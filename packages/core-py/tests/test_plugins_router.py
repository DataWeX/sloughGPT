"""Tests for plugins router."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add API server to path
api_server = Path(__file__).resolve().parent.parent.parent.parent / "apps" / "api" / "server"
if str(api_server) not in sys.path:
    sys.path.insert(0, str(api_server))

from routers.plugins import PluginsRouter


@pytest.fixture
def client():
    app = FastAPI()
    router = PluginsRouter()
    app.include_router(router.router)
    return TestClient(app)


class TestPluginsRouter:
    def test_list_plugins(self, client):
        response = client.get("/plugins")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "plugins" in data["data"]

    def test_enable_plugin(self, client):
        response = client.post("/plugins/test-plugin/enable")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_disable_plugin(self, client):
        response = client.post("/plugins/test-plugin/disable")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_reload_plugins(self, client):
        response = client.post("/plugins/reload")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
