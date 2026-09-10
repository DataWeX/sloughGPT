"""
Profiles Router Tests
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from routers.profiles import ProfilesRouter


def _make_profile(pid="default", label="Default"):
    p = MagicMock()
    p.to_dict.return_value = {"id": pid, "label": label}
    return p


def _app(router: ProfilesRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(router.router)
    return app


class TestListProfiles:
    @patch("routers.profiles.list_profiles")
    def test_returns_list(self, mock_list):
        mock_list.return_value = [_make_profile("p1", "Fast"), _make_profile("p2", "Balanced")]
        client = TestClient(_app(ProfilesRouter()))
        resp = client.get("/profiles")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert len(body["data"]) == 2

    @patch("routers.profiles.list_profiles")
    def test_empty_list(self, mock_list):
        mock_list.return_value = []
        client = TestClient(_app(ProfilesRouter()))
        resp = client.get("/profiles")
        assert resp.status_code == 200
        assert resp.json()["data"] == []


class TestGetProfile:
    @patch("routers.profiles.get_profile")
    def test_returns_profile(self, mock_get):
        mock_get.return_value = _make_profile("fast", "Fast")
        client = TestClient(_app(ProfilesRouter()))
        resp = client.get("/profiles/fast")
        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == "fast"

    @patch("routers.profiles.get_profile")
    def test_not_found(self, mock_get):
        mock_get.return_value = None
        client = TestClient(_app(ProfilesRouter()), raise_server_exceptions=False)
        resp = client.get("/profiles/nonexistent")
        assert resp.status_code == 404


class TestApplyProfile:
    @patch("routers.profiles.apply_profile")
    def test_applies(self, mock_apply):
        mock_apply.return_value = {"applied": True, "profile_id": "fast"}
        client = TestClient(_app(ProfilesRouter()))
        resp = client.post("/profiles/apply", json={"profile_id": "fast"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("routers.profiles.apply_profile")
    def test_invalid_profile(self, mock_apply):
        mock_apply.side_effect = ValueError("profile not found")
        client = TestClient(_app(ProfilesRouter()), raise_server_exceptions=False)
        resp = client.post("/profiles/apply", json={"profile_id": "bad"})
        assert resp.status_code == 400


class TestActiveProfile:
    @patch("routers.profiles.get_profile")
    @patch("routers.profiles.get_active_profile_id")
    def test_returns_active(self, mock_active, mock_get):
        mock_active.return_value = "default"
        mock_get.return_value = _make_profile("default", "Default")
        client = TestClient(_app(ProfilesRouter()))
        resp = client.get("/profiles/active")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["active_profile_id"] == "default"
        assert data["profile"]["id"] == "default"


class TestRecommendProfile:
    @patch("routers.profiles.detect_recommended_profile", return_value="balanced")
    def test_recommends(self, mock_detect):
        client = TestClient(_app(ProfilesRouter()))
        resp = client.get("/profiles/recommend")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["recommended_profile_id"] == "balanced"
        assert "detected_ram_gb" in data
        assert "has_gpu" in data
