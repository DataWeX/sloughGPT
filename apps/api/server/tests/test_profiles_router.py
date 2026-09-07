"""Tests for profiles router."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers.profiles import ProfilesRouter


@pytest.fixture()
def client():
    app = FastAPI()
    router = ProfilesRouter()
    app.include_router(router.router)
    return TestClient(app, raise_server_exceptions=False)


class TestListProfiles:
    def test_returns_profiles(self, client):
        res = client.get("/profiles")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        data = body["data"]
        assert isinstance(data, list)
        assert len(data) >= 5

    def test_each_profile_has_id(self, client):
        res = client.get("/profiles")
        for p in res.json()["data"]:
            assert "id" in p
            assert "name" in p
            assert "tier" in p


class TestGetProfile:
    def test_returns_profile(self, client):
        res = client.get("/profiles/balanced")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert body["data"]["id"] == "balanced"

    def test_404_for_unknown(self, client):
        res = client.get("/profiles/nonexistent")
        assert res.status_code == 404


class TestApplyProfile:
    def test_apply_balanced(self, client):
        res = client.post("/profiles/apply", json={"profile_id": "balanced"})
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["active_profile_id"] == "balanced"
        assert "live_settings" in data
        assert "requires_restart" in data

    def test_apply_unknown_returns_400(self, client):
        res = client.post("/profiles/apply", json={"profile_id": "nonexistent"})
        assert res.status_code == 400

    def test_apply_empty_id_returns_422(self, client):
        res = client.post("/profiles/apply", json={"profile_id": ""})
        assert res.status_code == 422


class TestActiveProfile:
    def test_returns_active(self, client):
        res = client.get("/profiles/active")
        assert res.status_code == 200
        body = res.json()
        assert "active_profile_id" in body["data"]

    def test_apply_then_active_updates(self, client):
        client.post("/profiles/apply", json={"profile_id": "cpu_only"})
        res = client.get("/profiles/active")
        assert res.json()["data"]["active_profile_id"] == "cpu_only"


class TestRecommendProfile:
    def test_returns_recommendation(self, client):
        res = client.get("/profiles/recommend")
        assert res.status_code == 200
        body = res.json()
        data = body["data"]
        assert "recommended_profile_id" in data
        assert "detected_ram_gb" in data
        assert "has_gpu" in data
        assert data["recommended_profile_id"] in [
            "cpu_only", "cpu_optimized", "balanced", "gpu_performance"
        ]
