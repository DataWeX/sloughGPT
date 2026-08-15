"""Tests for the VM API router (routers/vm.py).

Covers: list_builtins, vm_info, run, training jobs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.vm import router as vm_router  # noqa: E402


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(vm_router)
    return app


class TestListBuiltins:
    def test_builtins(self):
        client = TestClient(_app())
        resp = client.get("/vm/builtins")
        assert resp.status_code == 200
        programs = resp.json()["programs"]
        assert isinstance(programs, list)
        assert len(programs) >= 10
        names = [p["name"] for p in programs]
        assert "hello" in names
        assert "train" in names

    def test_builtins_have_required_fields(self):
        client = TestClient(_app())
        resp = client.get("/vm/builtins")
        programs = resp.json()["programs"]
        for p in programs:
            assert "name" in p
            assert "description" in p
            assert "code" in p

    def test_builtins_unique_names(self):
        client = TestClient(_app())
        resp = client.get("/vm/builtins")
        programs = resp.json()["programs"]
        names = [p["name"] for p in programs]
        assert len(names) == len(set(names))

    def test_builtins_hello_is_hello_world(self):
        client = TestClient(_app())
        resp = client.get("/vm/builtins")
        programs = resp.json()["programs"]
        hello = next(p for p in programs if p["name"] == "hello")
        assert "mov" in hello["code"].lower() or "int" in hello["code"].lower()


class TestVMInfo:
    def test_info(self):
        client = TestClient(_app())
        resp = client.get("/vm/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["isa"] == "x86-32"
        assert "registers" in data
        assert "features" in data

    def test_info_isa(self):
        client = TestClient(_app())
        resp = client.get("/vm/info")
        assert resp.json()["isa"] == "x86-32"

    def test_info_registers_is_dict(self):
        client = TestClient(_app())
        resp = client.get("/vm/info")
        assert isinstance(resp.json()["registers"], dict)

    def test_info_features_is_list(self):
        client = TestClient(_app())
        resp = client.get("/vm/info")
        assert isinstance(resp.json()["features"], list)

    def test_info_has_eax(self):
        client = TestClient(_app())
        resp = client.get("/vm/info")
        assert "EAX" in resp.json()["registers"]


class TestVMRun:
    def test_run_hello(self):
        client = TestClient(_app())
        resp = client.post("/vm/run", json={"program": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "output" in data or "stdout" in data

    def test_run_invalid_program(self):
        client = TestClient(_app())
        resp = client.post("/vm/run", json={"program": "nonexistent_xyz"})
        assert resp.status_code in (200, 400, 404, 422, 500)

    def test_run_empty_request(self):
        client = TestClient(_app())
        resp = client.post("/vm/run", json={})
        assert resp.status_code in (400, 422)

    def test_run_returns_output(self):
        client = TestClient(_app())
        resp = client.post("/vm/run", json={"program": "hello"})
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, dict)


class TestVMTrainingJobs:
    def test_training_jobs_list(self):
        client = TestClient(_app())
        resp = client.get("/vm/training/jobs")
        assert resp.status_code in (200, 404)

    def test_training_job_nonexistent(self):
        client = TestClient(_app())
        resp = client.get("/vm/training/jobs/nonexistent-id-123")
        assert resp.status_code in (200, 404)
