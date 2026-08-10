"""Tests for the VM API router (routers/vm.py).

Covers: list_builtins, vm_info.
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


class TestVMInfo:
    def test_info(self):
        client = TestClient(_app())
        resp = client.get("/vm/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["isa"] == "x86-32"
        assert "registers" in data
        assert "features" in data
