"""Tests for the world render API router (routers/world_render.py).

Covers: render_world, render_world_image, neural_process, run_tick, get_stats.
All domain imports are mocked to avoid heavy computation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.world_render import (
    WorldRenderRouter, RenderConfigRequest, SimTickRequest,
)


def _mock_bridge():
    bridge = MagicMock()
    bridge.stats = {"triangles": 100, "elapsed_ms": 1.5}
    bridge.render_state_tensors.return_value = {
        "color": np.zeros((4, 4, 3), dtype=np.float32),
        "depth": np.zeros((4, 4), dtype=np.float32),
    }
    bridge.render.return_value = np.zeros((4, 4, 3), dtype=np.float32)
    bridge.process_neural.return_value = {
        "embedding": np.zeros((1, 128), dtype=np.float32),
    }
    bridge.get_descriptor.return_value = {"objects": []}
    return bridge


def _mock_world():
    world = MagicMock()
    world.material = MagicMock()
    world.energy = MagicMock()
    world.idx.return_value = 0
    return world


def _mock_sim():
    sim = MagicMock()
    sim.step.return_value = []
    return sim


def _app():
    wrr = WorldRenderRouter()
    app = FastAPI()
    app.include_router(wrr.router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    return app


class TestRenderConfigRequest:
    def test_defaults(self):
        cfg = RenderConfigRequest()
        assert cfg.width == 160
        assert cfg.height == 120
        assert cfg.samples == 16
        assert cfg.camera_height == 40.0
        assert cfg.camera_distance == 30.0

    def test_custom(self):
        cfg = RenderConfigRequest(width=320, height=240, samples=32)
        assert cfg.width == 320
        assert cfg.height == 240
        assert cfg.samples == 32


class TestSimTickRequest:
    def test_defaults(self):
        req = SimTickRequest()
        assert req.max_ticks == 1
        assert req.render is True
        assert req.neural is False

    def test_custom(self):
        req = SimTickRequest(max_ticks=10, render=False, neural=True)
        assert req.max_ticks == 10
        assert req.render is False


class TestRenderWorld:
    @patch("domains.shell.simulation.WorldGrid")
    @patch("domains.shell.world_render.RenderBridge")
    def test_render_world_success(self, MockBridge, MockWorld):
        MockBridge.return_value = _mock_bridge()
        MockWorld.return_value = _mock_world()
        client = TestClient(_app())
        resp = client.post("/world/render", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "shapes" in body["data"]
        assert "tensor_keys" in body["data"]

    @patch("domains.shell.simulation.WorldGrid")
    @patch("domains.shell.world_render.RenderBridge")
    def test_render_world_custom_config(self, MockBridge, MockWorld):
        MockBridge.return_value = _mock_bridge()
        MockWorld.return_value = _mock_world()
        client = TestClient(_app())
        resp = client.post("/world/render", json={"width": 80, "height": 60})
        assert resp.status_code == 200


class TestRenderWorldImage:
    @patch("domains.shell.simulation.WorldGrid")
    @patch("domains.shell.world_render.RenderBridge")
    def test_image_returns_ppm(self, MockBridge, MockWorld):
        MockBridge.return_value = _mock_bridge()
        MockWorld.return_value = _mock_world()
        client = TestClient(_app())
        resp = client.post("/world/render/image", json={})
        assert resp.status_code == 200
        assert "portable-pixmap" in resp.headers["content-type"]

    @patch("domains.shell.simulation.WorldGrid")
    @patch("domains.shell.world_render.RenderBridge")
    def test_image_default_config(self, MockBridge, MockWorld):
        MockBridge.return_value = _mock_bridge()
        MockWorld.return_value = _mock_world()
        client = TestClient(_app())
        resp = client.post("/world/render/image")
        assert resp.status_code == 200


class TestNeuralProcess:
    @patch("domains.shell.simulation.WorldGrid")
    @patch("domains.shell.world_render.NeuralRenderBridge")
    def test_neural_success(self, MockBridge, MockWorld):
        MockBridge.return_value = _mock_bridge()
        MockWorld.return_value = _mock_world()
        client = TestClient(_app())
        resp = client.post("/world/neural", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "embedding_shape" in body["data"]
        assert "descriptor" in body["data"]


class TestRunTick:
    @patch("domains.shell.simulation.Simulation")
    @patch("domains.shell.simulation.SimScene")
    @patch("domains.shell.simulation.WorldParams")
    def test_tick_success(self, MockParams, MockScene, MockSim):
        MockScene.return_value = MagicMock(tick=0)
        MockSim.return_value = _mock_sim()
        client = TestClient(_app())
        resp = client.post("/world/tick", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["tick"] == 0

    @patch("domains.shell.world_render.RenderBridge")
    @patch("domains.shell.simulation.Simulation")
    @patch("domains.shell.simulation.SimScene")
    @patch("domains.shell.simulation.WorldParams")
    def test_tick_with_render(self, MockParams, MockScene, MockSim, MockRB):
        scene = MagicMock(tick=5)
        MockScene.return_value = scene
        MockSim.return_value = _mock_sim()
        MockRB.return_value = _mock_bridge()
        client = TestClient(_app())
        resp = client.post("/world/tick", json={"render": True})
        assert resp.status_code == 200
        assert resp.json()["data"]["tick"] == 5


class TestGetStats:
    def test_stats(self):
        client = TestClient(_app())
        resp = client.get("/world/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"]["status"] == "available"
        assert "RenderBridge" in body["data"]["components"]
        assert "air" in body["data"]["materials"]
