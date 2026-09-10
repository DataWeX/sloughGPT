"""Tests for the /world router — rendering, simulation, stats."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


def _mock_world():
    m = MagicMock()
    m.material = MagicMock()
    m.material.__gt__ = lambda self, x: MagicMock(sum=lambda: MagicMock(__int__=lambda s: 5))
    m.energy = MagicMock()
    m.idx = lambda x, y, z: x * 64 * 64 + y * 64 + z
    return m


def _mock_bridge():
    m = MagicMock()
    m.stats = {"render_time_ms": 1.2}
    m.render_state_tensors.return_value = {"depth": MagicMock(shape=(120, 160))}
    m.render.return_value = MagicMock()
    m.get_descriptor.return_value = "test descriptor"
    return m


def _mock_neural_bridge():
    m = MagicMock()
    m.stats = {"render_time_ms": 1.5}
    m.process_neural.return_value = {"embedding": MagicMock(shape=(1, 384))}
    m.get_descriptor.return_value = "neural descriptor"
    m._scene = None
    return m


def _mock_simulation():
    m = MagicMock()
    m.step.return_value = []
    return m


@pytest.fixture(autouse=True)
def _fresh_world():
    """Reset the world render router singleton state."""
    import routers.world_render as wr_mod
    for attr in dir(wr_mod):
        obj = getattr(wr_mod, attr, None)
        if hasattr(obj, '_world'):
            obj._world = None
        if hasattr(obj, '_scene'):
            obj._scene = None
        if hasattr(obj, '_last_render_bridge'):
            obj._last_render_bridge = None
    yield


class TestWorldStats:
    def setup_method(self):
        self.client = get_test_client()

    def test_stats_returns_success(self):
        resp = self.client.get("/world/stats")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_stats_has_components(self):
        resp = self.client.get("/world/stats")
        data = _d(resp)
        assert "components" in data
        assert isinstance(data["components"], list)
        assert "RenderBridge" in data["components"]

    def test_stats_has_world(self):
        resp = self.client.get("/world/stats")
        data = _d(resp)
        assert "world" in data
        assert "solid_blocks" in data["world"]
        assert "tick" in data["world"]

    def test_stats_has_materials(self):
        resp = self.client.get("/world/stats")
        data = _d(resp)
        assert "materials" in data
        assert data["materials"]["air"] == 0
        assert data["materials"]["ground"] == 1


class TestWorldTick:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.shell.simulation.Simulation")
    def test_tick_returns_success(self, MockSim):
        mock_sim = _mock_simulation()
        MockSim.return_value = mock_sim
        resp = self.client.post("/world/tick")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("domains.shell.simulation.Simulation")
    def test_tick_has_tick_count(self, MockSim):
        mock_sim = _mock_simulation()
        MockSim.return_value = mock_sim
        resp = self.client.post("/world/tick")
        data = _d(resp)
        assert "tick" in data
        assert "babies" in data

    @patch("domains.shell.simulation.Simulation")
    def test_tick_with_config(self, MockSim):
        mock_sim = _mock_simulation()
        MockSim.return_value = mock_sim
        resp = self.client.post("/world/tick", json={"max_ticks": 5, "render": False})
        assert resp.status_code == 200


class TestWorldRender:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.shell.world_render.RenderBridge")
    def test_render_returns_success(self, MockBridge):
        mock_bridge = _mock_bridge()
        MockBridge.return_value = mock_bridge
        resp = self.client.post("/world/render", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("domains.shell.world_render.RenderBridge")
    def test_render_has_shapes(self, MockBridge):
        mock_bridge = _mock_bridge()
        MockBridge.return_value = mock_bridge
        resp = self.client.post("/world/render", json={})
        data = _d(resp)
        assert "shapes" in data
        assert "tensor_keys" in data
        assert "stats" in data

    @patch("domains.shell.world_render.RenderBridge")
    def test_render_with_custom_config(self, MockBridge):
        mock_bridge = _mock_bridge()
        MockBridge.return_value = mock_bridge
        resp = self.client.post(
            "/world/render",
            json={"width": 320, "height": 240, "samples": 32},
        )
        assert resp.status_code == 200


class TestWorldRenderImage:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.shell.world_render.RenderBridge")
    def test_render_image_returns_png(self, MockBridge):
        mock_bridge = _mock_bridge()
        mock_bridge.render.return_value = np.random.rand(60, 80, 3)
        MockBridge.return_value = mock_bridge
        resp = self.client.post("/world/render/image", json={})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"


class TestWorldNeural:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.shell.world_render.NeuralRenderBridge")
    def test_neural_returns_success(self, MockNeural):
        mock_bridge = _mock_neural_bridge()
        MockNeural.return_value = mock_bridge
        resp = self.client.post("/world/neural", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("domains.shell.world_render.NeuralRenderBridge")
    def test_neural_has_embedding_shape(self, MockNeural):
        mock_bridge = _mock_neural_bridge()
        MockNeural.return_value = mock_bridge
        resp = self.client.post("/world/neural", json={})
        data = _d(resp)
        assert "embedding_shape" in data
        assert "descriptor" in data
        assert "stats" in data

    @patch("domains.shell.world_render.NeuralRenderBridge")
    def test_neural_with_config(self, MockNeural):
        mock_bridge = _mock_neural_bridge()
        MockNeural.return_value = mock_bridge
        resp = self.client.post(
            "/world/neural",
            json={"width": 80, "height": 60, "samples": 8},
        )
        assert resp.status_code == 200
