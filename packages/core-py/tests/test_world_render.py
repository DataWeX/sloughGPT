import numpy as np
import pytest

from domains.shell.simulation import WorldGrid
from domains.shell.world_render import (
    WorldToSceneMapper, RenderBridge, NeuralRenderBridge,
    RenderConfig, MATERIAL_AIR, MATERIAL_GROUND, MATERIAL_FOOD,
    MATERIAL_TOXIC, MATERIAL_SIGNAL, MATERIAL_NEST,
)


class TestWorldToSceneMapper:
    def test_empty_grid(self):
        mapper = WorldToSceneMapper()
        world = WorldGrid(size=(8, 4, 8))
        scene = mapper.grid_to_scene(world)
        assert scene is not None
        assert scene.camera is not None
        assert scene.background is not None

    def test_ground_material(self):
        mapper = WorldToSceneMapper()
        world = WorldGrid(size=(8, 4, 8))
        world.material[world.idx(3, 0, 3)] = MATERIAL_GROUND
        scene = mapper.grid_to_scene(world)
        assert len(scene.meshes) > 0
        mesh = scene.meshes[0]
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_multiple_materials(self):
        mapper = WorldToSceneMapper()
        world = WorldGrid(size=(8, 4, 8))
        world.material[world.idx(1, 0, 1)] = MATERIAL_GROUND
        world.material[world.idx(2, 0, 2)] = MATERIAL_FOOD
        world.material[world.idx(3, 0, 3)] = MATERIAL_TOXIC
        world.material[world.idx(4, 0, 4)] = MATERIAL_SIGNAL
        scene = mapper.grid_to_scene(world)
        assert len(scene.materials) >= 4
        assert len(scene.meshes) > 0

    def test_energy_emission(self):
        mapper = WorldToSceneMapper()
        world = WorldGrid(size=(8, 4, 8))
        world.material[world.idx(3, 0, 3)] = MATERIAL_SIGNAL
        world.energy[world.idx(3, 0, 3)] = 10.0
        scene = mapper.grid_to_scene(world)
        signal_mat = scene.materials[1]
        assert signal_mat.emission_strength > 0

    def test_camera_position(self):
        mapper = WorldToSceneMapper()
        world = WorldGrid(size=(8, 4, 8))
        scene = mapper.grid_to_scene(world)
        assert scene.camera.origin[1] > 0

    def test_custom_config(self):
        config = RenderConfig(
            width=320, height=240, samples=32,
            camera_height=60.0, camera_distance=50.0,
        )
        mapper = WorldToSceneMapper(config)
        world = WorldGrid(size=(8, 4, 8))
        scene = mapper.grid_to_scene(world)
        assert scene.camera.origin[1] == 60.0

    def test_baby_to_light(self):
        mapper = WorldToSceneMapper()
        from dataclasses import dataclass

        @dataclass
        class FakeBaby:
            alive: bool = True
            energy: float = 50.0
            position: np.ndarray = None

            def __post_init__(self):
                if self.position is None:
                    self.position = np.array([32.0, 0.0, 32.0])

        baby = FakeBaby()
        light = mapper.baby_to_light(baby)
        assert light is not None
        assert light.strength > 0

    def test_dead_baby_no_light(self):
        mapper = WorldToSceneMapper()
        from dataclasses import dataclass

        @dataclass
        class FakeBaby:
            alive: bool = False
            energy: float = 0.0
            position: np.ndarray = None

            def __post_init__(self):
                if self.position is None:
                    self.position = np.array([32.0, 0.0, 32.0])

        light = mapper.baby_to_light(FakeBaby())
        assert light is None

    def test_bvh_built(self):
        mapper = WorldToSceneMapper()
        world = WorldGrid(size=(8, 4, 8))
        world.material[world.idx(3, 0, 3)] = MATERIAL_GROUND
        scene = mapper.grid_to_scene(world)
        scene.build_bvh()
        assert len(scene._bvh_list) > 0


class TestRenderBridge:
    def test_build_scene(self):
        bridge = RenderBridge()
        world = WorldGrid(size=(8, 4, 8))
        world.material[world.idx(3, 0, 3)] = MATERIAL_GROUND
        scene = bridge.build_scene(world)
        assert scene is not None
        assert bridge.scene is scene

    def test_render(self):
        config = RenderConfig(width=32, height=24, samples=1)
        bridge = RenderBridge(config)
        world = WorldGrid(size=(8, 4, 8))
        world.material[world.idx(3, 0, 3)] = MATERIAL_GROUND
        world.material[world.idx(4, 0, 4)] = MATERIAL_FOOD
        world.energy[world.idx(4, 0, 4)] = 5.0
        bridge.build_scene(world)
        image = bridge.render()
        assert image.shape == (24, 32, 3)
        assert image.dtype == np.float32
        assert bridge.stats["renders"] == 1

    def test_render_state_tensors(self):
        config = RenderConfig(width=32, height=24, samples=1)
        bridge = RenderBridge(config)
        world = WorldGrid(size=(8, 4, 8))
        world.material[world.idx(3, 0, 3)] = MATERIAL_GROUND
        bridge.build_scene(world)
        tensors = bridge.render_state_tensors()
        assert "image" in tensors
        assert "depth" in tensors
        assert "normal" in tensors

    def test_render_tick(self):
        config = RenderConfig(width=32, height=24, samples=1)
        bridge = RenderBridge(config)
        world = WorldGrid(size=(8, 4, 8))
        world.material[world.idx(3, 0, 3)] = MATERIAL_GROUND
        image = bridge.render_tick(world)
        assert image.shape == (24, 32, 3)

    def test_render_no_grid(self):
        bridge = RenderBridge()
        image = bridge.render()
        assert image.shape[0] > 0
        assert image.shape[1] > 0

    def test_last_image(self):
        config = RenderConfig(width=32, height=24, samples=1)
        bridge = RenderBridge(config)
        world = WorldGrid(size=(8, 4, 8))
        world.material[world.idx(3, 0, 3)] = MATERIAL_GROUND
        bridge.build_scene(world)
        bridge.render()
        assert bridge.last_image is not None


class TestSimulationWithRender:
    def test_simulation_with_render_bridge(self):
        from domains.shell.simulation import SimScene, Simulation, WorldParams

        config = RenderConfig(width=32, height=24, samples=1)
        bridge = RenderBridge(config)

        params = WorldParams(grid_size=(8, 4, 8))
        scene = SimScene(params)
        sim = Simulation(scene, max_ticks=1, render_bridge=bridge)
        sim.step()
        assert bridge.stats["renders"] == 1

    def test_simulation_without_render_bridge(self):
        from domains.shell.simulation import SimScene, Simulation, WorldParams

        params = WorldParams(grid_size=(8, 4, 8))
        scene = SimScene(params)
        sim = Simulation(scene, max_ticks=1)
        results = sim.step()
        assert len(results) >= 0


class TestRenderDiff:
    def test_identical_images(self):
        from domains.shell.world_render import RenderDiff

        a = np.ones((8, 8, 3), dtype=np.float32) * 0.5
        b = a.copy()
        diff = RenderDiff(a, b)
        assert diff.mse == 0.0
        assert diff.mae == 0.0
        assert diff.change_ratio == 0.0

    def test_different_images(self):
        from domains.shell.world_render import RenderDiff

        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        diff = RenderDiff(a, b)
        assert diff.mse > 0.0
        assert diff.change_ratio == 1.0
        assert diff.max_diff == 1.0

    def test_summary(self):
        from domains.shell.world_render import RenderDiff

        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        diff = RenderDiff(a, b)
        s = diff.summary()
        assert "mse" in s
        assert "mae" in s
        assert "change_ratio" in s
        assert s["total_pixels"] == 64

    def test_heatmap(self):
        from domains.shell.world_render import RenderDiff

        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        diff = RenderDiff(a, b)
        hm = diff.heatmap()
        assert hm.shape == (8, 8)
        assert hm.max() <= 1.0
        assert hm.min() >= 0.0

    def test_diff_image(self):
        from domains.shell.world_render import RenderDiff

        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        diff = RenderDiff(a, b)
        di = diff.diff_image()
        assert di.shape == (8, 8, 3)
        assert di.max() <= 1.0

    def test_different_shapes(self):
        from domains.shell.world_render import RenderDiff

        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.zeros((16, 16, 3), dtype=np.float32)
        diff = RenderDiff(a, b)
        assert diff.mse == 0.0


class TestRenderHistory:
    def test_add_and_get(self):
        from domains.shell.world_render import RenderHistory

        history = RenderHistory(max_entries=10)
        img = np.ones((8, 8, 3), dtype=np.float32) * 0.5
        idx = history.add(img, tick=0)
        assert idx == 0
        assert len(history) == 1
        assert history.get(0) is not None
        assert history.get(1) is None

    def test_max_entries(self):
        from domains.shell.world_render import RenderHistory

        history = RenderHistory(max_entries=5)
        for i in range(10):
            img = np.ones((8, 8, 3), dtype=np.float32) * (i / 10)
            history.add(img, tick=i)
        assert len(history) == 5
        assert history[0]["tick"] == 5

    def test_diff(self):
        from domains.shell.world_render import RenderHistory

        history = RenderHistory()
        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        history.add(a, tick=0)
        history.add(b, tick=1)
        diff = history.diff(0, 1)
        assert diff is not None
        assert diff.mse > 0.0

    def test_diff_latest(self):
        from domains.shell.world_render import RenderHistory

        history = RenderHistory()
        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        history.add(a, tick=0)
        history.add(b, tick=1)
        diff = history.diff_latest()
        assert diff is not None
        assert diff.mse > 0.0

    def test_diff_latest_insufficient(self):
        from domains.shell.world_render import RenderHistory

        history = RenderHistory()
        a = np.zeros((8, 8, 3), dtype=np.float32)
        history.add(a, tick=0)
        diff = history.diff_latest()
        assert diff is None

    def test_timeline(self):
        from domains.shell.world_render import RenderHistory

        history = RenderHistory()
        for i in range(5):
            img = np.ones((8, 8, 3), dtype=np.float32) * i
            history.add(img, tick=i)
        tl = history.timeline()
        assert len(tl) == 5
        assert tl[0]["tick"] == 0
        assert tl[4]["tick"] == 4

    def test_recent(self):
        from domains.shell.world_render import RenderHistory

        history = RenderHistory()
        for i in range(10):
            img = np.ones((8, 8, 3), dtype=np.float32) * i
            history.add(img, tick=i)
        recent = history.recent(3)
        assert len(recent) == 3
        assert recent[0]["index"] == 7

    def test_clear(self):
        from domains.shell.world_render import RenderHistory

        history = RenderHistory()
        for i in range(5):
            img = np.ones((8, 8, 3), dtype=np.float32) * i
            history.add(img, tick=i)
        assert len(history) == 5
        history.clear()
        assert len(history) == 0

    def test_getitem(self):
        from domains.shell.world_render import RenderHistory

        history = RenderHistory()
        img = np.ones((8, 8, 3), dtype=np.float32) * 0.5
        history.add(img, tick=0)
        entry = history[0]
        assert entry is not None
        assert entry["tick"] == 0
        assert history[-1] is None


class TestRenderAnalyzer:
    def test_analyze_series(self):
        from domains.shell.world_render import RenderAnalyzer

        analyzer = RenderAnalyzer()
        for i in range(5):
            img = np.ones((8, 8, 3), dtype=np.float32) * (i / 5)
            analyzer.history.add(img, tick=i)
        result = analyzer.analyze_series()
        assert result["count"] == 5
        assert "mean_range" in result
        assert "mean_trend" in result

    def test_analyze_empty(self):
        from domains.shell.world_render import RenderAnalyzer

        analyzer = RenderAnalyzer()
        result = analyzer.analyze_series()
        assert result["count"] == 0

    def test_detect_significant_changes(self):
        from domains.shell.world_render import RenderAnalyzer

        analyzer = RenderAnalyzer()
        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        analyzer.history.add(a, tick=0)
        analyzer.history.add(b, tick=1)
        changes = analyzer.detect_significant_changes(threshold=0.1)
        assert len(changes) == 1
        assert changes[0]["from"] == 0
        assert changes[0]["to"] == 1

    def test_compare_ticks(self):
        from domains.shell.world_render import RenderAnalyzer

        analyzer = RenderAnalyzer()
        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        analyzer.history.add(a, tick=0)
        analyzer.history.add(b, tick=1)
        diff = analyzer.compare_ticks(0, 1)
        assert diff is not None
        assert diff.mse > 0.0

    def test_compare_ticks_missing(self):
        from domains.shell.world_render import RenderAnalyzer

        analyzer = RenderAnalyzer()
        diff = analyzer.compare_ticks(0, 1)
        assert diff is None

    def test_summary(self):
        from domains.shell.world_render import RenderAnalyzer

        analyzer = RenderAnalyzer()
        for i in range(3):
            img = np.ones((8, 8, 3), dtype=np.float32) * (i / 3)
            analyzer.history.add(img, tick=i)
        s = analyzer.summary()
        assert "count" in s
        assert "significant_changes" in s

    def test_render_diff_summary(self):
        from domains.shell.world_render import RenderAnalyzer

        analyzer = RenderAnalyzer()
        a = np.zeros((8, 8, 3), dtype=np.float32)
        b = np.ones((8, 8, 3), dtype=np.float32)
        analyzer.history.add(a, tick=0)
        analyzer.history.add(b, tick=1)
        text = analyzer.render_diff_summary(0, 1)
        assert "MSE" in text
        assert "MAE" in text

    def test_render_diff_summary_out_of_range(self):
        from domains.shell.world_render import RenderAnalyzer

        analyzer = RenderAnalyzer()
        text = analyzer.render_diff_summary(0, 1)
        assert "Cannot compare" in text
