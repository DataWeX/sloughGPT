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
