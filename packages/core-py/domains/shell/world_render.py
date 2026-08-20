from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from .cycles import Scene, Mesh, Material, Light, Camera, CyclesRenderer


MATERIAL_AIR = 0
MATERIAL_GROUND = 1
MATERIAL_FOOD = 2
MATERIAL_TOXIC = 3
MATERIAL_SIGNAL = 4
MATERIAL_NEST = 5
MATERIAL_WATER = 6


@dataclass
class RenderConfig:
    width: int = 160
    height: int = 120
    samples: int = 16
    voxel_size: float = 1.0
    camera_height: float = 40.0
    camera_distance: float = 30.0
    sun_position: tuple[float, float, float] = (32.0, 50.0, 32.0)
    sun_strength: float = 3.0
    material_color_map: dict[int, tuple[float, float, float]] = field(default_factory=lambda: {
        MATERIAL_AIR: (0.05, 0.05, 0.08),
        MATERIAL_GROUND: (0.3, 0.25, 0.2),
        MATERIAL_FOOD: (0.1, 0.7, 0.2),
        MATERIAL_TOXIC: (0.7, 0.1, 0.1),
        MATERIAL_SIGNAL: (0.2, 0.3, 0.9),
        MATERIAL_NEST: (0.6, 0.5, 0.3),
        MATERIAL_WATER: (0.1, 0.3, 0.7),
    })
    material_emission_map: dict[int, float] = field(default_factory=lambda: {
        MATERIAL_SIGNAL: 2.0,
        MATERIAL_FOOD: 0.3,
    })


class WorldToSceneMapper:
    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()

    def grid_to_scene(self, world_grid, camera_target: tuple[float, float, float] | None = None) -> Scene:
        scene = Scene()
        nx, ny, nz = world_grid.size
        vs = self.config.voxel_size

        solid_mask = world_grid.material != MATERIAL_AIR
        indices = np.argwhere(solid_mask)

        if len(indices) == 0:
            scene.background = np.array([0.02, 0.02, 0.04])
            scene.camera = Camera(
                origin=np.array([nx * vs / 2, self.config.camera_height, nz * vs / 2 + self.config.camera_distance]),
                look_at=np.array([nx * vs / 2, 0, nz * vs / 2]),
                fov=50.0,
            )
            return scene

        materials_used = {}
        material_indices_map = {}

        for flat_idx_arr in indices:
            fi = int(flat_idx_arr[0])
            mat_type = int(world_grid.material[fi])
            if mat_type not in materials_used:
                materials_used[mat_type] = mat_type
                material_indices_map[mat_type] = len(scene.materials)
                base_color = np.array(self.config.material_color_map.get(mat_type, (0.5, 0.5, 0.5)))
                emission = self.config.material_emission_map.get(mat_type, 0.0)
                energy_val = float(world_grid.energy[fi])
                emission_strength = emission * (1.0 + energy_val)

                scene.materials.append(Material(
                    name=f"mat_{mat_type}",
                    base_color=base_color,
                    emission_strength=emission_strength,
                    roughness=0.8,
                ))

        mesh = self._build_voxel_mesh(indices, world_grid, vs, material_indices_map)
        scene.add_mesh(mesh)

        cx, cy, cz = nx * vs / 2, 0, nz * vs / 2
        if camera_target:
            cx, cy, cz = camera_target

        scene.camera = Camera(
            origin=np.array([cx, self.config.camera_height, cz + self.config.camera_distance]),
            look_at=np.array([cx, cy, cz]),
            fov=50.0,
        )

        scene.lights.append(Light(
            position=np.array(list(self.config.sun_position)),
            color=np.ones(3),
            strength=self.config.sun_strength,
            size=10.0,
        ))

        scene.lights.append(Light(
            position=np.array([cx, self.config.camera_height * 0.5, cz]),
            color=np.array([0.4, 0.4, 0.6]),
            strength=1.0,
            size=5.0,
        ))

        scene.background = np.array([0.02, 0.02, 0.04])
        return scene

    def _build_voxel_mesh(self, indices: np.ndarray, world_grid, voxel_size: float,
                           material_indices_map: dict[int, int]) -> Mesh:
        vs = voxel_size
        half = vs * 0.5
        cube_verts = np.array([
            [-half, -half, -half], [half, -half, -half],
            [half, half, -half], [-half, half, -half],
            [-half, -half, half], [half, -half, half],
            [half, half, half], [-half, half, half],
        ], dtype=np.float32)
        cube_faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ], dtype=np.int32)

        max_voxels = len(indices)
        all_vertices = np.zeros((max_voxels * 8, 3), dtype=np.float32)
        all_faces = np.zeros((max_voxels * 12, 3), dtype=np.int32)
        all_mat_idx = np.zeros(max_voxels * 12, dtype=np.int32)

        vi = 0
        fi = 0
        for flat_idx_arr in indices:
            fi_flat = int(flat_idx_arr[0])
            x, y, z = world_grid.coords(fi_flat)
            mat_type = int(world_grid.material[fi_flat])
            mat_idx = material_indices_map.get(mat_type, 0)

            center = np.array([(x + 0.5) * vs, (y + 0.5) * vs, (z + 0.5) * vs], dtype=np.float32)
            verts = cube_verts + center
            offset = vi // 8 * 8
            faces = cube_faces + offset

            all_vertices[vi:vi + 8] = verts
            all_faces[fi:fi + 12] = faces
            all_mat_idx[fi:fi + 12] = mat_idx
            vi += 8
            fi += 12

        all_vertices = all_vertices[:vi]
        all_faces = all_faces[:fi]
        all_mat_idx = all_mat_idx[:fi]

        mesh = Mesh(vertices=all_vertices, faces=all_faces, material_idx=all_mat_idx)
        return mesh

    def baby_to_light(self, baby) -> Light | None:
        if not baby.alive:
            return None
        pos = baby.position.astype(np.float32)
        energy_norm = min(baby.energy / 100.0, 1.0)
        return Light(
            position=pos,
            color=np.array([1.0, 0.9, 0.6]),
            strength=0.5 + energy_norm * 2.0,
            size=0.3,
        )

    def nest_to_mesh(self, nest, vs: float = 1.0) -> Mesh | None:
        if not nest.alive:
            return None
        pos = nest.position.astype(np.float32) * vs
        radius = 0.5 + (nest.stored_energy / 50.0) * 0.5
        segments = 8
        theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        verts = [pos + np.array([0, 0, 0])]
        for t in theta:
            verts.append(pos + np.array([np.cos(t) * radius, 0, np.sin(t) * radius]))
        verts.append(pos + np.array([0, radius * 1.5, 0]))
        verts = np.array(verts, dtype=np.float32)
        faces = []
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([0, i + 1, next_i + 1])
            faces.append([i + 1, next_i + 1, len(verts) - 1])
        faces = np.array(faces, dtype=np.int32)
        return Mesh(vertices=verts, faces=faces)


class RenderBridge:
    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()
        self._mapper = WorldToSceneMapper(self.config)
        self._renderer: CyclesRenderer | None = None
        self._scene: Scene | None = None
        self._last_image: np.ndarray | None = None
        self._last_state_tensors: dict[str, np.ndarray] | None = None
        self.stats = {"renders": 0, "total_time_ms": 0.0}

    def build_scene(self, world_grid, babies: list | None = None,
                    nests: list | None = None,
                    camera_target: tuple[float, float, float] | None = None) -> Scene:
        self._scene = self._mapper.grid_to_scene(world_grid, camera_target)

        if babies:
            for baby in babies:
                light = self._mapper.baby_to_light(baby)
                if light:
                    self._scene.lights.append(light)

        if nests:
            for nest in nests:
                mesh = self._mapper.nest_to_mesh(nest)
                if mesh:
                    self._scene.add_mesh(mesh)

        self._scene.build_bvh()
        return self._scene

    def render(self) -> np.ndarray:
        if self._scene is None:
            return np.zeros((self.config.height, self.config.width, 3), dtype=np.float32)
        import time
        t0 = time.monotonic()
        self._renderer = CyclesRenderer(
            self._scene,
            width=self.config.width,
            height=self.config.height,
            samples=self.config.samples,
        )
        image = self._renderer.render()
        elapsed = (time.monotonic() - t0) * 1000
        self._last_image = image
        self.stats["renders"] += 1
        self.stats["total_time_ms"] += elapsed
        return image

    def render_state_tensors(self) -> dict[str, np.ndarray]:
        if self._scene is None:
            return {}
        if self._renderer is None:
            self.render()
        tensors = self._renderer.render_state_tensors()
        self._last_state_tensors = tensors
        return tensors

    def render_tick(self, world_grid, babies: list | None = None,
                    nests: list | None = None,
                    camera_target: tuple[float, float, float] | None = None) -> np.ndarray:
        self.build_scene(world_grid, babies, nests, camera_target)
        return self.render()

    @property
    def last_image(self) -> np.ndarray | None:
        return self._last_image

    @property
    def last_state_tensors(self) -> dict[str, np.ndarray] | None:
        return self._last_state_tensors

    @property
    def scene(self) -> Scene | None:
        return self._scene


class NeuralRenderBridge(RenderBridge):
    def __init__(self, config: RenderConfig | None = None,
                 embed_dim: int = 64, num_classes: int = 8):
        super().__init__(config)
        from .cycles_device import CyclesDevice
        from .render_neural import RenderNeuralDevice
        self._cycles_device = CyclesDevice(
            width=self.config.width,
            height=self.config.height,
            samples=self.config.samples,
        )
        self._neural = RenderNeuralDevice(
            cycles_device=self._cycles_device,
            embed_dim=embed_dim,
            num_classes=num_classes,
        )
        self._last_embedding: np.ndarray | None = None
        self._last_classification: dict | None = None
        self._last_descriptor: dict | None = None
        self.stats = {"renders": 0, "total_time_ms": 0.0, "neural_processes": 0}

    def render_tick(self, world_grid, babies: list | None = None,
                    nests: list | None = None,
                    camera_target: tuple[float, float, float] | None = None) -> np.ndarray:
        image = super().render_tick(world_grid, babies, nests, camera_target)
        if self._scene is not None:
            self._sync_scene_to_cycles_device()
            self._last_state_tensors = self._cycles_device.call("state_tensors")
        return image

    def _sync_scene_to_cycles_device(self) -> None:
        if self._scene is None:
            return
        cycles = self._cycles_device
        cycles.call("clear")
        for mesh in self._scene.meshes:
            for face_idx in range(len(mesh.faces)):
                pass
        for i, mat in enumerate(self._scene.materials):
            cycles.call("set_material", i,
                        mat.base_color[0], mat.base_color[1], mat.base_color[2],
                        mat.metallic, mat.roughness, mat.emission_strength,
                        mat.transmission, mat.ior)
        for light in self._scene.lights:
            cycles.call("add_light",
                        light.position[0], light.position[1], light.position[2],
                        light.color[0], light.color[1], light.color[2],
                        light.strength)
        cam = self._scene.camera
        cycles.call("set_camera",
                    cam.origin[0], cam.origin[1], cam.origin[2],
                    cam.look_at[0], cam.look_at[1], cam.look_at[2],
                    cam.fov)
        bg = self._scene.background
        cycles.call("set_background", bg[0], bg[1], bg[2])

    def process_neural(self) -> dict[str, Any]:
        if self._cycles_device is None:
            return {}
        result = self._neural.call("process")
        self._last_embedding = result.get("embedding")
        self._last_classification = result
        self.stats["neural_processes"] += 1
        return result

    def get_embedding(self) -> np.ndarray | None:
        if self._last_embedding is None:
            self.process_neural()
        return self._last_embedding

    def get_descriptor(self) -> dict:
        return self._neural.call("descriptor")

    @property
    def last_embedding(self) -> np.ndarray | None:
        return self._last_embedding

    @property
    def last_classification(self) -> dict | None:
        return self._last_classification
