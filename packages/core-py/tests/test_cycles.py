"""
Tests for the Cycles path tracer and CyclesDevice VM integration.

Run: PYTHONPATH=packages/core-py .venv/bin/python -m pytest tests/test_cycles.py -x -q
"""

import numpy as np
import pytest
from domains.shell.cycles import (
    CyclesRenderer, Scene, Camera, Material, Light, BVH,
    create_sphere, create_plane, create_cube,
)
from domains.shell.cycles_device import CyclesDevice
from domains.shell.render_neural import RenderNeuralDevice


class TestMeshPrimitives:
    def test_create_sphere_has_vertices(self):
        mesh = create_sphere(radius=0.5, center=np.array([0, 0, 0]))
        assert mesh.vertices.shape[1] == 3
        assert mesh.vertices.shape[0] > 0

    def test_create_sphere_has_faces(self):
        mesh = create_sphere(radius=1.0, segments=8)
        assert mesh.faces.shape[0] > 0
        assert mesh.faces.shape[1] == 3

    def test_create_sphere_normals_sum_to_one(self):
        mesh = create_sphere(radius=0.5)
        lens = np.linalg.norm(mesh.normals, axis=1)
        assert np.allclose(lens, 1.0, atol=1e-6)

    def test_create_sphere_with_offset(self):
        mesh = create_sphere(center=np.array([1, 2, 3]))
        center = mesh.vertices.mean(axis=0)
        assert np.allclose(center, [1, 2, 3], atol=0.1)

    def test_create_plane_has_vertices(self):
        mesh = create_plane(size=2.0, y=0.0)
        assert mesh.vertices.shape[0] > 0

    def test_create_plane_y_coordinate(self):
        mesh = create_plane(size=2.0, y=1.5)
        ys = mesh.vertices[:, 1]
        assert np.allclose(ys, 1.5)

    def test_create_cube_has_8_or_more_vertices(self):
        mesh = create_cube(size=1.0)
        assert mesh.vertices.shape[0] >= 8

    def test_create_cube_has_faces(self):
        mesh = create_cube(size=1.0)
        assert mesh.faces.shape[0] > 0

    def test_create_sphere_segments_parameter(self):
        m8 = create_sphere(segments=8)
        m16 = create_sphere(segments=16)
        assert m16.vertices.shape[0] > m8.vertices.shape[0]


class TestMaterial:
    def test_emission_zero_by_default(self):
        mat = Material()
        e = mat.emission()
        assert np.allclose(e, 0.0) if isinstance(e, np.ndarray) else e == 0.0

    def test_emission_with_strength(self):
        mat = Material(emission_strength=5.0)
        e = mat.emission()
        assert np.allclose(e, 5.0) if isinstance(e, np.ndarray) else e == 5.0

    def test_default_ior(self):
        mat = Material()
        assert mat.ior == pytest.approx(1.45)

    def test_base_color_stored(self):
        mat = Material(base_color=np.array([0.5, 0.0, 0.0]))
        assert mat.base_color[0] == pytest.approx(0.5)


class TestScene:
    def test_empty_scene(self):
        scene = Scene()
        assert len(scene.meshes) == 0
        assert len(scene.lights) == 0
        assert len(scene.materials) >= 1

    def test_add_mesh(self):
        scene = Scene()
        mesh = create_sphere()
        scene.add_mesh(mesh)
        assert len(scene.meshes) == 1

    def test_add_meshes(self):
        scene = Scene()
        scene.add_mesh(create_sphere())
        scene.add_mesh(create_cube())
        assert len(scene.meshes) == 2


class TestBVH:
    def test_bvh_construction(self):
        mesh = create_sphere(segments=6)
        bvh = BVH(mesh)
        assert len(bvh.nodes) > 0

    def test_bvh_hits_triangle(self):
        mesh = create_plane(size=10.0, y=0.0)
        bvh = BVH(mesh)
        origin = np.array([0.0, 1.0, 0.0])
        direction = np.array([0.0, -1.0, 0.0])
        t, fi = bvh.intersect(origin, direction)
        assert t < 1e30
        assert fi >= 0

    def test_bvh_misses(self):
        mesh = create_plane(size=2.0, y=0.0)
        bvh = BVH(mesh)
        origin = np.array([0.0, 1.0, 0.0])
        direction = np.array([0.0, 1.0, 0.0])
        t, fi = bvh.intersect(origin, direction)
        assert t >= 1e30


class TestCamera:
    def test_camera_rays(self):
        cam = Camera(fov=50)
        rays_o, rays_d = cam.generate_rays(4, 3)
        assert rays_o.shape == (3, 4, 3)
        assert rays_d.shape == (3, 4, 3)

    def test_camera_rays_normalized(self):
        cam = Camera()
        _, rays_d = cam.generate_rays(8, 8)
        lens = np.linalg.norm(rays_d, axis=-1)
        assert np.allclose(lens, 1.0, atol=1e-5)


class TestRendering:
    def _make_scene(self):
        scene = Scene(camera=Camera(origin=np.array([0, 2, 4]), look_at=np.array([0, 0, 0]), fov=50))
        scene.materials = [
            Material(base_color=np.array([0.3, 0.3, 0.3]), roughness=0.8),
            Material(base_color=np.array([0.8, 0.1, 0.1]), roughness=0.3),
        ]
        scene.meshes = [
            create_plane(size=6, y=-1, mat_idx=0),
            create_sphere(radius=0.6, center=np.array([0, 0, 0]), segments=8, mat_idx=1),
        ]
        scene.lights = [Light(position=np.array([2, 3, 2]), color=np.array([1, 0.95, 0.9]), strength=6)]
        return scene

    def test_render_returns_image(self):
        scene = self._make_scene()
        renderer = CyclesRenderer(scene, width=16, height=12, samples=1)
        scene.build_bvh()
        img = renderer.render()
        assert img.shape == (12, 16, 3)

    def test_render_nonzero_pixels(self):
        scene = self._make_scene()
        renderer = CyclesRenderer(scene, width=32, height=24, samples=2)
        scene.build_bvh()
        img = renderer.render()
        assert (img.sum(axis=-1) > 0.01).sum() > 0

    def test_render_pixel_range(self):
        scene = self._make_scene()
        renderer = CyclesRenderer(scene, width=16, height=12, samples=1)
        scene.build_bvh()
        img = renderer.render()
        assert img.min() >= 0.0
        assert img.max() <= 2.0

    def test_render_state_tensors(self):
        scene = self._make_scene()
        renderer = CyclesRenderer(scene, width=16, height=12, samples=1)
        scene.build_bvh()
        tensors = renderer.render_state_tensors()
        assert "image" in tensors
        assert "normal" in tensors
        assert "depth" in tensors
        assert "albedo" in tensors
        assert "emission" in tensors
        assert "mask" in tensors

    def test_state_tensor_shapes(self):
        scene = self._make_scene()
        renderer = CyclesRenderer(scene, width=16, height=12, samples=1)
        scene.build_bvh()
        tensors = renderer.render_state_tensors()
        assert tensors["image"].shape == (12, 16, 3)
        assert tensors["depth"].shape == (12, 16)
        assert tensors["normal"].shape == (12, 16, 3)
        assert tensors["mask"].shape == (12, 16)

    def test_render_empty_scene(self):
        scene = Scene(camera=Camera())
        renderer = CyclesRenderer(scene, width=8, height=8, samples=1)
        img = renderer.render()
        assert img.shape == (8, 8, 3)

    def test_render_all_black_background(self):
        scene = Scene(camera=Camera(), background=np.array([0, 0, 0]))
        renderer = CyclesRenderer(scene, width=8, height=8, samples=1)
        img = renderer.render()
        assert img.max() < 0.01


class TestCyclesDevice:
    def test_device_info(self):
        dev = CyclesDevice(width=32, height=24, samples=2)
        info = dev.call("info")
        assert info["type"] == "cycles"
        assert info["resolution"] == [32, 24]
        assert info["samples"] == 2

    def test_device_add_sphere(self):
        dev = CyclesDevice()
        idx = dev.call("add_sphere", 0.5, 0.0, 0.0, 0.0, 0, 8)
        assert idx[0] == 0

    def test_device_add_light(self):
        dev = CyclesDevice()
        idx = dev.call("add_light", 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 5.0)
        assert idx[0] == 0

    def test_device_set_material(self):
        dev = CyclesDevice()
        dev.call("set_material", 0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.45)
        assert len(dev._scene.materials) == 1

    def test_device_render(self):
        dev = CyclesDevice(width=16, height=12, samples=1)
        dev.call("add_sphere", 0.5, 0.0, -0.5, 0.0, 0, 6)
        dev.call("add_light", 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 5.0)
        img = dev.call("render")
        assert img.shape == (12, 16, 3)

    def test_device_state_tensors(self):
        dev = CyclesDevice(width=16, height=12, samples=1)
        dev.call("add_sphere", 0.5)
        tensors = dev.call("state_tensors")
        assert "image" in tensors
        assert "normal" in tensors
        assert tensors["image"].shape == (12, 16, 3)

    def test_device_clear(self):
        dev = CyclesDevice()
        dev.call("add_sphere", 0.5)
        dev.call("clear")
        info = dev.call("info")
        assert info["meshes"] == 0

    def test_device_set_resolution(self):
        dev = CyclesDevice()
        dev.call("set_resolution", 64, 48)
        assert dev._width == 64
        assert dev._height == 48

    def test_device_set_samples(self):
        dev = CyclesDevice()
        dev.call("set_samples", 32)
        assert dev._samples == 32

    def test_device_unknown_op_raises(self):
        dev = CyclesDevice()
        with pytest.raises(Exception):
            dev.call("nonexistent_op")

    def test_device_multiple_meshes(self):
        dev = CyclesDevice(width=16, height=12, samples=1)
        dev.call("set_material", 0, 0.3, 0.3, 0.3, 0, 0.8)
        dev.call("set_material", 1, 0.8, 0.1, 0.1, 0.1, 0.3)
        dev.call("add_plane", 4.0, -1.0, 0)
        dev.call("add_sphere", 0.3, 0.0, 0.0, 0.0, 1, 6)
        dev.call("add_cube", 0.2, 1.0, 0.0, 0.0, 1)
        dev.call("add_light", 0.0, 3.0, 0.0, 1.0, 1.0, 1.0, 6.0)
        img = dev.call("render")
        assert img.shape == (12, 16, 3)
        info = dev.call("info")
        assert info["meshes"] == 3

    def test_device_add_cube(self):
        dev = CyclesDevice()
        idx = dev.call("add_cube", 0.5, 0.0, 0.0, 0.0, 0)
        assert idx[0] == 0

    def test_device_add_plane(self):
        dev = CyclesDevice()
        idx = dev.call("add_plane", 2.0, -1.0, 0)
        assert idx[0] == 0

    def test_device_set_camera(self):
        dev = CyclesDevice()
        dev.call("set_camera", 0, 1, 4, 0, 0, 0, 60)
        assert dev._scene.camera.fov == 60

    def test_device_set_background(self):
        dev = CyclesDevice()
        dev.call("set_background", 0.5, 0.5, 0.5)
        assert np.allclose(dev._scene.background, [0.5, 0.5, 0.5])


def _make_cycles(width=16, height=12, samples=1):
    dev = CyclesDevice(width=width, height=height, samples=samples)
    dev.call("set_material", 0, 0.3, 0.3, 0.3, 0, 0.8)
    dev.call("set_material", 1, 0.8, 0.1, 0.1, 0.1, 0.3)
    dev.call("add_plane", 6.0, -1.0, 0)
    dev.call("add_sphere", 0.6, 0.0, 0.0, 0.0, 1, 6)
    dev.call("add_light", 2.0, 3.0, 2.0, 1.0, 1.0, 1.0, 6.0)
    return dev


class TestRenderNeuralDevice:
    def test_info(self):
        dev = RenderNeuralDevice()
        info = dev.call("info")
        assert info["type"] == "render_neural"
        assert info["embed_dim"] == 64
        assert info["num_classes"] == 8
        assert info["input_channels"] == 6

    def test_process_returns_keys(self):
        cycles = _make_cycles()
        neural = RenderNeuralDevice(cycles_device=cycles)
        out = neural.call("process")
        assert "embedding" in out
        assert "logits" in out
        assert "probabilities" in out
        assert "features" in out

    def test_embedding_shape_and_norm(self):
        cycles = _make_cycles()
        neural = RenderNeuralDevice(cycles_device=cycles)
        out = neural.call("process")
        assert out["embedding"].shape == (64,)
        norm = np.linalg.norm(out["embedding"])
        assert abs(norm - 1.0) < 1e-4

    def test_probabilities_sum_to_one(self):
        cycles = _make_cycles()
        neural = RenderNeuralDevice(cycles_device=cycles)
        out = neural.call("process")
        assert abs(out["probabilities"].sum() - 1.0) < 1e-5

    def test_embed_method(self):
        cycles = _make_cycles()
        neural = RenderNeuralDevice(cycles_device=cycles)
        emb = neural.call("embed")
        assert emb.shape == (64,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_classify_returns_labels(self):
        cycles = _make_cycles()
        neural = RenderNeuralDevice(cycles_device=cycles)
        cls = neural.call("classify")
        assert "labels" in cls
        assert "probabilities" in cls
        assert cls["labels"].ndim == 2

    def test_descriptor_returns_stats(self):
        cycles = _make_cycles()
        neural = RenderNeuralDevice(cycles_device=cycles)
        desc = neural.call("descriptor")
        assert "image" in desc
        assert "depth" in desc
        assert "neural_embedding_norm" in desc
        assert "neural_entropy" in desc
        assert "dominant_class" in desc
        assert 0 <= desc["dominant_class"] < 8

    def test_descriptor_image_stats(self):
        cycles = _make_cycles()
        neural = RenderNeuralDevice(cycles_device=cycles)
        desc = neural.call("descriptor")
        assert desc["image"]["mean"] >= 0
        assert desc["image"]["max"] >= desc["image"]["min"]

    def test_no_source_raises(self):
        neural = RenderNeuralDevice()
        with pytest.raises(Exception):
            neural.call("process")

    def test_set_source(self):
        neural = RenderNeuralDevice()
        cycles = _make_cycles()
        neural.call("set_source", cycles)
        out = neural.call("process")
        assert "embedding" in out

    def test_unknown_op_raises(self):
        neural = RenderNeuralDevice()
        with pytest.raises(Exception):
            neural.call("nonexistent_op")

    def test_forward_raw(self):
        cycles = _make_cycles()
        neural = RenderNeuralDevice(cycles_device=cycles)
        tensors = cycles.call("state_tensors")
        out = neural.call("forward", {"state_tensors": tensors})
        assert out["embedding"].shape == (64,)

    def test_custom_embed_dim(self):
        cycles = _make_cycles()
        neural = RenderNeuralDevice(cycles_device=cycles, embed_dim=32)
        out = neural.call("process")
        assert out["embedding"].shape == (32,)

    def test_custom_num_classes(self):
        cycles = _make_cycles()
        neural = RenderNeuralDevice(cycles_device=cycles, num_classes=4)
        out = neural.call("process")
        assert out["probabilities"].shape == (4,)
