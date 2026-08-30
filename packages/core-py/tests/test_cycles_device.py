"""Tests for domains.shell.cycles_device — CyclesDevice VM device wrapper."""

import numpy as np
import pytest
from domains.shell.cycles_device import CyclesDevice
from domains.shell.vm import DeviceFault


# ── Basic Info ─────────────────────────────────────────────────────────


class TestCyclesDeviceInfo:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_info(self):
        info = self.dev.info()
        assert info["type"] == "cycles"
        assert info["resolution"] == [32, 24]
        assert info["samples"] == 4
        assert info["meshes"] == 0
        assert info["lights"] == 0

    def test_info_materials_initially_one(self):
        """Scene starts with 1 default material."""
        info = self.dev.info()
        assert info["materials"] == 1

    def test_info_last_render_shape_initially_none(self):
        info = self.dev.info()
        assert info["last_render_shape"] is None

    def test_info_ops_list(self):
        info = self.dev.info()
        expected_ops = [
            "render", "state_tensors", "add_sphere", "add_cube",
            "add_plane", "add_light", "set_camera", "set_material",
            "set_background", "info", "clear", "set_samples",
            "set_resolution",
        ]
        assert set(info["ops"]) == set(expected_ops)

    def test_custom_dimensions(self):
        dev = CyclesDevice(width=640, height=480, samples=32)
        info = dev.info()
        assert info["resolution"] == [640, 480]
        assert info["samples"] == 32

    def test_info_returns_dict(self):
        info = self.dev.info()
        assert isinstance(info, dict)

    def test_default_dimensions(self):
        dev = CyclesDevice()
        info = dev.info()
        assert info["resolution"] == [160, 120]
        assert info["samples"] == 16

    def test_meshes_count_increases(self):
        self.dev._add_sphere()
        self.dev._add_sphere(cx=1)
        assert self.dev.info()["meshes"] == 2

    def test_lights_count_increases(self):
        self.dev._add_light()
        self.dev._add_light(x=1)
        assert self.dev.info()["lights"] == 2

    def test_materials_count_increases(self):
        self.dev.call("set_material", 0)
        self.dev.call("set_material", 1)
        assert self.dev.info()["materials"] == 2


# ── Unknown Op ─────────────────────────────────────────────────────────


class TestUnknownOp:
    def test_call_unknown_op(self):
        dev = CyclesDevice()
        with pytest.raises(DeviceFault, match="unknown op"):
            dev.call("nonexistent")

    def test_call_empty_string(self):
        dev = CyclesDevice()
        with pytest.raises(DeviceFault, match="unknown op"):
            dev.call("")

    def test_call_none(self):
        dev = CyclesDevice()
        with pytest.raises(DeviceFault, match="unknown op"):
            dev.call(None)

    def test_call_unknown_with_args(self):
        dev = CyclesDevice()
        with pytest.raises(DeviceFault):
            dev.call("bogus", 1, 2, 3)


# ── Add Sphere ─────────────────────────────────────────────────────────


class TestAddSphere:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_add_sphere(self):
        result = self.dev.call("add_sphere", 0.5, 0.0, 0.0, 0.0, 0, 16)
        assert result[0] == 0
        assert self.dev.info()["meshes"] == 1

    def test_add_sphere_defaults(self):
        result = self.dev._add_sphere()
        assert result[0] == 0

    def test_add_sphere_returns_mesh_index(self):
        self.dev._add_sphere(cx=0)
        result = self.dev._add_sphere(cx=2)
        assert result[0] == 1

    def test_add_sphere_stores_in_scene(self):
        self.dev._add_sphere(radius=1.0)
        assert len(self.dev._scene.meshes) == 1

    def test_add_sphere_via_call(self):
        self.dev.call("add_sphere")
        assert self.dev.info()["meshes"] == 1

    def test_add_sphere_numpy_result(self):
        result = self.dev._add_sphere()
        assert isinstance(result, np.ndarray)

    def test_add_sphere_custom_radius(self):
        self.dev._add_sphere(radius=2.0)
        assert self.dev.info()["meshes"] == 1

    def test_add_sphere_custom_center(self):
        self.dev._add_sphere(cx=1.0, cy=2.0, cz=3.0)
        assert self.dev.info()["meshes"] == 1

    def test_add_sphere_custom_segments(self):
        self.dev._add_sphere(segments=32)
        assert self.dev.info()["meshes"] == 1

    def test_add_sphere_mat_idx(self):
        self.dev._add_sphere(mat_idx=2)
        assert self.dev.info()["meshes"] == 1

    def test_add_many_spheres(self):
        for i in range(10):
            self.dev._add_sphere(cx=float(i))
        assert self.dev.info()["meshes"] == 10


# ── Add Cube ───────────────────────────────────────────────────────────


class TestAddCube:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_add_cube(self):
        result = self.dev.call("add_cube", 2.0)
        assert result[0] == 0
        assert self.dev.info()["meshes"] == 1

    def test_add_cube_defaults(self):
        result = self.dev._add_cube()
        assert result[0] == 0

    def test_add_cube_returns_mesh_index(self):
        self.dev._add_cube()
        result = self.dev._add_cube(size=2.0)
        assert result[0] == 1

    def test_add_cube_via_call_with_params(self):
        self.dev.call("add_cube", 1.5, 1.0, 2.0, 3.0, 0)
        assert self.dev.info()["meshes"] == 1

    def test_add_cube_numpy_result(self):
        result = self.dev._add_cube()
        assert isinstance(result, np.ndarray)

    def test_add_cube_custom_center(self):
        self.dev._add_cube(cx=1.0, cy=2.0, cz=3.0)
        assert self.dev.info()["meshes"] == 1

    def test_add_cube_mat_idx(self):
        self.dev._add_cube(mat_idx=3)
        assert self.dev.info()["meshes"] == 1


# ── Add Plane ──────────────────────────────────────────────────────────


class TestAddPlane:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_add_plane(self):
        result = self.dev.call("add_plane")
        assert result[0] == 0

    def test_add_plane_defaults(self):
        result = self.dev._add_plane()
        assert result[0] == 0

    def test_add_plane_custom_params(self):
        result = self.dev.call("add_plane", 5.0, -2.0, 1)
        assert result[0] == 0

    def test_add_plane_adds_to_scene(self):
        self.dev._add_plane()
        assert len(self.dev._scene.meshes) == 1

    def test_add_plane_numpy_result(self):
        result = self.dev._add_plane()
        assert isinstance(result, np.ndarray)

    def test_add_multiple_planes(self):
        self.dev._add_plane()
        self.dev._add_plane(y=0.0)
        assert self.dev.info()["meshes"] == 2


# ── Add Light ──────────────────────────────────────────────────────────


class TestAddLight:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_add_light(self):
        result = self.dev.call("add_light", 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0)
        assert result[0] == 0
        assert self.dev.info()["lights"] == 1

    def test_add_light_defaults(self):
        result = self.dev._add_light()
        assert result[0] == 0

    def test_add_light_returns_index(self):
        self.dev._add_light()
        result = self.dev._add_light(x=1.0)
        assert result[0] == 1

    def test_add_light_via_call(self):
        self.dev.call("add_light")
        assert self.dev.info()["lights"] == 1

    def test_add_multiple_lights(self):
        self.dev._add_light(r=1.0, g=0.0, b=0.0)
        self.dev._add_light(r=0.0, g=1.0, b=0.0)
        assert self.dev.info()["lights"] == 2

    def test_add_light_numpy_result(self):
        result = self.dev._add_light()
        assert isinstance(result, np.ndarray)

    def test_add_light_custom_color(self):
        self.dev._add_light(r=0.5, g=0.3, b=0.8, strength=2.0)
        assert self.dev.info()["lights"] == 1


# ── Set Camera ─────────────────────────────────────────────────────────


class TestSetCamera:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_set_camera(self):
        self.dev.call("set_camera", 0.0, 1.5, 4.0, 0.0, 0.0, 0.0, 50.0)
        assert self.dev._scene.camera is not None

    def test_set_camera_defaults(self):
        self.dev._set_camera()
        assert self.dev._scene.camera is not None

    def test_set_camera_invalidates_renderer(self):
        self.dev._add_sphere()
        self.dev.call("render")
        assert self.dev._renderer is not None
        self.dev.call("set_camera")
        assert self.dev._renderer is None

    def test_set_camera_custom_fov(self):
        self.dev.call("set_camera", 0.0, 1.5, 4.0, 0.0, 0.0, 0.0, 90.0)
        cam = self.dev._scene.camera
        assert cam.fov == 90.0

    def test_set_camera_origin(self):
        self.dev._set_camera(ox=5.0, oy=3.0, oz=1.0)
        cam = self.dev._scene.camera
        assert np.allclose(cam.origin, [5.0, 3.0, 1.0])

    def test_set_camera_look_at(self):
        self.dev._set_camera(lx=1.0, ly=2.0, lz=3.0)
        cam = self.dev._scene.camera
        assert np.allclose(cam.look_at, [1.0, 2.0, 3.0])


# ── Set Material ───────────────────────────────────────────────────────


class TestSetMaterial:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_set_material(self):
        self.dev.call("add_sphere")
        self.dev.call("set_material", 0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.45)
        assert len(self.dev._scene.materials) >= 1

    def test_set_material_creates_if_needed(self):
        self.dev.call("set_material", 5)
        assert len(self.dev._scene.materials) >= 6

    def test_set_material_updates_properties(self):
        self.dev.call("add_sphere")
        self.dev.call("set_material", 0, 1.0, 0.0, 0.0, 0.8, 0.3, 0.5, 0.1, 2.0)
        mat = self.dev._scene.materials[0]
        assert mat.metallic == 0.8
        assert mat.roughness == 0.3
        assert mat.emission_strength == 0.5
        assert mat.transmission == 0.1
        assert mat.ior == 2.0

    def test_set_material_invalidates_renderer(self):
        self.dev._add_sphere()
        self.dev.call("render")
        assert self.dev._renderer is not None
        self.dev.call("set_material", 0)
        assert self.dev._renderer is None

    def test_set_material_base_color(self):
        self.dev.call("add_sphere")
        self.dev.call("set_material", 0, 0.5, 0.3, 0.7)
        mat = self.dev._scene.materials[0]
        assert np.allclose(mat.base_color, [0.5, 0.3, 0.7])

    def test_set_material_defaults(self):
        self.dev.call("set_material", 0)
        mat = self.dev._scene.materials[0]
        assert mat.metallic == 0.0
        assert mat.roughness == 0.5

    def test_set_material_replaces_existing(self):
        self.dev.call("set_material", 0, 1.0, 0.0, 0.0)
        self.dev.call("set_material", 0, 0.0, 1.0, 0.0)
        mat = self.dev._scene.materials[0]
        assert np.allclose(mat.base_color, [0.0, 1.0, 0.0])

    def test_set_material_high_idx(self):
        self.dev.call("set_material", 100)
        assert len(self.dev._scene.materials) > 100


# ── Set Background ─────────────────────────────────────────────────────


class TestSetBackground:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_set_background(self):
        self.dev.call("set_background", 0.1, 0.2, 0.3)
        assert np.allclose(self.dev._scene.background, [0.1, 0.2, 0.3])

    def test_set_background_defaults(self):
        self.dev._set_background()
        assert np.allclose(self.dev._scene.background, [0.02, 0.02, 0.04])

    def test_set_background_does_not_invalidate(self):
        self.dev._add_sphere()
        self.dev.call("render")
        assert self.dev._renderer is not None
        self.dev.call("set_background", 0.5, 0.5, 0.5)
        # Background change does NOT invalidate renderer
        assert self.dev._renderer is not None

    def test_set_background_custom_values(self):
        self.dev._set_background(r=1.0, g=0.0, b=0.0)
        assert np.allclose(self.dev._scene.background, [1.0, 0.0, 0.0])

    def test_set_background_via_call(self):
        self.dev.call("set_background", 0.5, 0.5, 0.5)
        assert np.allclose(self.dev._scene.background, [0.5, 0.5, 0.5])


# ── Set Samples / Resolution ──────────────────────────────────────────


class TestSetSamplesResolution:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_set_samples(self):
        self.dev.call("set_samples", 32)
        assert self.dev._samples == 32
        assert self.dev._renderer is None

    def test_set_resolution(self):
        self.dev.call("set_resolution", 640, 480)
        assert self.dev._width == 640
        assert self.dev._height == 480

    def test_set_resolution_invalidates(self):
        self.dev._add_sphere()
        self.dev.call("render")
        assert self.dev._renderer is not None
        self.dev.call("set_resolution", 100, 100)
        assert self.dev._renderer is None

    def test_set_samples_invalidates(self):
        self.dev._add_sphere()
        self.dev.call("render")
        assert self.dev._renderer is not None
        self.dev.call("set_samples", 64)
        assert self.dev._renderer is None

    def test_set_samples_stores_int(self):
        self.dev._set_samples(n=8)
        assert self.dev._samples == 8

    def test_set_resolution_stores_int(self):
        self.dev._set_resolution(w=320, h=240)
        assert self.dev._width == 320
        assert self.dev._height == 240


# ── Clear ──────────────────────────────────────────────────────────────


class TestClear:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_clear(self):
        self.dev.call("add_sphere")
        self.dev.call("clear")
        assert self.dev.info()["meshes"] == 0
        assert self.dev._renderer is None

    def test_clear_removes_lights(self):
        self.dev._add_light()
        self.dev.call("clear")
        assert self.dev.info()["lights"] == 0

    def test_clear_removes_last_image(self):
        self.dev._add_sphere()
        self.dev._add_light()
        self.dev.call("render")
        assert self.dev._last_image is not None
        self.dev.call("clear")
        assert self.dev._last_image is None

    def test_clear_removes_last_state(self):
        self.dev._add_sphere()
        self.dev._add_light()
        self.dev.call("state_tensors")
        assert self.dev._last_state is not None
        self.dev.call("clear")
        assert self.dev._last_state is None

    def test_clear_resets_scene(self):
        self.dev._add_sphere()
        self.dev._add_cube()
        self.dev._add_light()
        self.dev.call("clear")
        info = self.dev.info()
        assert info["meshes"] == 0
        assert info["lights"] == 0

    def test_clear_empty_device(self):
        self.dev.call("clear")
        assert self.dev.info()["meshes"] == 0


# ── Render ─────────────────────────────────────────────────────────────


class TestRender:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_render(self):
        self.dev._add_sphere(radius=0.5)
        self.dev._add_light(y=3.0)
        result = self.dev.call("render")
        assert result.shape == (24, 32, 3)

    def test_render_stores_last_image(self):
        self.dev._add_sphere()
        self.dev._add_light()
        self.dev.call("render")
        assert self.dev._last_image is not None

    def test_render_info_last_render_shape(self):
        self.dev._add_sphere()
        self.dev._add_light()
        self.dev.call("render")
        info = self.dev.info()
        assert info["last_render_shape"] == [24, 32, 3]

    def test_render_returns_float_array(self):
        self.dev._add_sphere()
        self.dev._add_light()
        result = self.dev.call("render")
        assert result.dtype in (np.float32, np.float64)

    def test_render_custom_resolution(self):
        dev = CyclesDevice(width=16, height=12, samples=2)
        dev._add_sphere()
        dev._add_light()
        result = dev.call("render")
        assert result.shape == (12, 16, 3)

    def test_render_overwrites_last_image(self):
        self.dev._add_sphere()
        self.dev._add_light()
        img1 = self.dev.call("render")
        img2 = self.dev.call("render")
        assert self.dev._last_image is not None

    def test_render_state_tensors(self):
        self.dev._add_sphere()
        self.dev._add_light()
        result = self.dev.call("state_tensors")
        assert isinstance(result, dict)


# ── State Tensors ──────────────────────────────────────────────────────


class TestStateTensors:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_state_tensors(self):
        self.dev._add_sphere(radius=0.5)
        self.dev._add_light(y=3.0)
        result = self.dev.call("state_tensors")
        assert isinstance(result, dict)

    def test_state_tensors_stores_last_state(self):
        self.dev._add_sphere()
        self.dev._add_light()
        self.dev.call("state_tensors")
        assert self.dev._last_state is not None

    def test_state_tensors_stores_np_arrays(self):
        self.dev._add_sphere()
        self.dev._add_light()
        result = self.dev.call("state_tensors")
        for v in result.values():
            assert isinstance(v, np.ndarray)

    def test_state_tensors_custom_resolution(self):
        dev = CyclesDevice(width=16, height=12, samples=2)
        dev._add_sphere()
        dev._add_light()
        result = dev.call("state_tensors")
        assert isinstance(result, dict)


# ── Multiple Meshes / Invalidation ─────────────────────────────────────


class TestMultipleMeshes:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_multiple_meshes(self):
        self.dev._add_sphere(cx=0)
        self.dev._add_sphere(cx=2)
        self.dev._add_cube(cx=4)
        assert self.dev.info()["meshes"] == 3

    def test_invalidate_on_edit(self):
        self.dev._add_sphere()
        self.dev.call("render")
        assert self.dev._renderer is not None
        self.dev._add_cube()
        assert self.dev._renderer is None

    def test_add_sphere_after_render_invalidates(self):
        self.dev._add_sphere()
        self.dev._add_light()
        self.dev.call("render")
        self.dev._add_sphere(cx=5)
        assert self.dev._renderer is None

    def test_add_cube_after_render_invalidates(self):
        self.dev._add_sphere()
        self.dev._add_light()
        self.dev.call("render")
        self.dev._add_cube(cx=5)
        assert self.dev._renderer is None

    def test_add_plane_after_render_invalidates(self):
        self.dev._add_sphere()
        self.dev._add_light()
        self.dev.call("render")
        self.dev._add_plane()
        assert self.dev._renderer is None

    def test_set_material_after_render_invalidates(self):
        self.dev._add_sphere()
        self.dev._add_light()
        self.dev.call("render")
        self.dev.call("set_material", 0)
        assert self.dev._renderer is None

    def test_set_camera_after_render_invalidates(self):
        self.dev._add_sphere()
        self.dev._add_light()
        self.dev.call("render")
        self.dev.call("set_camera")
        assert self.dev._renderer is None


# ── Ensure Renderer ────────────────────────────────────────────────────


class TestEnsureRenderer:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_ensure_renderer_creates(self):
        assert self.dev._renderer is None
        self.dev._ensure_renderer()
        assert self.dev._renderer is not None

    def test_ensure_renderer_reuses(self):
        self.dev._ensure_renderer()
        r1 = self.dev._renderer
        self.dev._ensure_renderer()
        assert self.dev._renderer is r1

    def test_ensure_renderer_recreates_after_invalidation(self):
        self.dev._ensure_renderer()
        r1 = self.dev._renderer
        self.dev._add_sphere()
        self.dev._ensure_renderer()
        assert self.dev._renderer is not None
        assert self.dev._renderer is not r1

    def test_ensure_renderer_after_clear(self):
        self.dev._ensure_renderer()
        self.dev.call("clear")
        self.dev._ensure_renderer()
        assert self.dev._renderer is not None

    def test_ensure_renderer_after_samples_change(self):
        self.dev._ensure_renderer()
        r1 = self.dev._renderer
        self.dev.call("set_samples", 64)
        self.dev._ensure_renderer()
        assert self.dev._renderer is not r1

    def test_ensure_renderer_after_resolution_change(self):
        self.dev._ensure_renderer()
        r1 = self.dev._renderer
        self.dev.call("set_resolution", 64, 48)
        self.dev._ensure_renderer()
        assert self.dev._renderer is not r1
