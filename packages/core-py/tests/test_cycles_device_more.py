"""Coverage tests for CyclesDevice (domains.shell.cycles_device).

Comprehensive coverage of all device operations, scene mutations,
error handling, state tensor generation, and edge cases.
"""

import numpy as np
import pytest

from domains.shell.cycles_device import CyclesDevice
from domains.shell.vm import DeviceFault


# ── Initialization ───────────────────────────────────────────────────────────

class TestInit:
    def test_default_resolution(self):
        d = CyclesDevice()
        info = d.call("info")
        assert info["resolution"] == [160, 120]

    def test_default_samples(self):
        d = CyclesDevice()
        info = d.call("info")
        assert info["samples"] == 16

    def test_custom_resolution(self):
        d = CyclesDevice(width=320, height=240)
        info = d.call("info")
        assert info["resolution"] == [320, 240]

    def test_custom_samples(self):
        d = CyclesDevice(samples=32)
        info = d.call("info")
        assert info["samples"] == 32

    def test_initial_state_no_meshes(self):
        d = CyclesDevice()
        info = d.call("info")
        assert info["meshes"] == 0

    def test_initial_state_no_lights(self):
        d = CyclesDevice()
        info = d.call("info")
        assert info["lights"] == 0

    def test_initial_state_has_materials(self):
        d = CyclesDevice()
        info = d.call("info")
        # Default scene has one default material
        assert info["materials"] >= 1

    def test_initial_no_renderer(self):
        d = CyclesDevice()
        assert d._renderer is None

    def test_initial_no_last_image(self):
        d = CyclesDevice()
        assert d._last_image is None

    def test_initial_no_last_state(self):
        d = CyclesDevice()
        assert d._last_state is None


# ── Info ─────────────────────────────────────────────────────────────────────

class TestInfo:
    def test_info_type(self):
        d = CyclesDevice()
        info = d.call("info")
        assert info["type"] == "cycles"

    def test_info_ops_list(self):
        d = CyclesDevice()
        info = d.call("info")
        expected_ops = {
            "render", "state_tensors", "add_sphere", "add_cube", "add_plane",
            "add_light", "set_camera", "set_material", "set_background",
            "info", "clear", "set_samples", "set_resolution",
        }
        assert set(info["ops"]) == expected_ops

    def test_info_last_render_shape_none(self):
        d = CyclesDevice()
        info = d.call("info")
        assert info["last_render_shape"] is None

    def test_info_after_render(self):
        d = CyclesDevice(width=4, height=3, samples=1)
        d.call("render")
        info = d.call("info")
        assert info["last_render_shape"] == [3, 4, 3]

    def test_info_meshes_count(self):
        d = CyclesDevice()
        d.call("add_sphere")
        d.call("add_cube")
        info = d.call("info")
        assert info["meshes"] == 2

    def test_info_lights_count(self):
        d = CyclesDevice()
        d.call("add_light")
        d.call("add_light")
        info = d.call("info")
        assert info["lights"] == 2


# ── Unknown op ───────────────────────────────────────────────────────────────

class TestUnknownOp:
    def test_unknown_op_raises_device_fault(self):
        d = CyclesDevice()
        with pytest.raises(DeviceFault):
            d.call("bogus")

    def test_empty_string_op_raises(self):
        d = CyclesDevice()
        with pytest.raises(DeviceFault):
            d.call("")

    def test_none_op_raises(self):
        d = CyclesDevice()
        with pytest.raises((DeviceFault, TypeError)):
            d.call(None)


# ── Render ───────────────────────────────────────────────────────────────────

class TestRender:
    def test_render_shape(self):
        d = CyclesDevice(width=8, height=6, samples=1)
        d.call("add_light")
        img = d.call("render")
        assert img.shape == (6, 8, 3)

    def test_render_dtype(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        img = d.call("render")
        assert img.dtype == np.float32

    def test_render_pixel_range(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        img = d.call("render")
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_render_updates_last_image(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("add_light")
        img = d.call("render")
        assert d._last_image is img

    def test_render_updates_info_last_render_shape(self):
        d = CyclesDevice(width=5, height=3, samples=1)
        d.call("render")
        info = d.call("info")
        assert info["last_render_shape"] == [3, 5, 3]

    def test_render_reuses_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        renderer = d._renderer
        d.call("render")
        assert d._renderer is renderer

    def test_render_with_light(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("add_light")
        img = d.call("render")
        assert img.shape == (4, 4, 3)

    def test_render_empty_scene(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        img = d.call("render")
        assert img.shape == (4, 4, 3)


# ── State tensors ────────────────────────────────────────────────────────────

class TestStateTensors:
    def test_state_tensor_keys(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("add_light")
        state = d.call("state_tensors")
        for key in ("image", "depth", "normal", "albedo", "emission", "mask"):
            assert key in state

    def test_state_tensor_types(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("add_light")
        state = d.call("state_tensors")
        for key in ("image", "depth", "normal", "albedo", "emission"):
            assert isinstance(state[key], np.ndarray)
        assert isinstance(state["mask"], np.ndarray)

    def test_state_image_shape(self):
        d = CyclesDevice(width=8, height=6, samples=1)
        state = d.call("state_tensors")
        assert state["image"].shape == (6, 8, 3)

    def test_state_depth_shape(self):
        d = CyclesDevice(width=8, height=6, samples=1)
        state = d.call("state_tensors")
        assert state["depth"].shape == (6, 8)

    def test_state_normal_shape(self):
        d = CyclesDevice(width=8, height=6, samples=1)
        state = d.call("state_tensors")
        assert state["normal"].shape == (6, 8, 3)

    def test_state_albedo_shape(self):
        d = CyclesDevice(width=8, height=6, samples=1)
        state = d.call("state_tensors")
        assert state["albedo"].shape == (6, 8, 3)

    def test_state_emission_shape(self):
        d = CyclesDevice(width=8, height=6, samples=1)
        state = d.call("state_tensors")
        assert state["emission"].shape == (6, 8, 3)

    def test_state_mask_shape(self):
        d = CyclesDevice(width=8, height=6, samples=1)
        state = d.call("state_tensors")
        assert state["mask"].shape == (6, 8)

    def test_state_mask_dtype(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        state = d.call("state_tensors")
        assert state["mask"].dtype == np.int32

    def test_state_stored_on_device(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        state = d.call("state_tensors")
        assert d._last_state is state


# ── Add sphere ───────────────────────────────────────────────────────────────

class TestAddSphere:
    def test_add_sphere_returns_index(self):
        d = CyclesDevice()
        idx = d.call("add_sphere", 0.5, 0, 0, 0, 0, 8)
        assert idx[0] == 0

    def test_add_sphere_creates_mesh(self):
        d = CyclesDevice()
        d.call("add_sphere")
        assert d._scene.meshes[0] is not None

    def test_add_sphere_increments_count(self):
        d = CyclesDevice()
        d.call("add_sphere")
        assert d.call("info")["meshes"] == 1

    def test_add_multiple_spheres(self):
        d = CyclesDevice()
        d.call("add_sphere")
        d.call("add_sphere")
        d.call("add_sphere")
        assert d.call("info")["meshes"] == 3

    def test_add_sphere_default_params(self):
        d = CyclesDevice()
        idx = d.call("add_sphere")
        assert idx[0] == 0

    def test_add_sphere_with_material(self):
        d = CyclesDevice()
        idx = d.call("add_sphere", 0.5, 0, 0, 0, 1)
        mesh = d._scene.meshes[0]
        assert mesh.material_idx[0] == 1

    def test_add_sphere_custom_segments(self):
        d = CyclesDevice()
        idx = d.call("add_sphere", 0.5, 0, 0, 0, 0, 32)
        assert idx[0] == 0

    def test_add_sphere_invalidates_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        assert d._renderer is not None
        d.call("add_sphere")
        assert d._renderer is None


# ── Add cube ─────────────────────────────────────────────────────────────────

class TestAddCube:
    def test_add_cube_returns_index(self):
        d = CyclesDevice()
        idx = d.call("add_cube", 1.0, 0, 0, 0, 0)
        assert idx[0] == 0

    def test_add_cube_creates_mesh(self):
        d = CyclesDevice()
        d.call("add_cube")
        assert d.call("info")["meshes"] == 1

    def test_add_multiple_cubes(self):
        d = CyclesDevice()
        d.call("add_cube")
        d.call("add_cube")
        assert d.call("info")["meshes"] == 2

    def test_add_cube_default_params(self):
        d = CyclesDevice()
        idx = d.call("add_cube")
        assert idx[0] == 0

    def test_add_cube_with_material(self):
        d = CyclesDevice()
        d.call("add_cube", 1.0, 0, 0, 0, 2)
        mesh = d._scene.meshes[0]
        assert mesh.material_idx[0] == 2

    def test_add_cube_invalidates_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("add_cube")
        assert d._renderer is None


# ── Add plane ────────────────────────────────────────────────────────────────

class TestAddPlane:
    def test_add_plane_returns_index(self):
        d = CyclesDevice()
        idx = d.call("add_plane", 2.0, -1.0, 0)
        assert idx[0] == 0

    def test_add_plane_creates_mesh(self):
        d = CyclesDevice()
        d.call("add_plane")
        assert d.call("info")["meshes"] == 1

    def test_add_multiple_planes(self):
        d = CyclesDevice()
        d.call("add_plane")
        d.call("add_plane")
        assert d.call("info")["meshes"] == 2

    def test_add_plane_default_params(self):
        d = CyclesDevice()
        idx = d.call("add_plane")
        assert idx[0] == 0

    def test_add_plane_with_material(self):
        d = CyclesDevice()
        d.call("add_plane", 2.0, -1.0, 3)
        mesh = d._scene.meshes[0]
        assert mesh.material_idx[0] == 3

    def test_add_plane_invalidates_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("add_plane")
        assert d._renderer is None


# ── Add light ────────────────────────────────────────────────────────────────

class TestAddLight:
    def test_add_light_returns_index(self):
        d = CyclesDevice()
        idx = d.call("add_light", 0, 3, 0, 1, 1, 1, 2.0)
        assert idx[0] == 0

    def test_add_light_in_scene(self):
        d = CyclesDevice()
        d.call("add_light")
        assert len(d._scene.lights) == 1

    def test_add_light_strength(self):
        d = CyclesDevice()
        d.call("add_light", 0, 3, 0, 1, 1, 1, 2.0)
        assert d._scene.lights[0].strength == 2.0

    def test_add_light_position(self):
        d = CyclesDevice()
        d.call("add_light", 1.0, 2.0, 3.0)
        np.testing.assert_array_equal(
            d._scene.lights[0].position, np.array([1.0, 2.0, 3.0])
        )

    def test_add_light_color(self):
        d = CyclesDevice()
        d.call("add_light", 0, 0, 0, 0.5, 0.3, 0.1)
        np.testing.assert_array_equal(
            d._scene.lights[0].color, np.array([0.5, 0.3, 0.1])
        )

    def test_add_multiple_lights(self):
        d = CyclesDevice()
        d.call("add_light")
        d.call("add_light")
        d.call("add_light")
        assert len(d._scene.lights) == 3

    def test_add_light_default_params(self):
        d = CyclesDevice()
        idx = d.call("add_light")
        assert idx[0] == 0
        light = d._scene.lights[0]
        np.testing.assert_array_equal(light.position, np.array([0.0, 3.0, 0.0]))
        np.testing.assert_array_equal(light.color, np.array([1.0, 1.0, 1.0]))
        assert light.strength == 1.0

    def test_add_light_does_not_invalidate_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        assert d._renderer is not None
        d.call("add_light")
        # Lights don't invalidate renderer
        assert d._renderer is not None


# ── Set camera ───────────────────────────────────────────────────────────────

class TestSetCamera:
    def test_set_camera_origin(self):
        d = CyclesDevice()
        d.call("set_camera", 0, 1.5, 4, 0, 0, 0, 50)
        cam = d._scene.camera
        np.testing.assert_array_equal(cam.origin, np.array([0, 1.5, 4]))

    def test_set_camera_look_at(self):
        d = CyclesDevice()
        d.call("set_camera", 0, 0, 0, 1, 2, 3)
        cam = d._scene.camera
        np.testing.assert_array_equal(cam.look_at, np.array([1, 2, 3]))

    def test_set_camera_fov(self):
        d = CyclesDevice()
        d.call("set_camera", 0, 0, 0, 0, 0, 0, 90)
        cam = d._scene.camera
        assert cam.fov == 90

    def test_set_camera_default_params(self):
        d = CyclesDevice()
        d.call("set_camera")
        cam = d._scene.camera
        np.testing.assert_array_equal(cam.origin, np.array([0.0, 1.5, 4.0]))
        np.testing.assert_array_equal(cam.look_at, np.array([0.0, 0.0, 0.0]))
        assert cam.fov == 50.0

    def test_set_camera_invalidates_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        assert d._renderer is not None
        d.call("set_camera")
        assert d._renderer is None


# ── Set background ───────────────────────────────────────────────────────────

class TestSetBackground:
    def test_set_background(self):
        d = CyclesDevice()
        d.call("set_background", 0.1, 0.2, 0.3)
        np.testing.assert_array_equal(d._scene.background, np.array([0.1, 0.2, 0.3]))

    def test_set_background_default(self):
        d = CyclesDevice()
        d.call("set_background")
        np.testing.assert_array_equal(d._scene.background, np.array([0.02, 0.02, 0.04]))

    def test_set_background_zeros(self):
        d = CyclesDevice()
        d.call("set_background", 0.0, 0.0, 0.0)
        np.testing.assert_array_equal(d._scene.background, np.zeros(3))

    def test_set_background_ones(self):
        d = CyclesDevice()
        d.call("set_background", 1.0, 1.0, 1.0)
        np.testing.assert_array_equal(d._scene.background, np.ones(3))

    def test_set_background_does_not_invalidate_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("set_background", 0.5, 0.5, 0.5)
        # Background changes don't invalidate renderer
        assert d._renderer is not None


# ── Set material ─────────────────────────────────────────────────────────────

class TestSetMaterial:
    def test_set_material_grows_list(self):
        d = CyclesDevice()
        d.call("set_material", 5, 1, 0, 0)
        assert len(d._scene.materials) >= 6

    def test_set_material_base_color(self):
        d = CyclesDevice()
        d.call("set_material", 0, 0.9, 0.1, 0.1, 0.5, 0.2, 0.3, 0.1, 1.5)
        mat = d._scene.materials[0]
        np.testing.assert_array_equal(mat.base_color, np.array([0.9, 0.1, 0.1]))

    def test_set_material_metallic(self):
        d = CyclesDevice()
        d.call("set_material", 0, 0.8, 0.8, 0.8, 1.0)
        assert d._scene.materials[0].metallic == 1.0

    def test_set_material_roughness(self):
        d = CyclesDevice()
        d.call("set_material", 0, 0.8, 0.8, 0.8, 0.0, 0.9)
        assert d._scene.materials[0].roughness == 0.9

    def test_set_material_emission(self):
        d = CyclesDevice()
        d.call("set_material", 0, 0.8, 0.8, 0.8, 0.0, 0.5, 2.0)
        assert d._scene.materials[0].emission_strength == 2.0

    def test_set_material_transmission(self):
        d = CyclesDevice()
        d.call("set_material", 0, 0.8, 0.8, 0.8, 0.0, 0.5, 0.0, 0.8)
        assert d._scene.materials[0].transmission == 0.8

    def test_set_material_ior(self):
        d = CyclesDevice()
        d.call("set_material", 0, 0.8, 0.8, 0.8, 0.0, 0.5, 0.0, 0.0, 2.0)
        assert d._scene.materials[0].ior == 2.0

    def test_set_material_default_values(self):
        d = CyclesDevice()
        d.call("set_material", 0)
        mat = d._scene.materials[0]
        np.testing.assert_array_equal(mat.base_color, np.array([0.8, 0.8, 0.8]))
        assert mat.metallic == 0.0
        assert mat.roughness == 0.5
        assert mat.emission_strength == 0.0
        assert mat.transmission == 0.0
        assert mat.ior == 1.45

    def test_set_material_invalidates_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("set_material", 0, 1.0, 0.0, 0.0)
        assert d._renderer is None


# ── Clear ────────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_resets_scene(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("add_sphere")
        d.call("render")
        d.call("clear")
        assert len(d._scene.meshes) == 0

    def test_clear_resets_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("clear")
        assert d._renderer is None

    def test_clear_resets_last_image(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("clear")
        assert d._last_image is None

    def test_clear_resets_last_state(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("state_tensors")
        d.call("clear")
        assert d._last_state is None

    def test_clear_resets_info_last_render_shape(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("clear")
        info = d.call("info")
        assert info["last_render_shape"] is None

    def test_clear_resets_lights(self):
        d = CyclesDevice()
        d.call("add_light")
        d.call("add_light")
        d.call("clear")
        assert len(d._scene.lights) == 0

    def test_clear_then_render_works(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("clear")
        d.call("add_light")
        img = d.call("render")
        assert img.shape == (4, 4, 3)


# ── Set samples ──────────────────────────────────────────────────────────────

class TestSetSamples:
    def test_set_samples(self):
        d = CyclesDevice()
        d.call("set_samples", 4)
        assert d._samples == 4

    def test_set_samples_invalidates_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("set_samples", 32)
        assert d._renderer is None

    def test_set_samples_int_conversion(self):
        d = CyclesDevice()
        d.call("set_samples", 16.0)
        assert d._samples == 16

    def test_set_samples_updates_info(self):
        d = CyclesDevice()
        d.call("set_samples", 64)
        info = d.call("info")
        assert info["samples"] == 64


# ── Set resolution ───────────────────────────────────────────────────────────

class TestSetResolution:
    def test_set_resolution(self):
        d = CyclesDevice()
        d.call("set_resolution", 32, 24)
        assert d._width == 32
        assert d._height == 24

    def test_set_resolution_invalidates_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("set_resolution", 8, 8)
        assert d._renderer is None

    def test_set_resolution_int_conversion(self):
        d = CyclesDevice()
        d.call("set_resolution", 32.0, 24.0)
        assert d._width == 32
        assert d._height == 24

    def test_set_resolution_updates_info(self):
        d = CyclesDevice()
        d.call("set_resolution", 100, 50)
        info = d.call("info")
        assert info["resolution"] == [100, 50]

    def test_set_resolution_affects_render_shape(self):
        d = CyclesDevice(samples=1)
        d.call("add_light")
        d.call("set_resolution", 10, 8)
        img = d.call("render")
        assert img.shape == (8, 10, 3)


# ── Renderer invalidation ───────────────────────────────────────────────────

class TestInvalidation:
    def test_add_sphere_invalidates(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        assert d._renderer is not None
        d.call("add_sphere")
        assert d._renderer is None

    def test_add_cube_invalidates(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("add_cube")
        assert d._renderer is None

    def test_add_plane_invalidates(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("add_plane")
        assert d._renderer is None

    def test_set_camera_invalidates(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("set_camera")
        assert d._renderer is None

    def test_set_material_invalidates(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("set_material", 0)
        assert d._renderer is None

    def test_set_samples_invalidates(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("set_samples", 8)
        assert d._renderer is None

    def test_set_resolution_invalidates(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("set_resolution", 8, 8)
        assert d._renderer is None

    def test_add_light_does_not_invalidate(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("add_light")
        assert d._renderer is not None

    def test_set_background_does_not_invalidate(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        d.call("set_background", 0.5, 0.5, 0.5)
        assert d._renderer is not None

    def test_mutation_cycle(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        assert d._renderer is not None
        d.call("add_sphere")
        assert d._renderer is None
        d.call("render")
        assert d._renderer is not None


# ── Scene composition ────────────────────────────────────────────────────────

class TestSceneComposition:
    def test_full_scene(self):
        d = CyclesDevice(width=8, height=6, samples=1)
        d.call("add_sphere", 0.5, 0, 0, 0)
        d.call("add_plane", 4.0, -1.0)
        d.call("add_light", 2, 3, 2, 1, 1, 1, 5.0)
        d.call("set_camera", 0, 1.5, 4, 0, 0, 0, 50)
        d.call("set_background", 0.1, 0.1, 0.2)
        d.call("set_material", 0, 0.8, 0.2, 0.2)
        img = d.call("render")
        assert img.shape == (6, 8, 3)

    def test_multiple_meshes_render(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("add_sphere")
        d.call("add_cube")
        d.call("add_plane")
        d.call("add_light")
        img = d.call("render")
        assert img.shape == (4, 4, 3)

    def test_render_after_clear_and_rebuild(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("add_sphere")
        d.call("render")
        d.call("clear")
        d.call("add_light")
        img = d.call("render")
        assert img.shape == (4, 4, 3)


# ── Negative/edge indices ───────────────────────────────────────────────────

class TestEdgeCases:
    def test_add_sphere_negative_position(self):
        d = CyclesDevice()
        idx = d.call("add_sphere", 0.5, -1.0, -2.0, -3.0)
        assert idx[0] == 0

    def test_add_cube_negative_position(self):
        d = CyclesDevice()
        idx = d.call("add_cube", 1.0, -1.0, -2.0, -3.0)
        assert idx[0] == 0

    def test_add_light_negative_position(self):
        d = CyclesDevice()
        d.call("add_light", -5, -5, -5, 1, 0, 0, 10.0)
        light = d._scene.lights[0]
        np.testing.assert_array_equal(light.position, np.array([-5.0, -5.0, -5.0]))
        assert light.strength == 10.0

    def test_set_camera_extreme_fov(self):
        d = CyclesDevice()
        d.call("set_camera", 0, 0, 0, 0, 0, 0, 179)
        assert d._scene.camera.fov == 179

    def test_add_sphere_large_segments(self):
        d = CyclesDevice()
        idx = d.call("add_sphere", 0.5, 0, 0, 0, 0, 64)
        assert idx[0] == 0
        assert d.call("info")["meshes"] == 1

    def test_add_sphere_small_segments(self):
        d = CyclesDevice()
        idx = d.call("add_sphere", 0.5, 0, 0, 0, 0, 4)
        assert idx[0] == 0

    def test_set_material_high_emission(self):
        d = CyclesDevice()
        d.call("set_material", 0, 1.0, 0.0, 0.0, 0.0, 0.0, 100.0)
        assert d._scene.materials[0].emission_strength == 100.0

    def test_set_material_high_ior(self):
        d = CyclesDevice()
        d.call("set_material", 0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 5.0)
        assert d._scene.materials[0].ior == 5.0

    def test_add_light_zero_strength(self):
        d = CyclesDevice()
        d.call("add_light", 0, 3, 0, 1, 1, 1, 0.0)
        assert d._scene.lights[0].strength == 0.0

    def test_set_background_beyond_one(self):
        d = CyclesDevice()
        d.call("set_background", 2.0, 2.0, 2.0)
        np.testing.assert_array_equal(d._scene.background, np.array([2.0, 2.0, 2.0]))

    def test_set_resolution_1x1(self):
        d = CyclesDevice(samples=1)
        d.call("set_resolution", 1, 1)
        img = d.call("render")
        assert img.shape == (1, 1, 3)

    def test_state_tensors_after_render(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        state = d.call("state_tensors")
        assert state["image"].shape == (4, 4, 3)

    def test_multiple_render_accumulates(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("add_light")
        d.call("render")
        d.call("render")
        d.call("render")
        info = d.call("info")
        assert info["last_render_shape"] == [4, 4, 3]

    def test_clear_preserves_resolution(self):
        d = CyclesDevice(width=32, height=24, samples=8)
        d.call("clear")
        info = d.call("info")
        assert info["resolution"] == [32, 24]
        assert info["samples"] == 8

    def test_add_sphere_stores_mesh(self):
        d = CyclesDevice()
        idx = d.call("add_sphere", 1.0, 0, 0, 0, 0, 16)
        assert d._scene.meshes[0] is not None

    def test_add_cube_stores_mesh(self):
        d = CyclesDevice()
        d.call("add_cube", 2.0, 0, 0, 0, 0)
        assert d._scene.meshes[0] is not None

    def test_add_plane_stores_mesh(self):
        d = CyclesDevice()
        d.call("add_plane", 4.0, 0, 0)
        assert d._scene.meshes[0] is not None

    def test_call_returns_operation(self):
        d = CyclesDevice()
        result = d.call("add_sphere")
        assert isinstance(result, np.ndarray)

    def test_multiple_materials_independent(self):
        d = CyclesDevice()
        d.call("set_material", 0, 1.0, 0.0, 0.0)
        d.call("set_material", 1, 0.0, 1.0, 0.0)
        assert len(d._scene.materials) >= 2
        np.testing.assert_array_equal(d._scene.materials[0].base_color, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(d._scene.materials[1].base_color, np.array([0.0, 1.0, 0.0]))

    def test_set_material_full_chain(self):
        d = CyclesDevice()
        d.call("set_material", 0, 0.9, 0.1, 0.1, 0.5, 0.2, 0.3, 0.1, 1.5)
        mat = d._scene.materials[0]
        np.testing.assert_array_equal(mat.base_color, np.array([0.9, 0.1, 0.1]))
        assert mat.metallic == 0.5
        assert mat.roughness == 0.2
        assert mat.emission_strength == 0.3
        assert mat.transmission == 0.1
        assert mat.ior == 1.5

    def test_ops_list_complete(self):
        d = CyclesDevice()
        info = d.call("info")
        assert len(info["ops"]) == 13

    def test_add_sphere_default_material(self):
        d = CyclesDevice()
        d.call("add_sphere")
        mesh = d._scene.meshes[0]
        assert mesh.material_idx[0] == 0

    def test_add_cube_default_material(self):
        d = CyclesDevice()
        d.call("add_cube")
        mesh = d._scene.meshes[0]
        assert mesh.material_idx[0] == 0

    def test_add_plane_default_material(self):
        d = CyclesDevice()
        d.call("add_plane")
        mesh = d._scene.meshes[0]
        assert mesh.material_idx[0] == 0
