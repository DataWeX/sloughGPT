"""Coverage tests for CyclesDevice (domains.shell.cycles_device)."""

import numpy as np
import pytest

from domains.shell.cycles_device import CyclesDevice
from domains.shell.vm import DeviceFault


class TestCyclesDeviceOps:
    def test_unknown_op_raises(self):
        d = CyclesDevice()
        with pytest.raises(DeviceFault):
            d.call("bogus")

    def test_info_empty(self):
        d = CyclesDevice()
        info = d.call("info")
        assert info["type"] == "cycles"
        assert info["resolution"] == [160, 120]
        assert info["samples"] == 16
        assert info["meshes"] == 0
        assert info["last_render_shape"] is None
        assert "render" in info["ops"]
        assert "clear" in info["ops"]

    def test_render(self):
        d = CyclesDevice(width=8, height=6, samples=1)
        d.call("add_light")
        img = d.call("render")
        assert img.shape == (6, 8, 3)
        info = d.call("info")
        assert info["last_render_shape"] == [6, 8, 3]

    def test_render_uses_existing_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        renderer = d._renderer
        d.call("render")
        assert d._renderer is renderer

    def test_state_tensors(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("add_light")
        state = d.call("state_tensors")
        for key in ("image", "depth", "normal", "albedo", "emission", "mask"):
            assert key in state
            assert isinstance(state[key], np.ndarray)


class TestSceneMutation:
    def test_add_sphere(self):
        d = CyclesDevice()
        idx = d.call("add_sphere", 0.5, 0, 0, 0, 0, 8)
        assert idx[0] == 0
        assert d._scene.meshes[0] is not None
        assert d.call("info")["meshes"] == 1

    def test_add_cube(self):
        d = CyclesDevice()
        idx = d.call("add_cube", 1.0, 0, 0, 0, 0)
        assert idx[0] == 0
        assert d.call("info")["meshes"] == 1

    def test_add_plane(self):
        d = CyclesDevice()
        idx = d.call("add_plane", 2.0, -1.0, 0)
        assert idx[0] == 0
        assert d.call("info")["meshes"] == 1

    def test_add_light(self):
        d = CyclesDevice()
        idx = d.call("add_light", 0, 3, 0, 1, 1, 1, 2.0)
        assert idx[0] == 0
        assert len(d._scene.lights) == 1
        assert d._scene.lights[0].strength == 2.0

    def test_set_camera(self):
        d = CyclesDevice()
        d.call("set_camera", 0, 1.5, 4, 0, 0, 0, 50)
        cam = d._scene.camera
        assert cam is not None
        np.testing.assert_array_equal(cam.origin, np.array([0, 1.5, 4]))

    def test_set_background(self):
        d = CyclesDevice()
        d.call("set_background", 0.1, 0.2, 0.3)
        np.testing.assert_array_equal(d._scene.background, np.array([0.1, 0.2, 0.3]))

    def test_set_material_existing(self):
        d = CyclesDevice()
        d.call("set_material", 0, 0.9, 0.1, 0.1, 0.5, 0.2, 0.3, 0.1, 1.5)
        mat = d._scene.materials[0]
        np.testing.assert_array_equal(mat.base_color, np.array([0.9, 0.1, 0.1]))
        assert mat.metallic == 0.5
        assert mat.roughness == 0.2
        assert mat.emission_strength == 0.3
        assert mat.transmission == 0.1
        assert mat.ior == 1.5

    def test_set_material_grows_list(self):
        d = CyclesDevice()
        d.call("set_material", 5, 1, 0, 0)
        assert len(d._scene.materials) >= 6

    def test_clear(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("add_sphere")
        d.call("render")
        d.call("clear")
        assert len(d._scene.meshes) == 0
        assert d._renderer is None
        assert d._last_image is None
        assert d._last_state is None
        assert d.call("info")["last_render_shape"] is None

    def test_set_samples(self):
        d = CyclesDevice()
        d.call("set_samples", 4)
        assert d._samples == 4
        assert d._renderer is None

    def test_set_resolution(self):
        d = CyclesDevice()
        d.call("set_resolution", 32, 24)
        assert d._width == 32
        assert d._height == 24
        assert d._renderer is None

    def test_mutation_invalidates_renderer(self):
        d = CyclesDevice(width=4, height=4, samples=1)
        d.call("render")
        assert d._renderer is not None
        d.call("add_light")
        assert d._renderer is not None
        d.call("add_sphere")
        assert d._renderer is None
        d.call("render")
        assert d._renderer is not None
