"""Tests for domains.shell.cycles_device — CyclesDevice VM device wrapper."""

import numpy as np
import pytest
from domains.shell.cycles_device import CyclesDevice
from domains.shell.vm import DeviceFault


class TestCyclesDevice:
    def setup_method(self):
        self.dev = CyclesDevice(width=32, height=24, samples=4)

    def test_info(self):
        info = self.dev.info()
        assert info["type"] == "cycles"
        assert info["resolution"] == [32, 24]
        assert info["samples"] == 4
        assert info["meshes"] == 0
        assert info["lights"] == 0

    def test_call_unknown_op(self):
        with pytest.raises(DeviceFault, match="unknown op"):
            self.dev.call("nonexistent")

    def test_add_sphere(self):
        result = self.dev.call("add_sphere", 0.5, 0.0, 0.0, 0.0, 0, 16)
        assert result[0] == 0
        assert self.dev.info()["meshes"] == 1

    def test_add_sphere_defaults(self):
        result = self.dev._add_sphere()
        assert result[0] == 0

    def test_add_cube(self):
        result = self.dev.call("add_cube", 2.0)
        assert result[0] == 0
        assert self.dev.info()["meshes"] == 1

    def test_add_plane(self):
        result = self.dev.call("add_plane")
        assert result[0] == 0

    def test_add_light(self):
        result = self.dev.call("add_light", 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0)
        assert result[0] == 0
        assert self.dev.info()["lights"] == 1

    def test_add_light_defaults(self):
        result = self.dev._add_light()
        assert result[0] == 0

    def test_set_camera(self):
        self.dev.call("set_camera", 0.0, 1.5, 4.0, 0.0, 0.0, 0.0, 50.0)

    def test_set_material(self):
        self.dev.call("add_sphere")
        self.dev.call("set_material", 0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.45)
        assert len(self.dev._scene.materials) >= 1

    def test_set_background(self):
        self.dev.call("set_background", 0.1, 0.2, 0.3)
        assert np.allclose(self.dev._scene.background, [0.1, 0.2, 0.3])

    def test_set_samples(self):
        self.dev.call("set_samples", 32)
        assert self.dev._samples == 32
        assert self.dev._renderer is None

    def test_set_resolution(self):
        self.dev.call("set_resolution", 640, 480)
        assert self.dev._width == 640
        assert self.dev._height == 480

    def test_clear(self):
        self.dev.call("add_sphere")
        self.dev.call("clear")
        assert self.dev.info()["meshes"] == 0
        assert self.dev._renderer is None

    def test_render(self):
        self.dev._add_sphere(radius=0.5)
        self.dev._add_light(y=3.0)
        result = self.dev.call("render")
        assert result.shape == (24, 32, 3)

    def test_state_tensors(self):
        self.dev._add_sphere(radius=0.5)
        self.dev._add_light(y=3.0)
        result = self.dev.call("state_tensors")
        assert isinstance(result, dict)

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
