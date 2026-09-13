"""
CyclesDevice — VM device wrapper for the Cycles path tracer.

Bridges the RAM-resident rendering engine onto the VM device bus so
assembly programs and the NPU can render scenes and process state tensors.

Device bus protocol:
    DEV_OPEN   R0, cycles
    DEV_CALL   R1, R0, render          # R1 = image tensor (H*W*3)
    DEV_CALL   R1, R0, state_tensors   # R1 = dict of state tensors
    DEV_CALL   R1, R0, add_sphere, ...
    DEV_CALL   R1, R0, add_light, ...
    DEV_CALL   R1, R0, set_material, ...
    DEV_CALL   R1, R0, set_camera, ...
    DEV_CALL   R1, R0, info
"""

from __future__ import annotations

import numpy as np

from .vm import Device, DeviceFault
from .cycles import (
    CyclesRenderer, Scene, Camera, Material, Light,
    create_sphere, create_plane, create_cube,
)


class CyclesDevice(Device):
    """VM device that wraps the Cycles path tracer."""

    def __init__(self, width: int = 160, height: int = 120, samples: int = 16):
        self._scene = Scene()
        self._width = width
        self._height = height
        self._samples = samples
        self._renderer: CyclesRenderer | None = None
        self._last_image: np.ndarray | None = None
        self._last_state: dict[str, np.ndarray] | None = None

        self._ops = {
            "render": self._render,
            "state_tensors": self._state_tensors,
            "add_sphere": self._add_sphere,
            "add_cube": self._add_cube,
            "add_plane": self._add_plane,
            "add_light": self._add_light,
            "set_camera": self._set_camera,
            "set_material": self._set_material,
            "set_background": self._set_background,
            "info": self.info,
            "clear": self._clear,
            "set_samples": self._set_samples,
            "set_resolution": self._set_resolution,
        }

    def call(self, method, *args):
        fn = self._ops.get(method)
        if fn is None:
            raise DeviceFault(f"CyclesDevice: unknown op: {method}")
        return fn(*args)

    def info(self):
        return {
            "type": "cycles",
            "ops": list(self._ops.keys()),
            "resolution": [self._width, self._height],
            "samples": self._samples,
            "meshes": len(self._scene.meshes),
            "lights": len(self._scene.lights),
            "materials": len(self._scene.materials),
            "last_render_shape": list(self._last_image.shape) if self._last_image is not None else None,
        }

    def _render(self):
        self._ensure_renderer()
        self._scene.build_bvh()
        self._last_image = self._renderer.render()
        return self._last_image

    def _state_tensors(self):
        self._ensure_renderer()
        self._scene.build_bvh()
        self._last_state = self._renderer.render_state_tensors()
        return self._last_state

    def _add_sphere(self, radius=0.5, cx=0.0, cy=0.0, cz=0.0, mat_idx=0, segments=16):
        mesh = create_sphere(radius=radius, center=np.array([cx, cy, cz]),
                             segments=segments, mat_idx=int(mat_idx))
        self._scene.add_mesh(mesh)
        self._invalidate()
        return np.array([len(self._scene.meshes) - 1])

    def _add_cube(self, size=1.0, cx=0.0, cy=0.0, cz=0.0, mat_idx=0):
        mesh = create_cube(size=size, center=np.array([cx, cy, cz]),
                           mat_idx=int(mat_idx))
        self._scene.add_mesh(mesh)
        self._invalidate()
        return np.array([len(self._scene.meshes) - 1])

    def _add_plane(self, size=2.0, y=-1.0, mat_idx=0):
        mesh = create_plane(size=size, y=y, mat_idx=int(mat_idx))
        self._scene.add_mesh(mesh)
        self._invalidate()
        return np.array([len(self._scene.meshes) - 1])

    def _add_light(self, x=0.0, y=3.0, z=0.0, r=1.0, g=1.0, b=1.0, strength=1.0):
        light = Light(
            position=np.array([x, y, z]),
            color=np.array([r, g, b]),
            strength=strength,
        )
        self._scene.lights.append(light)
        return np.array([len(self._scene.lights) - 1])

    def _set_camera(self, ox=0.0, oy=1.5, oz=4.0,
                     lx=0.0, ly=0.0, lz=0.0, fov=50.0):
        self._scene.camera = Camera(
            origin=np.array([ox, oy, oz]),
            look_at=np.array([lx, ly, lz]),
            fov=fov,
        )
        self._invalidate()

    def _set_material(self, idx, base_r=0.8, base_g=0.8, base_b=0.8,
                       metallic=0.0, roughness=0.5, emission=0.0,
                       transmission=0.0, ior=1.45):
        idx = int(idx)
        while len(self._scene.materials) <= idx:
            self._scene.materials.append(Material())
        mat = self._scene.materials[idx]
        mat.base_color = np.array([base_r, base_g, base_b])
        mat.metallic = float(metallic)
        mat.roughness = float(roughness)
        mat.emission_strength = float(emission)
        mat.transmission = float(transmission)
        mat.ior = float(ior)
        self._invalidate()

    def _set_background(self, r=0.02, g=0.02, b=0.04):
        self._scene.background = np.array([r, g, b])

    def _clear(self):
        self._scene = Scene()
        self._renderer = None
        self._last_image = None
        self._last_state = None

    def _set_samples(self, n=16):
        self._samples = int(n)
        self._renderer = None

    def _set_resolution(self, w=160, h=120):
        self._width = int(w)
        self._height = int(h)
        self._renderer = None

    def _ensure_renderer(self):
        if self._renderer is None:
            self._renderer = CyclesRenderer(
                self._scene, self._width, self._height, self._samples
            )

    def _invalidate(self):
        self._renderer = None
