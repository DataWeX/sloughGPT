"""Tests for domains.shell.cycles — vector math and data classes."""

import numpy as np
import pytest
from domains.shell.cycles import (
    _normalize, _dot, _cross, _reflect, _refract,
    Material, Mesh, Light, Camera, Scene, BVH,
    create_sphere, create_plane, create_cube,
    _ggx_distribution, _ggx_geometry, _fresnel_schlick,
)


class TestVectorMath:
    def test_normalize_unit_vector(self):
        v = np.array([1.0, 0.0, 0.0])
        assert np.allclose(_normalize(v), [1.0, 0.0, 0.0])

    def test_normalize_diagonal(self):
        v = np.array([1.0, 1.0, 0.0])
        result = _normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_normalize_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        result = _normalize(v)
        assert np.allclose(result, [0.0, 0.0, 0.0])

    def test_dot_perpendicular(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert _dot(a, b) == pytest.approx(0.0)

    def test_dot_parallel(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _dot(a, b) == pytest.approx(14.0)

    def test_cross_perpendicular(self):
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        result = _cross(x, y)
        assert np.allclose(result, [0.0, 0.0, 1.0])

    def test_reflect(self):
        I = np.array([1.0, -1.0, 0.0])
        N = np.array([0.0, 1.0, 0.0])
        result = _reflect(I, N)
        assert np.allclose(result, [1.0, 1.0, 0.0])

    def test_refract_total_internal(self):
        # Going from dense (eta=2.0) to less dense, grazing angle → TIR
        I = np.array([0.99, -0.1, 0.0])
        N = np.array([0.0, 1.0, 0.0])
        result, mask = _refract(I, N, 2.0)
        assert not mask  # total internal reflection

    def test_refract_normal_incidence(self):
        I = np.array([0.0, -1.0, 0.0])
        N = np.array([0.0, 1.0, 0.0])
        result, mask = _refract(I, N, 1.0)
        assert mask  # should refract


class TestMaterial:
    def test_default_material(self):
        mat = Material()
        assert mat.metallic == 0.0
        assert mat.roughness == 0.5

    def test_emission_off(self):
        mat = Material()
        assert np.allclose(mat.emission(), [0.0, 0.0, 0.0])

    def test_emission_on(self):
        mat = Material()
        mat.emission_strength = 2.0
        mat.base_color = np.array([1.0, 0.5, 0.0])
        result = mat.emission()
        assert result[0] == pytest.approx(2.0)


class TestMesh:
    def test_create_sphere(self):
        mesh = create_sphere(radius=1.0, segments=8)
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.shape[1] == 3
        assert len(mesh.vertices) > 0

    def test_create_plane(self):
        mesh = create_plane(size=2.0)
        assert mesh.vertices.shape[1] == 3

    def test_create_cube(self):
        mesh = create_cube(size=1.0)
        assert mesh.vertices.shape[1] == 3
        assert len(mesh.vertices) == 8  # 8 unique corners, faces index into them


class TestLight:
    def test_light_defaults(self):
        light = Light()
        assert light.strength == 1.0
        assert len(light.position) == 3


class TestCamera:
    def test_generate_rays(self):
        cam = Camera()
        rays_o, rays_d = cam.generate_rays(4, 3)
        assert rays_o.shape == (3, 4, 3)
        assert rays_d.shape == (3, 4, 3)


class TestScene:
    def test_empty_scene(self):
        scene = Scene()
        assert len(scene.meshes) == 0
        assert len(scene.lights) == 0

    def test_add_mesh(self):
        scene = Scene()
        mesh = create_sphere(radius=0.5, segments=4)
        scene.add_mesh(mesh)
        assert len(scene.meshes) == 1


class TestGGX:
    def test_ggx_distribution_peak(self):
        # NdotH=1 (perfect mirror) should give high distribution
        result = _ggx_distribution(np.array(1.0), 0.01)
        assert result > 10  # high peak at perfect mirror angle

    def test_ggx_geometry(self):
        result = _ggx_geometry(0.5, 0.5, 0.25)
        assert 0.0 <= result <= 1.0

    def test_fresnel_grazing(self):
        F0 = np.array([0.04, 0.04, 0.04])
        result = _fresnel_schlick(0.0, F0)
        # At grazing angle, Fresnel approaches 1
        assert result[0] > 0.9


class TestBVH:
    def test_build_bvh(self):
        scene = Scene()
        mesh = create_sphere(radius=0.5, segments=4)
        scene.add_mesh(mesh)
        bvh = BVH(mesh)
        assert bvh is not None
