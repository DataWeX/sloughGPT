"""Coverage tests for the Cycles path tracer (domains.shell.cycles)."""

import numpy as np
import pytest

from domains.shell.cycles import (
    BVH, Camera, CyclesRenderer, Light, Material, Mesh, Scene,
    create_cube, create_plane, create_sphere,
    _refract, _normalize, _reflect,
)


def _scene_with_mesh(mat_idx=0):
    scene = Scene()
    scene.add_mesh(create_plane(size=3.0, y=-1.0, mat_idx=mat_idx))
    scene.add_mesh(create_sphere(radius=0.6, center=np.array([0.0, 0.0, 0.0]),
                                 segments=12, mat_idx=mat_idx))
    scene.lights.append(Light(position=np.array([0.0, 3.0, 0.0]),
                              color=np.ones(3), strength=2.0))
    return scene


class TestMaterial:
    def test_eval_zero_on_perpendicular(self):
        mat = Material()
        N = np.array([0.0, 1.0, 0.0])
        wo = np.array([0.0, -1.0, 0.0])
        wi = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_equal(mat.eval(wi, wo, N), np.zeros(3))

    def test_eval_diffuse(self):
        mat = Material()
        N = np.array([0.0, 1.0, 0.0])
        wi = _normalize(np.array([0.3, 1.0, 0.2]))
        wo = _normalize(np.array([-0.2, 1.0, 0.4]))
        out = mat.eval(wi, wo, N)
        assert out.shape == (3,)
        assert out.min() >= 0

    def test_eval_metallic(self):
        mat = Material(metallic=1.0, roughness=0.1)
        N = np.array([0.0, 1.0, 0.0])
        wi = _normalize(np.array([0.3, 1.0, 0.2]))
        wo = _normalize(np.array([-0.2, 1.0, 0.4]))
        out = mat.eval(wi, wo, N)
        assert out.shape == (3,)

    def test_emission(self):
        mat = Material(emission_strength=2.0, emission_color=np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(mat.emission(), np.array([2.0, 0.0, 0.0]))


class TestMesh:
    def test_mesh_without_material_idx(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        mesh = Mesh(vertices=verts, faces=faces, normals=np.ones((3, 3)))
        np.testing.assert_array_equal(mesh.material_idx, np.zeros(1, dtype=np.int32))
        assert mesh._aabb_min is not None

    def test_mesh_auto_normals(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = Mesh(vertices=verts, faces=faces)
        assert mesh.normals.shape == (4, 3)

    def test_primitives(self):
        plane = create_plane(size=2.0, y=-1.0, mat_idx=3)
        assert plane.material_idx[0] == 3
        sphere = create_sphere(radius=1.0, segments=6, mat_idx=1)
        assert sphere.vertices.shape[0] > 10
        cube = create_cube(size=1.0, mat_idx=2)
        assert len(cube.faces) == 12

    def test_create_sphere_default_center(self):
        sphere = create_sphere(radius=0.5, segments=6)
        assert sphere is not None

    def test_create_cube_default_center(self):
        cube = create_cube(size=1.0)
        assert cube is not None


class TestCameraAndRefract:
    def test_camera_generate_rays(self):
        cam = Camera()
        origins, dirs = cam.generate_rays(4, 3)
        assert origins.shape == (3, 4, 3)
        assert dirs.shape == (3, 4, 3)
        norms = np.linalg.norm(dirs, axis=-1)
        assert np.allclose(norms, 1.0)

    def test_refract_internal(self):
        N = np.array([0.0, 1.0, 0.0])
        I = np.array([0.0, -1.0, 0.0])
        result, mask = _refract(I, N, 1.5)
        assert mask.dtype == np.bool_
        assert result.shape == (3,)

    def test_reflect(self):
        I = np.array([1.0, -1.0, 0.0])
        N = np.array([0.0, 1.0, 0.0])
        out = _reflect(I, N)
        np.testing.assert_allclose(out, [1.0, 1.0, 0.0], atol=1e-6)


class TestBVH:
    def _plane_bvh(self):
        mesh = create_plane(size=2.0, y=0.0)
        return BVH(mesh)

    def test_build_and_hit(self):
        bvh = self._plane_bvh()
        t, fi = bvh.intersect(np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]))
        assert t < 1e30
        assert fi >= 0

    def test_parallel_ray(self):
        bvh = self._plane_bvh()
        t = bvh._ray_triangle(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0)
        assert t == 1e30

    def test_miss(self):
        bvh = self._plane_bvh()
        t, fi = bvh.intersect(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        assert t >= 1e30
        assert fi == -1

    def test_many_triangles(self):
        scene = _scene_with_mesh()
        scene.build_bvh()
        bvh = scene._bvh_list[0]
        assert bvh._root >= 0
        for i in range(20):
            orig = np.random.rand(3) * 1.5 - 0.75
            orig[1] = 2.0
            dire = np.array([0.0, -1.0, 0.0])
            t, fi = bvh.intersect(orig, dire)
            assert fi >= 0
            assert t < 1e30


class TestRendering:
    def test_render_empty_scene(self):
        scene = Scene()
        img = CyclesRenderer(scene, 8, 6, 1).render()
        assert img.shape == (6, 8, 3)
        assert img.dtype == np.float32

    def test_render_diffuse(self):
        scene = _scene_with_mesh()
        img = CyclesRenderer(scene, 16, 12, 1).render()
        assert img.shape == (12, 16, 3)

    def test_render_metallic(self):
        scene = _scene_with_mesh()
        scene.materials[0].metallic = 1.0
        scene.materials[0].roughness = 0.1
        img = CyclesRenderer(scene, 16, 12, 1).render()
        assert img.shape == (12, 16, 3)

    def test_render_glass(self):
        scene = _scene_with_mesh()
        scene.materials[0].transmission = 1.0
        scene.materials[0].ior = 1.5
        img = CyclesRenderer(scene, 16, 12, 1).render()
        assert img.shape == (12, 16, 3)

    def test_render_multi_sample(self):
        scene = _scene_with_mesh()
        img = CyclesRenderer(scene, 8, 6, 2).render()
        assert img.shape == (6, 8, 3)

    def test_render_state_tensors_with_mesh(self):
        scene = _scene_with_mesh()
        scene.materials[0].emission_strength = 0.5
        r = CyclesRenderer(scene, 16, 12, 1)
        state = r.render_state_tensors()
        for key in ("image", "depth", "normal", "albedo", "emission", "mask"):
            assert key in state
        assert state["depth"].shape == (12, 16)
        assert state["mask"].shape == (12, 16)
        assert state["mask"].max() >= 1
        assert state["depth"].max() < 1e30
        norms = np.linalg.norm(state["normal"][state["mask"] > 0], axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-3)

    def test_trace_single_hit_and_miss(self):
        scene = _scene_with_mesh()
        scene.build_bvh()
        r = CyclesRenderer(scene, 16, 12, 1)
        t, fi, mi = r._trace_single(np.array([0.0, 2.0, 0.0]), np.array([0.0, -1.0, 0.0]))
        assert fi >= 0
        assert mi >= 0
        t2, fi2, _ = r._trace_single(np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        assert fi2 == -1


class TestForcedBranches:
    def test_bsdf_pdf_zero_kills_ray(self, monkeypatch):
        scene = _scene_with_mesh()
        r = CyclesRenderer(scene, 16, 12, 1)
        monkeypatch.setattr(r, "_bsdf_pdf", lambda wo, wi, N, mat: 0.0)
        img = r.render()
        assert img.shape == (12, 16, 3)

    def test_direct_sample_bsdf_diffuse(self):
        r = CyclesRenderer(Scene(), 8, 6, 1)
        mat = Material()
        N = np.array([0.0, 1.0, 0.0])
        wo = _normalize(np.array([-0.2, 1.0, 0.0]))
        wi = r._sample_bsdf(wo, N, mat)
        assert wi.shape == (3,)

    def test_direct_bsdf_pdf_transmission(self):
        r = CyclesRenderer(Scene(), 8, 6, 1)
        mat = Material(transmission=1.0)
        N = np.array([0.0, 1.0, 0.0])
        wo = _normalize(np.array([-0.2, 1.0, 0.0]))
        wi = _normalize(np.array([0.2, 1.0, 0.0]))
        assert r._bsdf_pdf(wo, wi, N, mat) == pytest.approx(1.0 / (4 * np.pi))

    def test_direct_bsdf_pdf_metallic(self):
        r = CyclesRenderer(Scene(), 8, 6, 1)
        mat = Material(metallic=1.0, roughness=0.3)
        N = np.array([0.0, 1.0, 0.0])
        wo = _normalize(np.array([-0.2, 1.0, 0.0]))
        wi = _normalize(np.array([0.2, 1.0, 0.0]))
        pdf = r._bsdf_pdf(wo, wi, N, mat)
        assert pdf > 0

    def test_sample_ggx(self):
        r = CyclesRenderer(Scene(), 8, 6, 1)
        h = r._sample_ggx(0.3, 0.7, 0.2)
        assert h.shape == (3,)

    def test_tangent_frame(self):
        r = CyclesRenderer(Scene(), 8, 6, 1)
        t, b = r._tangent_frame(np.array([0.0, 1.0, 0.0]))
        assert t.shape == (3,)
        t2, b2 = r._tangent_frame(np.array([0.0, 1.0, 0.0]))
        assert np.allclose(t, t2)
        up = np.array([0.0, 0.999, 0.0])
        t3, b3 = r._tangent_frame(up)
        assert t3.shape == (3,)

    def test_triangle_barycentric_degenerate(self):
        scene = Scene()
        r = CyclesRenderer(scene, 8, 6, 1)
        v0 = np.array([0.0, 0.0, 0.0])
        bary = r._triangle_barycentric(np.zeros(3), np.zeros(3), 0.0, v0, v0, v0)
        np.testing.assert_allclose(bary, [1 / 3, 1 / 3, 1 / 3])

    def test_rays_terminate_at_max_depth(self, monkeypatch):
        class _Rng:
            def uniform(self, low, high, size):
                return np.zeros(size)

            def random(self):
                return 0.0

        scene = _scene_with_mesh()
        r = CyclesRenderer(scene, 16, 12, 1)
        r._rng = _Rng()

        def _always_hit(orig, dire):
            n = len(orig)
            return (np.full(n, 1.0, dtype=np.float32),
                    np.zeros(n, dtype=np.int32),
                    orig + 0.5 * dire,
                    np.tile(np.array([0.0, 1.0, 0.0]), (n, 1)),
                    np.zeros(n, dtype=np.int32),
                    np.zeros(n, dtype=np.int32))

        monkeypatch.setattr(r, "_intersect_scene", _always_hit)
        img = r.render()
        assert img.shape == (12, 16, 3)
