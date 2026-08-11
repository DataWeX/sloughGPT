"""
Tests for the live realm view (``domains/shell/realm_view.py``).

Covers scene construction, frame rendering (materials, babies, skyline sun),
the live stepping loop, and determinism of the seeded viewer.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pytest

from domains.shell.realm_view import live_view, make_live_scene, render_frame
from domains.shell.simulation import (
    MATERIAL_STONE,
    MATERIAL_ORGANIC,
    SimBaby,
    SimScene,
    Simulation,
    WorldParams,
)
from domains.shell.evolution import Genome


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestMakeLiveScene:
    def test_builds_solar_lit_generated_world(self):
        scene = make_live_scene(grid=(16, 8, 16), population=6, seed=3)
        assert isinstance(scene, SimScene)
        assert scene.params.solar_enabled is True
        assert scene.params.cells_input_dim == 6
        assert scene.world.nx == 16 and scene.world.ny == 8 and scene.world.nz == 16

    def test_spawns_all_babies_alive_on_surface(self):
        scene = make_live_scene(grid=(16, 8, 16), population=8, seed=7)
        alive = scene.alive_babies
        assert len(alive) == 8
        for b in alive:
            x, y, z = (int(round(float(c))) for c in b.position)
            assert 0 <= x < 16 and 0 <= z < 16
            assert y >= 1  # standing just above the topmost ground cell

    def test_deterministic_scene(self):
        a = make_live_scene(grid=(16, 8, 16), population=8, seed=7)
        b = make_live_scene(grid=(16, 8, 16), population=8, seed=7)
        assert np.allclose(a.world.energy, b.world.energy)
        for ba, bb in zip(a.babies, b.babies):
            assert np.allclose(ba.position, bb.position)
            assert np.allclose(ba.energy, bb.energy)


class TestRenderFrame:
    def _noon_scene(self):
        scene = make_live_scene(grid=(16, 8, 16), population=8, day_ticks=16, seed=7)
        sim = Simulation(scene, max_ticks=8)
        for _ in range(4):  # phase 4 of 16 -> sin(pi/2) = full noon
            sim.step()
        return scene

    def test_sun_visible_at_noon(self):
        scene = self._noon_scene()
        assert scene.world.light > 0.9
        frame = _strip_ansi("\n".join(render_frame(scene, tick=4)))
        assert "☀" in frame

    def test_sun_absent_before_first_tick(self):
        scene = make_live_scene(grid=(16, 8, 16), population=8, seed=7)
        assert scene.world.light == 0.0
        frame = _strip_ansi("\n".join(render_frame(scene, tick=0)))
        assert "☀" not in frame

    def test_material_glyphs_rendered(self):
        scene = self._noon_scene()
        frame = _strip_ansi("\n".join(render_frame(scene, tick=4)))
        assert "#" in frame or "M" in frame  # generated world has ground/metal
        assert "stone" in frame and "baby" in frame

    def test_all_alive_babies_overlaid(self):
        scene = make_live_scene(grid=(16, 8, 16), population=3, seed=11)
        sim = Simulation(scene, max_ticks=2)
        sim.step()
        frame = _strip_ansi("\n".join(render_frame(scene, tick=1)))
        assert frame.count("B") >= 3

    def test_energy_heat_background_written(self):
        scene = self._noon_scene()
        frame = render_frame(scene, tick=4)
        assert any("48;5;" in line for line in frame)  # heat backgrounds present


class TestLiveView:
    def test_runs_and_returns_stats(self):
        scene = make_live_scene(grid=(16, 8, 16), population=8, day_ticks=16, seed=7)
        out = io.StringIO()
        stats = live_view(scene, ticks=24, fps=0, out=out)
        assert set(stats) == {"ticks", "energy_total", "solar_in", "alive",
                              "population", "births", "deaths"}
        assert stats["ticks"] == 24
        assert stats["solar_in"] > 0.0  # the sky deposited energy
        assert 0 <= stats["alive"] <= stats["population"]

    def test_emits_one_cleared_frame_per_tick(self):
        scene = make_live_scene(grid=(16, 8, 16), population=4, day_ticks=16, seed=7)
        out = io.StringIO()
        live_view(scene, ticks=6, fps=0, out=out)
        frames = out.getvalue().split("\x1b[2J\x1b[H")
        assert len(frames) - 1 == 6

    def test_deterministic_run(self):
        a = make_live_scene(grid=(16, 8, 16), population=6, day_ticks=16, seed=5)
        b = make_live_scene(grid=(16, 8, 16), population=6, day_ticks=16, seed=5)
        sa = live_view(a, ticks=20, fps=0, out=io.StringIO())
        sb = live_view(b, ticks=20, fps=0, out=io.StringIO())
        assert sa["energy_total"] == sb["energy_total"]
        assert sa["solar_in"] == sb["solar_in"]
        assert sa["alive"] == sb["alive"]

    def test_births_and_deaths_tracked(self):
        params = WorldParams(grid_size=(16, 8, 16), generate_world=True,
                             world_seed=7, lifecycle_enabled=True,
                             solar_enabled=False, max_entities=32,
                             start_energy=40.0, learning_enabled=True)
        scene = SimScene(params=params)
        for i in range(8):
            b = SimBaby(initial_energy=params.start_energy, params=params)
            Genome.random(params, np.random.default_rng(1)).apply_to(b)
            scene.add_baby(b)
        out = io.StringIO()
        stats = live_view(scene, ticks=60, fps=0, out=out)
        assert stats["births"] >= 0
        assert stats["deaths"] >= 0
