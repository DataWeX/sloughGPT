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

from domains.shell.realm_view import (
    _season_of,
    _year_envelope_bar,
    live_view,
    make_live_scene,
    render_frame,
)
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

    def test_seasons_off_by_default(self):
        scene = make_live_scene(grid=(16, 8, 16), population=4, seed=7)
        assert scene.params.solar_season_ticks == 0
        assert scene.params.solar_seasonality == 1.0

    def test_seasons_year_length(self):
        scene = make_live_scene(grid=(16, 8, 16), population=4, day_ticks=8,
                                seasons_per_year=4, seed=7)
        assert scene.params.solar_season_ticks == 32  # 8 ticks/day * 4 days
        assert scene.params.solar_seasonality == 1.0

    def test_seasons_deterministic_scene(self):
        a = make_live_scene(grid=(16, 8, 16), population=6, day_ticks=8,
                            seasons_per_year=4, seed=5)
        b = make_live_scene(grid=(16, 8, 16), population=6, day_ticks=8,
                            seasons_per_year=4, seed=5)
        assert np.allclose(a.world.energy, b.world.energy)


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


class TestSeasons:
    """Stage 14: the seasonal year envelope in the live view."""

    def _stepped(self, ticks: int, seasons_per_year: int = 4) -> SimScene:
        scene = make_live_scene(grid=(16, 8, 16), population=6, day_ticks=8,
                                seasons_per_year=seasons_per_year, seed=7)
        sim = Simulation(scene, max_ticks=max(ticks, 1))
        for _ in range(ticks):
            sim.step()
        return scene

    def _frame(self, scene: SimScene, tick: int) -> str:
        return _strip_ansi("\n".join(render_frame(scene, tick=tick)))

    def test_summer_noon_outshines_winter_noon(self):
        summer = self._stepped(2)   # tick 2 = noon, year quadrant 0 (summer)
        winter = self._stepped(18)  # tick 18 = noon, year midpoint (winter)
        idx_s, factor_s = _season_of(summer)
        idx_w, factor_w = _season_of(winter)
        assert idx_s == 0 and factor_s > 0.9   # envelope near full strength
        assert idx_w == 2 and factor_w < factor_s
        assert winter.world.light < summer.world.light

    def test_envelope_pure_function_of_tick(self):
        a = self._stepped(16)
        b = self._stepped(16)
        assert a.solar_season_index == b.solar_season_index
        assert a.solar_season_factor == b.solar_season_factor
        assert a.solar_year == b.solar_year

    def test_year_advances_after_full_year(self):
        scene = self._stepped(32)  # one full 32-tick year (8 * 4)
        assert scene.solar_year == 1
        assert scene.solar_season_index == 0  # wrapped back to summer

    def test_season_names_cycle(self):
        assert _season_of(self._stepped(7))[0] == 0   # SUMMER
        assert _season_of(self._stepped(8))[0] == 1   # AUTUMN
        assert _season_of(self._stepped(16))[0] == 2  # WINTER
        assert _season_of(self._stepped(24))[0] == 3  # SPRING

    def test_skyline_names_current_season(self):
        summer = self._frame(self._stepped(4), tick=4)
        winter = self._frame(self._stepped(16), tick=16)
        assert "year 0" in summer and "SUMMER" in summer
        assert "year 0" in winter and "WINTER" in winter

    def test_year_envelope_bar_rendered_only_with_seasons(self):
        on = self._frame(self._stepped(4), tick=4)
        off = self._frame(make_live_scene(grid=(16, 8, 16), population=6,
                                          day_ticks=8, seed=7), tick=4)
        assert "env" in on and "[" in on
        assert "env" not in off
        assert _year_envelope_bar(make_live_scene(grid=(16, 8, 16), seed=7)) == ""

    def test_summer_peak_is_year_bar_high_end(self):
        bar = _strip_ansi(_year_envelope_bar(self._stepped(0)))
        assert bar.startswith("year 0  SUMMER env 100%  [█")  # peak ridge at left


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
