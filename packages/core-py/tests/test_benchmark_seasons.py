"""
Tests for the Stage 14 seasonal boundary (``domains/shell/evolution.py``).

Covers the year envelope physics (world.light modulation, season labels,
season index mapping, year counting), conservation under the seasonal
boundary, RNG isolation of the year envelope, and the full ``benchmark_seasons``
verdict. Seasons are a pure function of ``tick`` — no RNG — so the locked
selection proofs must stay bit-identical with the year on or off.
"""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.evolution import (
    Genome,
    _conservation_sweep,
    benchmark_seasons,
)
from domains.shell.simulation import SimScene, WorldParams


def _seasonal_params(**kw) -> WorldParams:
    base = dict(
        grid_size=(16, 8, 16),
        generate_world=True,
        world_seed=7,
        learning_enabled=True,
        solar_enabled=True,
        solar_day_ticks=24,
        solar_max_intensity=1.0,
        solar_deposit_rate=0.1,
        solar_season_ticks=96,   # 4 full days per year
        solar_seasonality=1.0,
    )
    base.update(kw)
    return WorldParams(**base)


class TestSeasonEnvelope:
    """The year envelope is a pure, deterministic function of tick."""

    NOON = 6  # day = 24 ticks, diurnal peak at day//4

    def _noon(self, params: WorldParams, tick: int) -> float:
        scene = SimScene(params=params)
        scene._tick = tick
        scene.apply_solar()
        return float(scene.world.light)

    def test_summer_noon_brighter_than_winter_noon(self):
        p = _seasonal_params()
        assert self._noon(p, self.NOON) > self._noon(p, 96 // 2 + self.NOON)

    def test_envelope_ceiling_never_exceeds_one(self):
        p = _seasonal_params(solar_seasonality=0.6)
        for tick in range(0, 96, 4):
            assert self._noon(p, tick) <= 1.0 + 1e-9

    def test_no_seasons_reproduces_stage13(self):
        p = _seasonal_params(solar_season_ticks=0, solar_seasonality=1.0)
        assert self._noon(p, self.NOON) == pytest.approx(1.0)
        assert self._noon(p, 96 // 2 + self.NOON) == pytest.approx(1.0)

    def test_season_index_maps_year_quadrants(self):
        p = _seasonal_params()
        scene = SimScene(params=p)
        expected = {0: 0, 23: 0, 24: 1, 47: 1, 48: 2, 71: 2, 72: 3, 95: 3}
        for tick, want in expected.items():
            scene._tick = tick
            scene.apply_solar()
            assert scene.solar_season_index == want
            assert scene.solar_season_factor == pytest.approx(
                (1 - p.solar_seasonality) + p.solar_seasonality * (
                    0.5 + 0.5 * np.cos(2 * np.pi * tick / p.solar_season_ticks)))

    def test_year_counter_increments(self):
        p = _seasonal_params()
        scene = SimScene(params=p)
        scene._tick = 191
        scene.apply_solar()
        assert scene.solar_year == 1  # 191 // 96


class TestSeasonalConservation:
    def test_monotonic_under_seasonal_boundary(self):
        params = _seasonal_params()
        genomes = [Genome.random(params, np.random.default_rng(7), group_id=0)
                   for _ in range(8)]
        result = _conservation_sweep(params, genomes, ticks=96)
        assert result["monotonic"] is True
        assert result["violations"] == []
        assert result["boundary_deposit_total"] > 0.0

    def test_deterministic_sweep(self):
        params = _seasonal_params()
        genomes = [Genome.random(params, np.random.default_rng(7), group_id=0)
                   for _ in range(8)]
        a = _conservation_sweep(params, genomes, ticks=48)
        b = _conservation_sweep(params, genomes, ticks=48)
        assert a["end_total"] == b["end_total"]


class TestRNGIsolation:
    def test_brains_bit_identical_year_on_off(self):
        day = 24
        off = WorldParams(grid_size=(16, 8, 16), generate_world=True,
                          world_seed=7, solar_enabled=True,
                          solar_day_ticks=day, solar_max_intensity=1.0,
                          solar_season_ticks=0, solar_seasonality=1.0)
        on = WorldParams(grid_size=(16, 8, 16), generate_world=True,
                         world_seed=7, solar_enabled=True,
                         solar_day_ticks=day, solar_max_intensity=1.0,
                         solar_season_ticks=4 * day, solar_seasonality=1.0)
        rng_off = np.random.default_rng(7)
        rng_on = np.random.default_rng(7)
        g_off = Genome.random(off, rng_off, group_id=0)
        g_on = Genome.random(on, rng_on, group_id=0)
        for name in ("cells", "body", "entity", "move"):
            for suf in ("W", "b"):
                assert np.allclose(g_off.tensors[f"{name}.{suf}"],
                                   g_on.tensors[f"{name}.{suf}"])
                assert g_off.tensors[f"{name}.{suf}"].dtype \
                    == g_on.tensors[f"{name}.{suf}"].dtype


class TestBenchmarkSeasons:
    def test_benchmark_verdicts(self):
        result = benchmark_seasons(
            population_size=6, generations=3, ticks_per_generation=24,
            organic_pools=2, solar_deposit_rate=0.1, seasonality=1.0,
            seasons_per_year=4, seed=7,
        )
        assert result["seasonal_conservation_exact"] is True
        assert result["closed_monotonic"] is True
        assert result["brains_identical"] is True
        assert result["summer_noon"] > result["winter_noon"]
        assert result["deposited"] > 0.0
        assert result["sunshine"] > 0.0
        assert result["seasons_emerged"] is True

    def test_benchmark_shape(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.6, seasons_per_year=4, seed=3,
        )
        assert result["control_last_avg"] >= 0.0
        assert result["seasonal_last_avg"] >= 0.0
        assert result["seasonal_start_total"] > 0.0
        assert result["seasonal_boundary_deposit"] > 0.0
        assert len(result["seasonal"]["history"]) == 2
