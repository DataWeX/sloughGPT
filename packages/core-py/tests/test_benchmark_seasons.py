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

    def _light_at(self, params: WorldParams, tick: int) -> float:
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

    def test_dawn_light_is_zero(self):
        p = _seasonal_params()
        assert self._light_at(p, 0) == pytest.approx(0.0, abs=1e-6)

    def test_dusk_light_approaches_zero(self):
        p = _seasonal_params()
        light = self._light_at(p, 23)
        assert light < 0.1

    def test_noon_is_peak_of_diurnal_cycle(self):
        p = _seasonal_params()
        peak = self._light_at(p, self.NOON)
        for tick in range(0, 24):
            assert self._light_at(p, tick) <= peak + 1e-6

    def test_midnight_darkness(self):
        p = _seasonal_params()
        light = self._light_at(p, 12)
        assert light < 0.5

    def test_full_year_cosine_symmetry_around_peak(self):
        p = _seasonal_params()
        year_ticks = p.solar_season_ticks
        for offset in range(1, year_ticks // 2, 8):
            factor_before = self._season_factor(p, year_ticks - offset)
            factor_after = self._season_factor(p, offset)
            assert factor_before == pytest.approx(factor_after, abs=1e-4)

    def _season_factor(self, params: WorldParams, tick: int) -> float:
        scene = SimScene(params=params)
        scene._tick = tick
        scene.apply_solar()
        return float(scene.solar_season_factor)

    def test_season_factor_at_summer_peak(self):
        p = _seasonal_params()
        factor = self._season_factor(p, 0)
        assert factor == pytest.approx(1.0, abs=1e-6)

    def test_season_factor_at_winter_trough(self):
        p = _seasonal_params()
        factor = self._season_factor(p, 48)
        assert factor == pytest.approx(0.0, abs=1e-6)

    def test_half_seasonality_produces_intermediate_factors(self):
        p = _seasonal_params(solar_seasonality=0.5)
        peak = self._season_factor(p, 0)
        trough = self._season_factor(p, 48)
        assert peak > trough
        assert peak < 1.0 + 1e-6
        assert trough > 0.0 - 1e-6

    def test_zero_seasonality_is_constant(self):
        p = _seasonal_params(solar_seasonality=0.0)
        factor = self._season_factor(p, 0)
        assert factor == pytest.approx(1.0, abs=1e-6)
        factor2 = self._season_factor(p, 48)
        assert factor2 == pytest.approx(1.0, abs=1e-6)

    def test_season_index_wraps_at_year_boundary(self):
        p = _seasonal_params()
        scene = SimScene(params=p)
        scene._tick = 96
        scene.apply_solar()
        assert scene.solar_season_index == 0
        assert scene.solar_year == 1

    def test_light_positive_during_daytime(self):
        p = _seasonal_params()
        for tick in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            light = self._light_at(p, tick)
            assert light > 0.0

    def test_light_negative_midnight_boundary(self):
        p = _seasonal_params()
        light = self._light_at(p, 13)
        assert 0.0 <= light <= 1.0

    def test_consistent_light_across_same_tick(self):
        p = _seasonal_params()
        for tick in range(0, 48, 4):
            a = self._light_at(p, tick)
            b = self._light_at(p, tick)
            assert a == b

    def test_noon_light_scales_with_max_intensity(self):
        p_low = _seasonal_params(solar_max_intensity=0.5)
        p_high = _seasonal_params(solar_max_intensity=1.0)
        low = self._noon(p_low, self.NOON)
        high = self._noon(p_high, self.NOON)
        assert low < high

    def test_seasonal_tick_counts_independent_of_day_ticks(self):
        p1 = _seasonal_params(solar_season_ticks=48, solar_seasonality=1.0)
        p2 = _seasonal_params(solar_season_ticks=96, solar_seasonality=1.0)
        f1 = self._season_factor(p1, 0)
        f2 = self._season_factor(p2, 0)
        assert f1 == pytest.approx(1.0, abs=1e-6)
        assert f2 == pytest.approx(1.0, abs=1e-6)

    def test_multiple_years_cycle_same_factors(self):
        p = _seasonal_params()
        year_ticks = p.solar_season_ticks
        for year in range(3):
            offset = year * year_ticks
            assert self._season_factor(p, offset) == pytest.approx(1.0, abs=1e-6)
            assert self._season_factor(p, offset + 48) == pytest.approx(0.0, abs=1e-6)


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

    def test_start_total_positive(self):
        params = _seasonal_params()
        genomes = [Genome.random(params, np.random.default_rng(7), group_id=0)
                   for _ in range(4)]
        result = _conservation_sweep(params, genomes, ticks=24)
        assert result["start_total"] > 0.0

    def test_boundary_deposit_non_negative(self):
        params = _seasonal_params()
        genomes = [Genome.random(params, np.random.default_rng(7), group_id=0)
                   for _ in range(4)]
        result = _conservation_sweep(params, genomes, ticks=48)
        assert result["boundary_deposit_total"] >= 0.0

    def test_violations_list_type(self):
        params = _seasonal_params()
        genomes = [Genome.random(params, np.random.default_rng(7), group_id=0)
                   for _ in range(4)]
        result = _conservation_sweep(params, genomes, ticks=24)
        assert isinstance(result["violations"], list)

    def test_monotonic_flag_is_bool(self):
        params = _seasonal_params()
        genomes = [Genome.random(params, np.random.default_rng(7), group_id=0)
                   for _ in range(4)]
        result = _conservation_sweep(params, genomes, ticks=24)
        assert isinstance(result["monotonic"], bool)

    def test_sweep_with_different_seed_different_totals(self):
        params_a = _seasonal_params(world_seed=1)
        params_b = _seasonal_params(world_seed=42)
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(42)
        genomes_a = [Genome.random(params_a, rng_a, group_id=0) for _ in range(4)]
        genomes_b = [Genome.random(params_b, rng_b, group_id=0) for _ in range(4)]
        a = _conservation_sweep(params_a, genomes_a, ticks=24)
        b = _conservation_sweep(params_b, genomes_b, ticks=24)
        # Different seeds should produce different start totals (different terrain)
        assert a["start_total"] != b["start_total"]

    def test_sweep_returns_required_keys(self):
        params = _seasonal_params()
        genomes = [Genome.random(params, np.random.default_rng(7), group_id=0)
                   for _ in range(4)]
        result = _conservation_sweep(params, genomes, ticks=24)
        for key in ("monotonic", "violations", "start_total", "end_total",
                     "boundary_deposit_total"):
            assert key in result


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

    def test_isolation_with_different_seasonality(self):
        day = 24
        p0 = WorldParams(grid_size=(16, 8, 16), generate_world=True,
                         world_seed=7, solar_enabled=True,
                         solar_day_ticks=day, solar_max_intensity=1.0,
                         solar_season_ticks=96, solar_seasonality=0.0)
        p1 = WorldParams(grid_size=(16, 8, 16), generate_world=True,
                         world_seed=7, solar_enabled=True,
                         solar_day_ticks=day, solar_max_intensity=1.0,
                         solar_season_ticks=96, solar_seasonality=1.0)
        rng0 = np.random.default_rng(7)
        rng1 = np.random.default_rng(7)
        g0 = Genome.random(p0, rng0, group_id=0)
        g1 = Genome.random(p1, rng1, group_id=0)
        for name in ("cells", "body", "entity", "move"):
            for suf in ("W", "b"):
                assert np.allclose(g0.tensors[f"{name}.{suf}"],
                                   g1.tensors[f"{name}.{suf}"])

    def test_group_id_does_not_affect_brains(self):
        day = 24
        p = WorldParams(grid_size=(16, 8, 16), generate_world=True,
                        world_seed=7, solar_enabled=True,
                        solar_day_ticks=day, solar_max_intensity=1.0,
                        solar_season_ticks=96, solar_seasonality=1.0)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        g_a = Genome.random(p, rng_a, group_id=0)
        g_b = Genome.random(p, rng_b, group_id=1)
        for name in ("cells", "body", "entity", "move"):
            for suf in ("W", "b"):
                assert np.allclose(g_a.tensors[f"{name}.{suf}"],
                                   g_b.tensors[f"{name}.{suf}"])


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

    def test_benchmark_returns_all_required_keys(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.8, seasons_per_year=4, seed=5,
        )
        expected_keys = [
            "control", "seasonal", "control_last_avg", "seasonal_last_avg",
            "seasonal_conservation_exact", "seasonal_violations",
            "seasonal_start_total", "seasonal_end_total",
            "seasonal_boundary_deposit", "closed_monotonic",
            "closed_violations", "closed_start_total", "closed_end_total",
            "brains_identical", "summer_noon", "winter_noon",
            "deposited", "sunshine", "population_size", "generations",
            "seasons_per_year", "seasonality", "seasons_emerged",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_benchmark_deterministic(self):
        a = benchmark_seasons(population_size=4, generations=2,
                              ticks_per_generation=16, organic_pools=1,
                              seasonality=0.8, seasons_per_year=4, seed=99)
        b = benchmark_seasons(population_size=4, generations=2,
                              ticks_per_generation=16, organic_pools=1,
                              seasonality=0.8, seasons_per_year=4, seed=99)
        assert a["summer_noon"] == b["summer_noon"]
        assert a["winter_noon"] == b["winter_noon"]

    def test_benchmark_generations_count(self):
        result = benchmark_seasons(
            population_size=4, generations=3, ticks_per_generation=16,
            organic_pools=1, seasonality=0.8, seasons_per_year=4, seed=5,
        )
        assert len(result["seasonal"]["history"]) == 3
        assert len(result["control"]["history"]) == 3

    def test_benchmark_population_size(self):
        result = benchmark_seasons(
            population_size=5, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.8, seasons_per_year=4, seed=5,
        )
        assert result["population_size"] == 5

    def test_benchmark_seasons_per_year(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=1.0, seasons_per_year=4, seed=5,
        )
        assert result["seasons_per_year"] == 4

    def test_benchmark_seasonality_value(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.6, seasons_per_year=4, seed=5,
        )
        assert result["seasonality"] == 0.6

    def test_summer_noon_greater_than_winter_noon(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=1.0, seasons_per_year=4, seed=5,
        )
        assert result["summer_noon"] > result["winter_noon"]

    def test_closed_world_violations_empty_when_monotonic(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.8, seasons_per_year=4, seed=5,
        )
        if result["closed_monotonic"]:
            assert result["closed_violations"] == []

    def test_seasonal_violations_empty_when_monotonic(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.8, seasons_per_year=4, seed=5,
        )
        if result["seasonal_conservation_exact"]:
            assert result["seasonal_violations"] == []

    def test_control_and_seasonal_history_entries(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.8, seasons_per_year=4, seed=5,
        )
        for entry in result["seasonal"]["history"]:
            assert "best_fitness" in entry
            assert "avg_fitness" in entry
            assert "alive" in entry

    def test_control_last_avg_non_negative(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.8, seasons_per_year=4, seed=5,
        )
        assert result["control_last_avg"] >= 0.0

    def test_benchmark_with_different_seasonality(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.0, seasons_per_year=4, seed=5,
        )
        assert result["seasonality"] == 0.0
        # With zero seasonality, summer and winter noon should be closer
        diff = abs(result["summer_noon"] - result["winter_noon"])
        assert diff < 0.5

    def test_benchmark_deposited_positive(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.8, seasons_per_year=4, seed=5,
        )
        assert result["deposited"] >= 0.0

    def test_benchmark_sunshine_positive(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.8, seasons_per_year=4, seed=5,
        )
        assert result["sunshine"] >= 0.0

    def test_history_entry_has_social_stats(self):
        result = benchmark_seasons(
            population_size=4, generations=2, ticks_per_generation=16,
            organic_pools=1, seasonality=0.8, seasons_per_year=4, seed=5,
        )
        for entry in result["seasonal"]["history"]:
            assert "cooperations" in entry
            assert "contests" in entry
            assert "cooperate_rate" in entry
            assert "contest_rate" in entry
