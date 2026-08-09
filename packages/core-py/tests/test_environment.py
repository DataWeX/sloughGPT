"""
Tests for the world environment: material behaviors (combustion, ember,
living growth, water, metal, metabolism), temperature relaxation, and
deterministic terrain generation.

Every behavior is animated by the existing fields (energy, temperature,
signal) and parameters live in WorldParams so tests can tune them.
"""

import numpy as np
import pytest

from domains.shell.simulation import (
    WorldGrid, WorldParams,
    cell_update_combustion, cell_update_metabolism, cell_update_ember,
    cell_update_living, cell_update_water, cell_update_conduction,
    cell_update_temperature, cell_update_materials, cell_update_default,
    generate_world, SimScene, Simulation,
    MATERIAL_AIR, MATERIAL_STONE, MATERIAL_WATER, MATERIAL_ORGANIC,
    MATERIAL_METAL, MATERIAL_EMBER, MATERIAL_LIVING,
)


# ── Combustion & temperature ─────────────────────────────────────────────────

class TestCombustion:
    def test_organic_ignites_above_ignition_temp(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_ORGANIC, energy=50.0, temperature=200.0)
        params = WorldParams(ignition_temp=100.0, burn_temp=150.0)
        cell_update_combustion(g, params)
        assert g.material[i] == MATERIAL_EMBER
        assert g.temperature[i] == 150.0
        assert g.energy[i] == 50.0  # fuel preserved

    def test_cool_organic_stays_organic(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_ORGANIC, energy=50.0, temperature=20.0)
        cell_update_combustion(g, WorldParams(ignition_temp=100.0))
        assert g.material[i] == MATERIAL_ORGANIC

    def test_full_default_pipeline_ignites(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_ORGANIC, energy=80.0, temperature=250.0)
        params = WorldParams(
            diffusion_rate=0.0, ambient_cooling=0.0, energy_loss=0.0,
            ignition_temp=100.0, burn_temp=150.0,
        )
        cell_update_default(g, params)
        assert g.material[i] == MATERIAL_EMBER

    def test_temperature_relaxes_to_ambient(self):
        g = WorldGrid((4, 4, 4))
        np.copyto(g.temperature, np.full(g.total, 100.0, dtype=np.float32))
        params = WorldParams(ambient_temp=20.0, ambient_cooling=0.5)
        cell_update_temperature(g, params)
        assert np.allclose(g.temperature, 60.0)


# ── Metabolism & ember ────────────────────────────────────────────────────────

class TestMetabolism:
    def test_organic_rots(self):
        g = WorldGrid((4, 4, 4))
        g.place_material(1, 1, 1, MATERIAL_ORGANIC, energy=100.0)
        cell_update_metabolism(g, WorldParams(organic_metabolism=0.1))
        assert g.energy[g.idx(1, 1, 1)] == pytest.approx(90.0)

    def test_air_unaffected(self):
        g = WorldGrid((4, 4, 4))
        g.place_material(1, 1, 1, MATERIAL_STONE, energy=100.0)
        cell_update_metabolism(g, WorldParams(organic_metabolism=0.1))
        assert g.energy[g.idx(1, 1, 1)] == 100.0


class TestEmber:
    def test_radiates_energy_and_heat(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_EMBER, energy=100.0, temperature=150.0)
        params = WorldParams(
            ember_heat_rate=0.5, ember_energy_fraction=0.5, heat_to_temp=1.0,
        )
        before = g.total_energy
        cell_update_ember(g, params)
        assert g.energy[i] == pytest.approx(50.0)  # burned half its fuel
        assert g.total_energy < before  # the heat portion left the energy pool
        assert abs(g.total_energy - (before - 25.0)) < 1e-4  # energy part conserved
        neighbor = g.get_cell(0, 1, 1)
        assert neighbor.temperature > 20.0  # heat reached a neighbor
        assert g.get_cell(1, 1, 1).temperature == 150.0  # ember sustains burn_temp

    def test_exhausted_ember_cools_to_stone(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_EMBER, energy=0.5, temperature=150.0)
        params = WorldParams(ember_heat_rate=1.0)
        cell_update_ember(g, params)
        assert g.material[i] == MATERIAL_STONE

    def test_burnout_without_signal(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_EMBER, energy=0.0, temperature=150.0)
        cell_update_ember(g, WorldParams())
        assert g.material[i] == MATERIAL_STONE

    def test_adjacent_embers_each_burn_from_their_own_fuel(self):
        g = WorldGrid((4, 4, 4))
        ia = g.idx(1, 1, 1)
        ib = g.idx(2, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_EMBER, energy=100.0)
        g.place_material(2, 1, 1, MATERIAL_EMBER, energy=100.0)
        params = WorldParams(
            ember_heat_rate=0.5, ember_energy_fraction=0.5, heat_to_temp=1.0,
        )
        before = g.total_energy
        cell_update_ember(g, params)
        # Each ember burns half its own 100 fuel (50) exactly once, then
        # receives a 25/6 radiation share from its adjacent ember — the
        # received energy is NOT re-burned within the same tick.
        share = 25.0 / 6.0
        assert g.energy[ia] == pytest.approx(50.0 + share)
        assert g.energy[ib] == pytest.approx(50.0 + share)
        # The heat half leaves the pool (2 embers, 2 x 25 lost).
        assert abs(g.total_energy - (before - 2 * 25.0)) < 1e-3


# ── Living growth ─────────────────────────────────────────────────────────────

class TestLiving:
    def test_grows_into_adjacent_air(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_LIVING, energy=100.0)
        params = WorldParams(
            living_growth_rate=0.5, living_growth_cost=10.0,
            growth_transfer_fraction=0.8,
        )
        cell_update_living(g, params)
        spawned = np.flatnonzero(g.material == MATERIAL_ORGANIC)
        assert len(spawned) == 1
        assert g.energy[spawned[0]] == pytest.approx(8.0)
        assert g.energy[i] == pytest.approx(90.0)

    def test_too_poor_to_grow_rests(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_LIVING, energy=3.0)
        params = WorldParams(
            living_growth_rate=0.5, living_growth_cost=10.0,
        )
        cell_update_living(g, params)
        assert np.flatnonzero(g.material == MATERIAL_ORGANIC).size == 0
        assert g.energy[i] == 3.0

    def test_growth_is_deterministic(self):
        g1 = WorldGrid((4, 4, 4))
        g2 = WorldGrid((4, 4, 4))
        g1.place_material(1, 1, 1, MATERIAL_LIVING, energy=100.0)
        g2.place_material(1, 1, 1, MATERIAL_LIVING, energy=100.0)
        params = WorldParams(living_growth_rate=0.5, living_growth_cost=10.0)
        cell_update_living(g1, params)
        cell_update_living(g2, params)
        assert np.array_equal(g1.material, g2.material)

    def test_grows_into_later_neighbor_when_first_is_blocked(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_LIVING, energy=100.0)
        g.place_material(0, 1, 1, MATERIAL_STONE)  # block first scan neighbor
        params = WorldParams(
            living_growth_rate=0.5, living_growth_cost=10.0,
            growth_transfer_fraction=0.8,
        )
        cell_update_living(g, params)
        assert g.material[g.idx(2, 1, 1)] == MATERIAL_ORGANIC
        assert g.energy[g.idx(2, 1, 1)] == pytest.approx(8.0)
        assert g.energy[i] == pytest.approx(90.0)

    def test_shared_target_claimed_once(self):
        g = WorldGrid((4, 4, 4))
        a = g.idx(1, 1, 1)
        b = g.idx(3, 1, 1)
        shared = g.idx(0, 1, 1)  # A's first neighbor AND B's wrapped (1,0,0) neighbor
        g.place_material(1, 1, 1, MATERIAL_LIVING, energy=100.0)
        g.place_material(3, 1, 1, MATERIAL_LIVING, energy=100.0)
        g.place_material(2, 1, 1, MATERIAL_STONE)  # block B's first scan neighbor
        params = WorldParams(
            living_growth_rate=0.5, living_growth_cost=10.0,
            growth_transfer_fraction=0.8,
        )
        cell_update_living(g, params)
        # A claims (0,1,1) first; B cannot re-use it (wrapped (1,0,0) neighbor),
        # so the organic cell receives exactly one growth transfer.
        assert g.material[shared] == MATERIAL_ORGANIC
        assert g.energy[shared] == pytest.approx(8.0)
        assert g.energy[a] == pytest.approx(90.0)
        assert g.energy[b] == pytest.approx(90.0)
        assert np.count_nonzero(g.material == MATERIAL_ORGANIC) == 2


# ── Water ─────────────────────────────────────────────────────────────────────

class TestWater:
    def test_damps_signal(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_WATER)
        g.signal[i] = 10.0
        cell_update_water(g, WorldParams(water_signal_dampen=0.5))
        assert g.signal[i] == pytest.approx(5.0)

    def test_cools_toward_ambient(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.place_material(1, 1, 1, MATERIAL_WATER, temperature=100.0)
        params = WorldParams(ambient_temp=20.0, water_cool_rate=0.5)
        cell_update_water(g, params)
        assert g.temperature[i] == pytest.approx(60.0)


# ── Metal conduction ──────────────────────────────────────────────────────────

class TestMetal:
    def test_metal_spreads_energy_faster_than_baseline(self):
        base = WorldGrid((4, 4, 4))
        metal = WorldGrid((4, 4, 4))
        for g in (base, metal):
            g.place_material(1, 1, 1, MATERIAL_METAL, energy=100.0)
            g.place_material(2, 1, 1, MATERIAL_METAL, energy=0.0)
        params = WorldParams(diffusion_rate=0.1, metal_conduction_boost=0.0)
        cell_update_diffusion_pair = _diffuse_only
        cell_update_diffusion_pair(base, params)
        params_boost = WorldParams(diffusion_rate=0.1, metal_conduction_boost=4.0)
        cell_update_conduction(metal, params_boost)
        assert metal.energy[metal.idx(1, 1, 1)] < base.energy[base.idx(1, 1, 1)]

    def test_conduction_conserves_energy(self):
        g = WorldGrid((4, 4, 4))
        g.place_material(1, 1, 1, MATERIAL_METAL, energy=100.0)
        g.place_material(2, 1, 1, MATERIAL_METAL, energy=0.0)
        before = g.total_energy
        cell_update_conduction(g, WorldParams(diffusion_rate=0.1, metal_conduction_boost=2.0))
        assert abs(g.total_energy - before) < 1e-4


def _diffuse_only(g, params):
    from domains.shell.simulation import cell_update_diffusion
    cell_update_diffusion(g, params)


# ── Terrain generation ────────────────────────────────────────────────────────

class TestGenerateWorld:
    def test_creates_floor_food_water_and_ember(self):
        g = WorldGrid((16, 8, 16))
        params = WorldParams(world_seed=42)
        generate_world(g, params)
        assert (g.material == MATERIAL_STONE).any()
        assert (g.material == MATERIAL_ORGANIC).any()
        assert (g.material == MATERIAL_WATER).any()
        assert (g.material == MATERIAL_EMBER).any()

    def test_deterministic_on_seed(self):
        g1 = WorldGrid((16, 8, 16))
        g2 = WorldGrid((16, 8, 16))
        generate_world(g1, WorldParams(world_seed=7))
        generate_world(g2, WorldParams(world_seed=7))
        assert np.array_equal(g1.material, g2.material)
        assert np.array_equal(g1.energy, g2.energy)
        assert np.array_equal(g1.temperature, g2.temperature)

    def test_different_seed_different_terrain(self):
        g1 = WorldGrid((16, 8, 16))
        g2 = WorldGrid((16, 8, 16))
        generate_world(g1, WorldParams(world_seed=1))
        generate_world(g2, WorldParams(world_seed=2))
        assert not np.array_equal(g1.material, g2.material)

    def test_does_not_consume_global_rng(self):
        g = WorldGrid((16, 8, 16))
        state_before = np.random.get_state()
        generate_world(g, WorldParams(world_seed=3))
        state_after = np.random.get_state()
        assert state_before[0] == state_after[0]
        assert np.array_equal(state_before[1], state_after[1])


# ── Scene integration ─────────────────────────────────────────────────────────

class TestSceneTerrain:
    def test_scene_generates_world_when_enabled(self):
        scene = SimScene(WorldParams(
            grid_size=(16, 8, 16), generate_world=True, world_seed=5,
        ))
        assert (scene.world.material == MATERIAL_ORGANIC).any()

    def test_scene_does_not_generate_by_default(self):
        scene = SimScene(WorldParams(grid_size=(16, 8, 16)))
        assert np.all(scene.world.material == MATERIAL_AIR)

    def test_babies_spawn_on_surface(self):
        params = WorldParams(grid_size=(16, 8, 16), generate_world=True, world_seed=5)
        scene = SimScene(params)
        scene.spawn_babies(count=3)
        for baby in scene.babies:
            x, z = int(baby.position[0]), int(baby.position[2])
            assert int(baby.position[1]) == scene._surface_y(x, z)

    def test_terrain_round_trips_through_persistence(self):
        params = WorldParams(grid_size=(16, 8, 16), generate_world=True, world_seed=5)
        scene = SimScene(params)
        restored = SimScene.from_dict(scene.to_dict())
        assert np.array_equal(restored.world.material, scene.world.material)
        assert np.array_equal(restored.world.energy, scene.world.energy)

    def test_resume_matches_continuous_run_with_terrain(self):
        ticks = 6
        np.random.seed(7)
        continuous = SimScene(WorldParams(
            grid_size=(16, 8, 16), generate_world=True, world_seed=5,
        ))
        continuous.spawn_babies()
        Simulation(continuous, max_ticks=ticks).run()

        np.random.seed(7)
        resumed = SimScene(WorldParams(
            grid_size=(16, 8, 16), generate_world=True, world_seed=5,
        ))
        resumed.spawn_babies()
        Simulation(resumed, max_ticks=3).run()
        restored = SimScene.from_dict(resumed.to_dict())
        Simulation(restored, max_ticks=3).run()

        assert restored.tick == continuous.tick == ticks
        assert np.array_equal(restored.world.material, continuous.world.material)
        assert np.array_equal(restored.world.energy, continuous.world.energy)
        assert np.array_equal(restored.world.temperature, continuous.world.temperature)
        assert np.array_equal(restored.world.signal, continuous.world.signal)

    def test_babies_survive_longer_with_food(self):
        fed = WorldParams(grid_size=(16, 8, 16), generate_world=True, world_seed=5)
        starved = WorldParams(grid_size=(16, 8, 16), generate_world=False)
        np.random.seed(3)
        s1 = SimScene(fed)
        s1.spawn_babies(count=2)
        Simulation(s1, max_ticks=60).run()
        np.random.seed(3)
        s2 = SimScene(starved)
        s2.spawn_babies(count=2)
        Simulation(s2, max_ticks=60).run()
        assert len(s1.alive_babies) > len(s2.alive_babies)
