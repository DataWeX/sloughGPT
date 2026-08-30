"""Tests for packages/core-py/domains/shell/simulation.py — pure logic only."""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.simulation import (
    MATERIAL_AIR,
    MATERIAL_EMBER,
    MATERIAL_LIVING,
    MATERIAL_METAL,
    MATERIAL_ORGANIC,
    MATERIAL_SIGNAL,
    MATERIAL_STONE,
    MATERIAL_WATER,
    NUM_MATERIALS,
    BabyAction,
    CellWrite,
    Entity,
    EntityType,
    Nest,
    Perception,
    Perceptron,
    SimBaby,
    SimScene,
    Simulation,
    WorldCell,
    WorldGrid,
    WorldParams,
    cell_update_combustion,
    cell_update_conduction,
    cell_update_default,
    cell_update_diffusion,
    cell_update_ember,
    cell_update_energy_conservation,
    cell_update_living,
    cell_update_metabolism,
    cell_update_temperature,
    cell_update_waves,
    cell_update_water,
    generate_world,
)


# ── WorldCell ────────────────────────────────────────────────────────────────

class TestWorldCell:
    def test_defaults(self):
        c = WorldCell()
        assert c.material == MATERIAL_AIR
        assert c.energy == 0.0
        assert c.temperature == 20.0

    def test_custom(self):
        c = WorldCell(material=MATERIAL_WATER, energy=5.0, temperature=100.0)
        assert c.material == MATERIAL_WATER
        assert c.energy == 5.0
        assert c.temperature == 100.0


# ── WorldGrid ────────────────────────────────────────────────────────────────

class TestWorldGrid:
    def test_init_default(self):
        g = WorldGrid()
        assert g.size == (64, 32, 64)
        assert g.total == 64 * 32 * 64
        assert g.material.shape == (g.total,)
        assert g.energy.shape == (g.total,)
        assert g.temperature.shape == (g.total,)
        assert g.signal.shape == (g.total,)

    def test_init_small(self):
        g = WorldGrid((4, 3, 5))
        assert g.nx == 4 and g.ny == 3 and g.nz == 5
        assert g.total == 60

    def test_reset(self):
        g = WorldGrid((2, 2, 2))
        g.energy[0] = 99.0
        g.reset()
        assert g.material[0] == MATERIAL_AIR
        assert g.energy[0] == 0.0
        assert g.temperature[0] == 20.0
        assert g.signal[0] == 0.0

    def test_idx_wraps(self):
        g = WorldGrid((4, 3, 5))
        assert g.idx(0, 0, 0) == 0
        assert g.idx(4, 0, 0) == g.idx(0, 0, 0)
        assert g.idx(-1, 0, 0) == g.idx(3, 0, 0)

    def test_coords_roundtrip(self):
        g = WorldGrid((4, 3, 5))
        for x in range(4):
            for y in range(3):
                for z in range(5):
                    idx = g.idx(x, y, z)
                    assert g.coords(idx) == (x, y, z)

    def test_get_set_cell(self):
        g = WorldGrid((4, 3, 5))
        cell = WorldCell(material=MATERIAL_WATER, energy=10.0, temperature=50.0)
        g.set_cell(1, 2, 3, cell)
        got = g.get_cell(1, 2, 3)
        assert got.material == MATERIAL_WATER
        assert got.energy == 10.0
        assert got.temperature == 50.0

    def test_place_material(self):
        g = WorldGrid((4, 3, 5))
        g.place_material(1, 2, 3, MATERIAL_SIGNAL, energy=5.0)
        i = g.idx(1, 2, 3)
        assert g.material[i] == MATERIAL_SIGNAL
        assert g.energy[i] == 5.0
        assert g.signal[i] == 5.0

    def test_place_material_non_signal_no_signal(self):
        g = WorldGrid((4, 3, 5))
        g.place_material(1, 2, 3, MATERIAL_ORGANIC, energy=5.0)
        assert g.signal[g.idx(1, 2, 3)] == 0.0

    def test_write_cell_success(self):
        g = WorldGrid((4, 3, 5))
        assert g.write_cell(2, 1, 3, MATERIAL_EMBER, energy=7.0) is True
        i = g.idx(2, 1, 3)
        assert g.material[i] == MATERIAL_EMBER
        assert g.energy[i] == 7.0

    def test_write_cell_out_of_bounds(self):
        g = WorldGrid((4, 3, 5))
        assert g.write_cell(10, 0, 0, MATERIAL_AIR) is False

    def test_write_cell_signal_adds_energy(self):
        g = WorldGrid((4, 3, 5))
        g.signal[g.idx(1, 1, 1)] = 3.0
        g.write_cell(1, 1, 1, MATERIAL_SIGNAL, energy=2.0)
        assert g.signal[g.idx(1, 1, 1)] == pytest.approx(5.0)

    def test_get_nearby_cells(self):
        g = WorldGrid((8, 8, 8))
        g.place_material(4, 4, 4, MATERIAL_ORGANIC, energy=10.0, temperature=30.0)
        cells = g.get_nearby_cells(4, 4, 4, 1.5)
        assert cells["count"] > 0
        assert cells["material"].shape == (cells["count"],)
        assert cells["energy"].shape == (cells["count"],)
        assert cells["distance"].shape == (cells["count"],)

    def test_get_nearby_cells_zero_radius(self):
        g = WorldGrid((8, 8, 8))
        cells = g.get_nearby_cells(0, 0, 0, 0.0)
        assert cells["count"] == 1

    def test_total_energy(self):
        g = WorldGrid((2, 2, 2))
        g.energy[0] = 5.0
        g.energy[3] = 3.0
        assert g.total_energy == pytest.approx(8.0)

    def test_total_signal(self):
        g = WorldGrid((2, 2, 2))
        g.signal[0] = 2.0
        g.signal[7] = 1.0
        assert g.total_signal == pytest.approx(3.0)

    def test_to_dict_from_dict_roundtrip(self):
        g = WorldGrid((4, 3, 5))
        g.place_material(1, 2, 3, MATERIAL_ORGANIC, energy=42.0, temperature=37.0)
        g.signal[g.idx(0, 0, 0)] = 9.0
        d = g.to_dict()
        g2 = WorldGrid.from_dict(d)
        assert g2.size == g.size
        np.testing.assert_array_equal(g2.material, g.material)
        np.testing.assert_array_almost_equal(g2.energy, g.energy)
        np.testing.assert_array_almost_equal(g2.temperature, g.temperature)
        np.testing.assert_array_almost_equal(g2.signal, g.signal)

    def test_from_dict_size_mismatch_raises(self):
        g = WorldGrid((2, 2, 2))
        d = g.to_dict()
        d["material"] = [0] * 100
        with pytest.raises(ValueError, match="does not match"):
            WorldGrid.from_dict(d)


# ── Cell update functions ───────────────────────────────────────────────────

class TestCellUpdateDiffusion:
    def test_conserves_total_energy(self):
        g = WorldGrid((4, 4, 4))
        g.energy[0] = 100.0
        g.energy[63] = 50.0
        params = WorldParams(grid_size=(4, 4, 4), diffusion_rate=0.1)
        before = g.total_energy
        cell_update_diffusion(g, params)
        assert g.total_energy == pytest.approx(before, abs=1e-4)

    def test_spreads_energy(self):
        g = WorldGrid((8, 8, 8))
        center = g.idx(4, 4, 4)
        g.energy[center] = 100.0
        params = WorldParams(grid_size=(8, 8, 8), diffusion_rate=0.1)
        cell_update_diffusion(g, params)
        assert g.energy[center] < 100.0
        total_nearby = sum(g.energy[g.idx(4+dx, 4+dy, 4+dz)]
                          for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                          for dz in (-1, 0, 1))
        assert total_nearby > 0.0


class TestCellUpdateWaves:
    def test_signal_propagates(self):
        g = WorldGrid((8, 8, 8))
        center = g.idx(4, 4, 4)
        g.signal[center] = 10.0
        params = WorldParams(grid_size=(8, 8, 8), wave_speed=0.5, signal_decay=0.0)
        before_total = g.total_signal
        cell_update_waves(g, params)
        assert g.total_signal == pytest.approx(before_total, abs=1e-4)
        neighbor = g.idx(5, 4, 4)
        assert g.signal[neighbor] > 0.0


class TestCellUpdateCombustion:
    def test_organic_ignites_above_threshold(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.material[i] = MATERIAL_ORGANIC
        g.temperature[i] = 150.0
        params = WorldParams(ignition_temp=100.0, burn_temp=150.0)
        cell_update_combustion(g, params)
        assert g.material[i] == MATERIAL_EMBER

    def test_organic_below_threshold_stays(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.material[i] = MATERIAL_ORGANIC
        g.temperature[i] = 50.0
        params = WorldParams(ignition_temp=100.0)
        cell_update_combustion(g, params)
        assert g.material[i] == MATERIAL_ORGANIC


class TestCellUpdateMetabolism:
    def test_organic_rotts(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.material[i] = MATERIAL_ORGANIC
        g.energy[i] = 100.0
        params = WorldParams(organic_metabolism=0.1)
        cell_update_metabolism(g, params)
        assert g.energy[i] == pytest.approx(90.0)

    def test_zero_metabolism_no_change(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.material[i] = MATERIAL_ORGANIC
        g.energy[i] = 100.0
        params = WorldParams(organic_metabolism=0.0)
        cell_update_metabolism(g, params)
        assert g.energy[i] == 100.0


class TestCellUpdateEmber:
    def test_ember_burns_and_cools(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.material[i] = MATERIAL_EMBER
        g.energy[i] = 100.0
        g.temperature[i] = 200.0
        params = WorldParams(
            ember_heat_rate=0.5, ember_energy_fraction=0.5,
            heat_to_temp=1.0, burn_temp=150.0,
        )
        cell_update_ember(g, params)
        assert g.energy[i] < 100.0
        ni = g.idx(2, 1, 1)
        assert g.energy[ni] > 0.0

    def test_exhausted_becomes_stone(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.material[i] = MATERIAL_EMBER
        g.energy[i] = 0.001
        params = WorldParams(ember_heat_rate=1.0, ember_energy_fraction=0.5)
        cell_update_ember(g, params)
        assert g.material[i] == MATERIAL_STONE


class TestCellUpdateLiving:
    def test_living_grows_organic(self):
        g = WorldGrid((8, 8, 8))
        li = g.idx(4, 4, 4)
        g.material[li] = MATERIAL_LIVING
        g.energy[li] = 200.0
        params = WorldParams(
            living_growth_rate=0.1, living_growth_cost=2.0,
            growth_transfer_fraction=0.8,
        )
        cell_update_living(g, params)
        grew = False
        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                           (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            nx = (4 + dx) % 8
            ny = (4 + dy) % 8
            nz = (4 + dz) % 8
            ni = g.idx(nx, ny, nz)
            if g.material[ni] == MATERIAL_ORGANIC:
                grew = True
                break
        assert grew


class TestCellUpdateWater:
    def test_water_dampens_signal(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.material[i] = MATERIAL_WATER
        g.signal[i] = 10.0
        params = WorldParams(water_signal_dampen=0.5, water_cool_rate=0.0,
                             ambient_temp=20.0)
        cell_update_water(g, params)
        assert g.signal[i] == pytest.approx(5.0)

    def test_water_cools_temperature(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.material[i] = MATERIAL_WATER
        g.temperature[i] = 80.0
        params = WorldParams(water_signal_dampen=0.0, water_cool_rate=0.1,
                             ambient_temp=20.0)
        cell_update_water(g, params)
        assert g.temperature[i] == pytest.approx(80.0 + (20.0 - 80.0) * 0.1)


class TestCellUpdateConduction:
    def test_metal_conducts(self):
        g = WorldGrid((8, 8, 8))
        i = g.idx(4, 4, 4)
        j = g.idx(5, 4, 4)
        g.material[i] = MATERIAL_METAL
        g.material[j] = MATERIAL_STONE
        g.energy[i] = 100.0
        g.energy[j] = 0.0
        params = WorldParams(grid_size=(8, 8, 8), metal_conduction_boost=3.0,
                             diffusion_rate=0.1)
        cell_update_conduction(g, params)
        assert g.energy[i] < 100.0
        assert g.energy[j] > 0.0


class TestCellUpdateTemperature:
    def test_relaxes_toward_ambient(self):
        g = WorldGrid((4, 4, 4))
        g.temperature[0] = 100.0
        params = WorldParams(ambient_temp=20.0, ambient_cooling=0.1)
        cell_update_temperature(g, params)
        assert g.temperature[0] == pytest.approx(100.0 + (20.0 - 100.0) * 0.1)


class TestCellUpdateEnergyConservation:
    def test_energy_decreases(self):
        g = WorldGrid((4, 4, 4))
        g.energy[0] = 100.0
        params = WorldParams(energy_loss=0.05)
        cell_update_energy_conservation(g, params)
        assert g.energy[0] == pytest.approx(95.0)

    def test_energy_clamps_nonnegative(self):
        g = WorldGrid((4, 4, 4))
        g.energy[0] = -5.0
        params = WorldParams(energy_loss=0.0)
        cell_update_energy_conservation(g, params)
        assert g.energy[0] == 0.0

    def test_temperature_clamps(self):
        g = WorldGrid((4, 4, 4))
        g.temperature[0] = 2000.0
        params = WorldParams(energy_loss=0.0)
        cell_update_energy_conservation(g, params)
        assert g.temperature[0] == 1000.0


class TestGenerateWorld:
    def test_generates_stone_floor(self):
        g = WorldGrid((16, 8, 16))
        params = WorldParams(grid_size=(16, 8, 16))
        generate_world(g, params, seed=42)
        stone_count = 0
        for x in range(16):
            for z in range(16):
                mat = g.material[g.idx(x, 0, z)]
                if mat == MATERIAL_STONE:
                    stone_count += 1
        assert stone_count > 0

    def test_deterministic(self):
        g1 = WorldGrid((8, 4, 8))
        g2 = WorldGrid((8, 4, 8))
        params = WorldParams(grid_size=(8, 4, 8))
        generate_world(g1, params, seed=42)
        generate_world(g2, params, seed=42)
        np.testing.assert_array_equal(g1.material, g2.material)

    def test_different_seeds_differ(self):
        g1 = WorldGrid((8, 4, 8))
        g2 = WorldGrid((8, 4, 8))
        params = WorldParams(grid_size=(8, 4, 8))
        generate_world(g1, params, seed=1)
        generate_world(g2, params, seed=2)
        assert not np.array_equal(g1.material, g2.material)


# ── Entity ──────────────────────────────────────────────────────────────────

class TestEntity:
    def test_defaults(self):
        e = Entity()
        assert e.id == 0
        assert e.energy == 100.0
        assert e.alive is True

    def test_distance_to(self):
        a = Entity(position=np.array([0.0, 0.0, 0.0]))
        b = Entity(position=np.array([3.0, 4.0, 0.0]))
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_to_point(self):
        e = Entity(position=np.array([1.0, 0.0, 0.0]))
        assert e.distance_to_point(np.array([4.0, 0.0, 0.0])) == pytest.approx(3.0)

    def test_to_dict_from_dict(self):
        e = Entity(id=7, position=np.array([1.0, 2.0, 3.0]),
                   energy=50.0, entity_type=EntityType.AGENT, alive=True)
        d = e.to_dict()
        e2 = Entity.from_dict(d)
        assert e2.id == 7
        np.testing.assert_array_almost_equal(e2.position, [1.0, 2.0, 3.0])
        assert e2.energy == 50.0
        assert e2.entity_type == EntityType.AGENT
        assert e2.alive is True


# ── Nest ────────────────────────────────────────────────────────────────────

class TestNest:
    def test_distance_to_point(self):
        n = Nest(id=1, position=np.array([0.0, 0.0, 0.0]),
                 stored_energy=50.0, owner_group_id=0)
        assert n.distance_to_point(np.array([3.0, 4.0, 0.0])) == pytest.approx(5.0)

    def test_to_dict_from_dict(self):
        n = Nest(id=5, position=np.array([1.0, 2.0, 3.0]),
                 stored_energy=100.0, owner_group_id=2, alive=True)
        d = n.to_dict()
        n2 = Nest.from_dict(d)
        assert n2.id == 5
        np.testing.assert_array_almost_equal(n2.position, [1.0, 2.0, 3.0])
        assert n2.stored_energy == 100.0
        assert n2.owner_group_id == 2
        assert n2.alive is True


# ── Perceptron ──────────────────────────────────────────────────────────────

class TestPerceptron:
    def test_forward_shape(self):
        p = Perceptron(5, 3)
        x = np.zeros(5, dtype=np.float32)
        out = p.forward(x)
        assert out.shape == (3,)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_sigmoid_output_range(self):
        p = Perceptron(3, 2)
        x = np.random.randn(3).astype(np.float32)
        out = p.forward(x)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_hidden_layer(self):
        p = Perceptron(4, 2, hidden_units=8)
        assert p.hidden_units == 8
        assert p.H.shape == (4, 8)
        assert p.bh.shape == (8,)
        x = np.zeros(4, dtype=np.float32)
        out = p.forward(x)
        assert out.shape == (2,)

    def test_update_changes_weights(self):
        p = Perceptron(3, 2)
        W_before = p.W.copy()
        x = np.ones(3, dtype=np.float32)
        error = np.array([1.0, -1.0], dtype=np.float32)
        p.update(x, error, lr=0.1)
        assert not np.allclose(p.W, W_before)

    def test_to_dict_from_dict(self):
        p = Perceptron(4, 2)
        d = p.to_dict()
        p2 = Perceptron.from_dict(d)
        np.testing.assert_array_almost_equal(p2.W, p.W)
        np.testing.assert_array_almost_equal(p2.b, p.b)

    def test_to_dict_from_dict_with_hidden(self):
        p = Perceptron(4, 2, hidden_units=6)
        d = p.to_dict()
        p2 = Perceptron.from_dict(d)
        np.testing.assert_array_almost_equal(p2.W, p.W)
        np.testing.assert_array_almost_equal(p2.H, p.H)
        np.testing.assert_array_almost_equal(p2.bh, p.bh)
        assert p2.hidden_units == 6

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(42)
        p1 = Perceptron(3, 2)
        p1.W = rng.standard_normal(p1.W.shape).astype(np.float32)
        rng2 = np.random.default_rng(42)
        p2 = Perceptron(3, 2)
        p2.W = rng2.standard_normal(p2.W.shape).astype(np.float32)
        x = np.ones(3, dtype=np.float32)
        np.testing.assert_array_almost_equal(p1.forward(x), p2.forward(x))


# ── SimBaby ─────────────────────────────────────────────────────────────────

class TestSimBaby:
    def test_init_defaults(self):
        b = SimBaby(position=np.array([5.0, 5.0, 5.0]), initial_energy=100.0)
        assert b.energy == 100.0
        assert b.alive is True
        assert b.entity.entity_type == EntityType.AGENT
        assert b.position[0] == 5.0

    def test_perceptrons_created(self):
        b = SimBaby()
        assert b.perceptron_cells is not None
        assert b.perceptron_body is not None
        assert b.perceptron_entity is not None
        assert b.perceptron_move is not None

    def test_perceptrons_with_optional_channels(self):
        p = WorldParams(message_enabled=True, teaching_enabled=True,
                        predation_enabled=True, territoriality_enabled=True,
                        lifecycle_enabled=True, specialization_enabled=True)
        b = SimBaby(params=p)
        assert b.perceptron_message is not None
        assert b.perceptron_teach is not None
        assert b.perceptron_predation is not None
        assert b.perceptron_territory is not None
        assert b.perceptron_reproduce is not None
        assert b.perceptron_role is not None

    def test_perceive(self):
        g = WorldGrid((8, 8, 8))
        b = SimBaby(position=np.array([4.0, 4.0, 4.0]), initial_energy=100.0)
        p = b.perceive(g)
        assert isinstance(p, Perception)
        assert p.nearby_cells["count"] > 0
        assert b._last_perception is p

    def test_feel_positive(self):
        b = SimBaby(initial_energy=100.0)
        b.entity.energy = 110.0
        assert b.feel(100.0) == pytest.approx(10.0)

    def test_feel_negative(self):
        b = SimBaby(initial_energy=100.0)
        b.entity.energy = 90.0
        assert b.feel(100.0) == pytest.approx(-10.0)

    def test_react_energy_above_threshold(self):
        g = WorldGrid((8, 8, 8))
        b = SimBaby(position=np.array([4.0, 4.0, 4.0]), initial_energy=100.0)
        p = b.perceive(g)
        action = b.react(p, 0.0)
        assert isinstance(action, BabyAction)
        assert action.writes is not None

    def test_react_energy_below_threshold_no_writes(self):
        b = SimBaby(initial_energy=5.0)
        p = Perception(
            nearby_cells={"material": np.zeros(1), "energy": np.zeros(1),
                          "temperature": np.full(1, 20.0), "signal": np.zeros(1),
                          "distance": np.zeros(1), "count": 1},
            nearby_entities=[], agent_body={"position": [0, 0, 0], "energy": 5.0},
        )
        action = b.react(p, 0.0)
        assert len(action.writes) == 0

    def test_apply_action(self):
        g = WorldGrid((8, 8, 8))
        b = SimBaby(position=np.array([4.0, 4.0, 4.0]))
        action = BabyAction(writes=[
            CellWrite(4, 4, 4, MATERIAL_ORGANIC, 5.0),
            CellWrite(5, 4, 4, MATERIAL_WATER, 3.0),
        ])
        written = b.apply_action(action, g)
        assert written == 2
        assert g.energy[g.idx(4, 4, 4)] == pytest.approx(5.0)
        assert g.material[g.idx(5, 4, 4)] == MATERIAL_WATER

    def test_absorb_energy(self):
        g = WorldGrid((8, 8, 8))
        b = SimBaby(position=np.array([4.0, 4.0, 4.0]))
        g.material[g.idx(4, 4, 4)] = MATERIAL_ORGANIC
        g.energy[g.idx(4, 4, 4)] = 10.0
        absorbed = b.absorb_energy(g, radius=1)
        assert absorbed > 0.0

    def test_absorb_no_organic(self):
        g = WorldGrid((8, 8, 8))
        b = SimBaby(position=np.array([4.0, 4.0, 4.0]))
        absorbed = b.absorb_energy(g, radius=1)
        assert absorbed == 0.0

    def test_share_energy(self):
        a = SimBaby(initial_energy=200.0)
        b = SimBaby(initial_energy=50.0)
        transferred = a.share_energy(b)
        assert transferred > 0.0
        assert a.energy < 200.0
        assert b.energy > 50.0

    def test_share_energy_dead_target(self):
        a = SimBaby(initial_energy=200.0)
        b = SimBaby(initial_energy=50.0)
        b.entity.alive = False
        assert a.share_energy(b) == 0.0

    def test_contest_energy_weaker(self):
        a = SimBaby(initial_energy=200.0)
        b = SimBaby(initial_energy=50.0)
        taken = a.contest_energy(b)
        assert taken > 0.0
        assert a.energy > 200.0
        assert b.energy < 50.0

    def test_contest_energy_equal_or_stronger(self):
        a = SimBaby(initial_energy=100.0)
        b = SimBaby(initial_energy=100.0)
        assert a.contest_energy(b) == 0.0
        b.entity.energy = 150.0
        assert a.contest_energy(b) == 0.0

    def test_social_step_cooperate(self):
        a = SimBaby(initial_energy=200.0, params=WorldParams(cooperate_threshold=0.3))
        b = SimBaby(initial_energy=50.0)
        result = a.social_step(b)
        assert result["act"] in ("cooperate", "none", "contest")
        assert result["energy_moved"] >= 0.0

    def test_social_step_dead_neighbor(self):
        a = SimBaby(initial_energy=200.0)
        b = SimBaby(initial_energy=50.0)
        b.entity.alive = False
        result = a.social_step(b)
        assert result["act"] == "none"
        assert result["energy_moved"] == 0.0

    def test_alive_property(self):
        b = SimBaby(initial_energy=100.0)
        assert b.alive is True
        b.entity.alive = False
        assert b.alive is False
        b.entity.alive = True
        b.entity.energy = 0.0
        assert b.alive is False

    def test_info(self):
        b = SimBaby(initial_energy=100.0)
        info = b.info()
        assert "id" in info
        assert "energy" in info
        assert "alive" in info
        assert "ticks" in info

    def test_to_dict_from_dict(self):
        b = SimBaby(position=np.array([3.0, 2.0, 1.0]), initial_energy=80.0,
                    params=WorldParams(), group_id=2)
        d = b.to_dict()
        b2 = SimBaby.from_dict(d, params=WorldParams())
        assert b2.energy == pytest.approx(80.0)
        assert b2.group_id == 2
        np.testing.assert_array_almost_equal(b2.position, [3.0, 2.0, 1.0])

    def test_spawn_child(self):
        parent = SimBaby(position=np.array([4.0, 4.0, 4.0]), initial_energy=200.0,
                         group_id=1)
        child = parent.spawn_child(np.array([5.0, 4.0, 4.0]))
        assert child.energy == pytest.approx(parent.params.birth_cost)
        assert child.group_id == 1

    def test_decide_move_above_threshold(self):
        b = SimBaby(position=np.array([4.0, 4.0, 4.0]), initial_energy=100.0)
        g = WorldGrid((8, 8, 8))
        p = b.perceive(g)
        direction = b.decide_move(p)
        assert direction is not None
        assert direction.shape == (3,)

    def test_decide_move_below_threshold(self):
        b = SimBaby(initial_energy=1.0,
                    params=WorldParams(move_threshold=10.0))
        p = Perception(
            nearby_cells={"material": np.zeros(1), "energy": np.zeros(1),
                          "temperature": np.full(1, 20.0), "signal": np.zeros(1),
                          "distance": np.zeros(1), "count": 1},
            nearby_entities=[], agent_body={"position": [0, 0, 0], "energy": 1.0},
        )
        assert b.decide_move(p) is None

    def test_decide_role_none_when_disabled(self):
        b = SimBaby()
        b.perceptron_role = None
        assert b.decide_role() == 0.0

    def test_decide_reproduce_none_when_disabled(self):
        b = SimBaby()
        b.perceptron_reproduce = None
        assert b.decide_reproduce() == 0.0

    def test_hunt(self):
        predator = SimBaby(initial_energy=100.0)
        predator.perceptron_predation = Perceptron(5, 1)
        prey = SimBaby(initial_energy=50.0)
        gained = predator.hunt(prey)
        assert gained == pytest.approx(50.0)
        assert prey.alive is False
        assert predator.energy == pytest.approx(150.0)

    def test_hunt_dead_prey(self):
        predator = SimBaby(initial_energy=100.0)
        predator.perceptron_predation = Perceptron(5, 1)
        prey = SimBaby(initial_energy=50.0)
        prey.entity.alive = False
        assert predator.hunt(prey) == 0.0

    def test_defend(self):
        defender = SimBaby(initial_energy=100.0, params=WorldParams(defend_take_fraction=0.5))
        trespasser = SimBaby(initial_energy=80.0)
        defender.perceptron_territory = Perceptron(5, 1)
        gained = defender.defend(trespasser, np.array([0.0, 0.0, 0.0]))
        assert gained >= 0.0
        assert defender.energy >= 100.0

    def test_defend_dead_trespasser(self):
        defender = SimBaby(initial_energy=100.0)
        defender.perceptron_territory = Perceptron(5, 1)
        trespasser = SimBaby(initial_energy=80.0)
        trespasser.entity.alive = False
        assert defender.defend(trespasser, np.array([0.0, 0.0, 0.0])) == 0.0

    def test_recall_memories(self):
        b = SimBaby(initial_energy=100.0)
        g = WorldGrid((8, 8, 8))
        p = b.perceive(g)
        b.react(p, 0.0)
        b.learn(1.0)
        memories = b.recall_memories(k=1)
        assert len(memories) >= 0


# ── SimScene ────────────────────────────────────────────────────────────────

class TestSimScene:
    def test_init(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8)))
        assert scene.world.size == (8, 8, 8)
        assert len(scene.babies) == 0

    def test_add_baby(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8)))
        b = SimBaby(initial_energy=100.0)
        scene.add_baby(b)
        assert len(scene.babies) == 1
        assert len(scene.entities) == 1

    def test_alive_babies(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8)))
        b1 = SimBaby(initial_energy=100.0)
        b2 = SimBaby(initial_energy=100.0)
        scene.add_baby(b1)
        scene.add_baby(b2)
        b1.entity.alive = False
        assert len(scene.alive_babies) == 1

    def test_nearby_babies(self):
        scene = SimScene(params=WorldParams(grid_size=(16, 16, 16)))
        b1 = SimBaby(position=np.array([8.0, 8.0, 8.0]))
        b2 = SimBaby(position=np.array([9.0, 8.0, 8.0]))
        b3 = SimBaby(position=np.array([15.0, 15.0, 15.0]))
        scene.add_baby(b1)
        scene.add_baby(b2)
        scene.add_baby(b3)
        nearby = scene.nearby_babies(b1.position, radius=2.0, exclude_id=b1.entity.id)
        assert len(nearby) == 1
        assert nearby[0].entity.id == b2.entity.id

    def test_nearest_nest(self):
        scene = SimScene(params=WorldParams(grid_size=(16, 16, 16)))
        scene.nests.append(Nest(id=1, position=np.array([8.0, 8.0, 8.0]),
                                stored_energy=50.0, owner_group_id=0))
        nest = scene.nearest_nest(np.array([7.0, 8.0, 8.0]), radius=2.0)
        assert nest is not None
        assert nest.id == 1

    def test_nearest_nest_filters_by_group(self):
        scene = SimScene(params=WorldParams(grid_size=(16, 16, 16)))
        scene.nests.append(Nest(id=1, position=np.array([8.0, 8.0, 8.0]),
                                stored_energy=50.0, owner_group_id=0))
        nest = scene.nearest_nest(np.array([8.0, 8.0, 8.0]), radius=2.0, group_id=1)
        assert nest is None

    def test_deposit_memory(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8),
                                            memory_enabled=True, memory_deposit=2))
        b = SimBaby(initial_energy=100.0)
        scene.add_baby(b)
        deposited = scene.deposit_memory(b)
        assert deposited >= 0

    def test_draw_nest(self):
        scene = SimScene(params=WorldParams(grid_size=(16, 16, 16),
                                            structure_enabled=True,
                                            nest_draw_rate=1.0))
        scene.nests.append(Nest(id=1, position=np.array([8.0, 8.0, 8.0]),
                                stored_energy=50.0, owner_group_id=0))
        b = SimBaby(position=np.array([8.0, 8.0, 8.0]), initial_energy=50.0)
        drawn = scene.draw_nest(b)
        assert drawn > 0.0

    def test_draw_nest_full_energy(self):
        scene = SimScene(params=WorldParams(grid_size=(16, 16, 16),
                                            structure_enabled=True))
        scene.nests.append(Nest(id=1, position=np.array([8.0, 8.0, 8.0]),
                                stored_energy=50.0, owner_group_id=0))
        b = SimBaby(position=np.array([8.0, 8.0, 8.0]), initial_energy=200.0)
        assert scene.draw_nest(b) == 0.0

    def test_raid_nest_disabled(self):
        scene = SimScene(params=WorldParams(grid_size=(16, 16, 16)))
        b = SimBaby(initial_energy=50.0)
        assert scene.raid_nest(b) == 0.0

    def test_update_nests(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8),
                                            structure_enabled=True,
                                            nest_decay=0.1))
        scene.nests.append(Nest(id=1, position=np.array([4.0, 4.0, 4.0]),
                                stored_energy=10.0, owner_group_id=0))
        scene.update_nests()
        assert scene.nests[0].stored_energy == pytest.approx(9.0)

    def test_info(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8)))
        info = scene.info()
        assert "tick" in info
        assert "total_energy" in info
        assert "alive_babies" in info

    def test_to_dict_from_dict(self):
        scene = SimScene(params=WorldParams(grid_size=(4, 4, 4)))
        b = SimBaby(position=np.array([2.0, 2.0, 2.0]), initial_energy=80.0)
        scene.add_baby(b)
        d = scene.to_dict()
        scene2 = SimScene.from_dict(d)
        assert len(scene2.babies) == 1
        assert scene2.babies[0].energy == pytest.approx(80.0)


# ── Simulation ──────────────────────────────────────────────────────────────

class TestSimulation:
    def test_step_runs(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8)))
        scene.add_baby(SimBaby(position=np.array([4.0, 4.0, 4.0]),
                               initial_energy=100.0))
        sim = Simulation(scene, max_ticks=1)
        results = sim.step()
        assert len(results) == 1
        assert results[0]["tick"] == 1

    def test_run_multiple_ticks(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8)))
        scene.add_baby(SimBaby(position=np.array([4.0, 4.0, 4.0]),
                               initial_energy=100.0))
        sim = Simulation(scene, max_ticks=5)
        results = sim.run()
        assert len(results) > 0

    def test_no_ticks(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8)))
        scene.add_baby(SimBaby(position=np.array([4.0, 4.0, 4.0]),
                               initial_energy=100.0))
        sim = Simulation(scene, max_ticks=0)
        results = sim.run()
        assert len(results) == 0

    def test_summary(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8)))
        scene.add_baby(SimBaby(position=np.array([4.0, 4.0, 4.0]),
                               initial_energy=100.0))
        sim = Simulation(scene, max_ticks=3)
        sim.run()
        s = sim.summary()
        assert s["total_ticks"] == 3
        assert s["total_baby_ticks"] > 0
        assert "cooperations" in s
        assert "contests" in s
        assert "deaths" in s

    def test_summary_empty(self):
        scene = SimScene(params=WorldParams(grid_size=(8, 8, 8)))
        sim = Simulation(scene, max_ticks=1)
        s = sim.summary()
        assert s["total_ticks"] == 0
